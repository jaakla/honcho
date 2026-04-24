import asyncio
import logging
import re
import time
from enum import Enum
from functools import cache
from inspect import cleandoc as c
from typing import TypedDict

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.cache.client import cache as cache_client
from src.config import settings
from src.crud.session import session_cache_key
from src.dependencies import tracked_db
from src.exceptions import ResourceNotFoundException
from src.models import Message
from src.telemetry import prometheus_metrics
from src.telemetry.events import AgentToolSummaryCreatedEvent, emit
from src.telemetry.logging import accumulate_metric, conditional_observe
from src.telemetry.prometheus.metrics import (
    DeriverComponents,
    DeriverTaskTypes,
    TokenTypes,
)
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call
from src.utils.formatting import utc_now_iso
from src.utils.tokens import estimate_tokens, track_deriver_input_tokens

from .. import crud, models

logger = logging.getLogger(__name__)


# TypedDict definitions for summary data
class Summary(TypedDict):
    """
    A summary object. Stored in session metadata and used in a session's get_context.

    Attributes:
        content: The summary text.
        message_id: The primary key ID of the message that this summary covers up to.
        summary_type: The type of summary (short or long).
        created_at: The timestamp of when the summary was created (ISO format string).
        token_count: The number of tokens in the summary text.
    """

    content: str
    message_id: int
    summary_type: str
    created_at: str
    token_count: int
    message_public_id: str


class SummaryRefreshMessage(TypedDict):
    """Detached-safe message snapshot used for summary refresh."""

    id: int
    public_id: str
    peer_name: str
    content: str
    token_count: int


class SummarySegment(TypedDict):
    """Intermediate summary segment used during balanced refresh rebuilds."""

    content: str
    token_count: int
    message_id: int
    message_public_id: str


def to_schema_summary(s: Summary) -> schemas.Summary:
    return schemas.Summary(
        content=s["content"],
        message_id=s["message_id"],
        summary_type=s["summary_type"],
        created_at=s["created_at"],
        token_count=s["token_count"],
        message_public_id=s.get("message_public_id", ""),
    )


# Export the public functions
__all__ = [
    "refresh_session_summaries",
    "get_summary",
    "get_both_summaries",
    "get_summarized_history",
    "get_session_context",
    "get_session_context_formatted",
    "SummaryType",
    "Summary",
    "to_schema_summary",
]


# Configuration constants for summaries
MESSAGES_PER_SHORT_SUMMARY = settings.SUMMARY.MESSAGES_PER_SHORT_SUMMARY
MESSAGES_PER_LONG_SUMMARY = settings.SUMMARY.MESSAGES_PER_LONG_SUMMARY
MIN_SHORT_SUMMARY_WORDS = 120
SHORT_SUMMARY_WORDS_PER_OUTPUT_TOKEN = 0.25
LONG_SUMMARY_WORDS_PER_OUTPUT_TOKEN = 0.35

SUMMARIES_KEY = "summaries"


# The types of summary to store in the session metadata
class SummaryType(Enum):
    SHORT = "honcho_chat_summary_short"
    LONG = "honcho_chat_summary_long"


def _hit_output_length_limit(finish_reasons: list[str]) -> bool:
    """Return True when the model stopped because it ran out of output budget."""
    for reason in finish_reasons:
        normalized = reason.strip().lower()
        if normalized == "length":
            return True
        if "max_tokens" in normalized or "max_output_tokens" in normalized:
            return True
        if "token" in normalized and "max" in normalized:
            return True
    return False


def _trim_partial_summary(summary_text: str) -> str:
    """Trim capped summaries back to a clean word boundary."""
    trimmed = summary_text.rstrip()
    if not trimmed:
        return trimmed

    if trimmed[-1] in ".!?)]}\"'":
        return trimmed

    if trimmed[-1].isalnum():
        boundary = max(
            trimmed.rfind(" "),
            trimmed.rfind("\n"),
            trimmed.rfind("\t"),
        )
        if boundary > 0:
            trimmed = trimmed[:boundary].rstrip()
        else:
            trimmed = re.sub(r"\w+$", "", trimmed).rstrip()

    if trimmed and trimmed[-1] not in ".!?":
        return f"{trimmed}..."
    return trimmed


def short_summary_prompt(
    formatted_messages: str,
    output_words: int,
    previous_summary_text: str,
) -> str:
    """Generate the short summary prompt."""
    return c(f"""
You are a system that summarizes parts of a conversation to create a concise and accurate summary. Focus on capturing:

1. Key facts and information shared (**Capture as many explicit facts as possible**)
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed
5. Decisions made, open questions, and pending follow-up items

If there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.

LANGUAGE PRESERVATION:
- Write the summary in the same primary language as the conversation and previous summary.
- Do NOT translate the conversation into English or any other language.
- If the conversation is mixed-language, keep the dominant language of the conversation and preserve quoted phrases in their original language when useful.

Provide a compact but complete factual summary that captures the conversation well enough to continue it later without rereading the messages. Preserve names, numbers, decisions, requests, and unresolved items whenever they matter. Prefer a thorough chronological narrative over a list of bullet points.

Return only the summary without any explanation or meta-commentary.

<previous_summary>
{previous_summary_text}
</previous_summary>

<conversation>
{formatted_messages}
</conversation>

Target length: about {output_words} words. It is acceptable to go somewhat shorter or longer if needed to preserve critical context, but do not waste space on minor repetition.
""")


def long_summary_prompt(
    formatted_messages: str,
    output_words: int,
    previous_summary_text: str,
) -> str:
    """Generate the long summary prompt."""
    return c(f"""
You are a system that creates thorough, comprehensive summaries of conversations. Focus on capturing:

1. Key facts and information shared (**Capture as many explicit facts as possible**)
2. User preferences, opinions, and questions
3. Important context and requests
4. Core topics discussed in detail
5. User's apparent emotional state and personality traits
6. Important themes and patterns across the conversation

If there is a previous summary, ALWAYS make your new summary inclusive of both it and the new messages, therefore capturing the ENTIRE conversation. Prioritize key facts across the entire conversation.

LANGUAGE PRESERVATION:
- Write the summary in the same primary language as the conversation and previous summary.
- Do NOT translate the conversation into English or any other language.
- If the conversation is mixed-language, keep the dominant language of the conversation and preserve quoted phrases in their original language when useful.

Provide a thorough and detailed summary that captures the essence of the conversation. Your summary should serve as a comprehensive record of the important information in this conversation. Prefer an exhaustive chronological narrative over a list of bullet points.

Return only the summary without any explanation or meta-commentary.

<previous_summary>
{previous_summary_text}
</previous_summary>

<conversation>
{formatted_messages}
</conversation>

Hard limit: {output_words} words maximum. If needed, drop lower-priority detail to stay within the limit.
""")


def short_summary_merge_prompt(
    formatted_segments: str,
    output_words: int,
) -> str:
    """Generate the short summary merge prompt for balanced refresh rebuilds."""
    return c(f"""
You are a system that merges chronological summary segments into one accurate session summary.

Each segment summarizes a different part of the same conversation. Your job is to combine them into one coherent summary without over-weighting the earliest segments.

Focus on preserving:
1. Key facts and information shared
2. Important decisions, requests, and open questions
3. Names, numbers, dates, and commitments
4. The chronological development of the conversation

Rules:
- Treat all segments as equally important evidence for the final summary.
- Preserve later developments, corrections, and reversals when they matter.
- Avoid repeating the same detail multiple times.
- Write in the same primary language as the segments.
- Return only the merged summary.

<summary_segments>
{formatted_segments}
</summary_segments>

Target length: about {output_words} words. It is acceptable to go somewhat shorter or longer if needed to preserve critical context, but do not waste space on repetition.
""")


def long_summary_merge_prompt(
    formatted_segments: str,
    output_words: int,
) -> str:
    """Generate the long summary merge prompt for balanced refresh rebuilds."""
    return c(f"""
You are a system that merges chronological summary segments into one thorough, comprehensive session summary.

Each segment summarizes a different part of the same conversation. Your job is to combine them into one coherent summary without over-weighting the earliest segments.

Focus on preserving:
1. Key facts and information shared
2. Important decisions, requests, and open questions
3. Names, numbers, dates, and commitments
4. Themes, patterns, and shifts in the conversation
5. The chronological development of the conversation, including later updates or reversals

Rules:
- Treat all segments as equally important evidence for the final summary.
- Preserve later developments, corrections, and reversals when they matter.
- Avoid repeating the same detail multiple times.
- Write in the same primary language as the segments.
- Return only the merged summary.

<summary_segments>
{formatted_segments}
</summary_segments>

Hard limit: {output_words} words maximum. If needed, drop lower-priority detail to stay within the limit.
""")


@cache
def estimate_short_summary_prompt_tokens() -> int:
    """Estimate tokens for the short summary prompt (without messages/previous_summary)."""
    try:
        return estimate_tokens(
            short_summary_prompt(
                formatted_messages="",
                output_words=0,
                previous_summary_text="",
            )
        )
    except Exception:
        # Return a rough estimate if estimation fails
        return 200


@cache
def estimate_long_summary_prompt_tokens() -> int:
    """Estimate tokens for the long summary prompt (without messages/previous_summary)."""
    try:
        return estimate_tokens(
            long_summary_prompt(
                formatted_messages="",
                output_words=0,
                previous_summary_text="",
            )
        )
    except Exception:
        # Return a rough estimate if estimation fails
        return 200


@cache
def estimate_short_summary_merge_prompt_tokens() -> int:
    """Estimate tokens for the short summary merge prompt (without segments)."""
    try:
        return estimate_tokens(
            short_summary_merge_prompt(
                formatted_segments="",
                output_words=0,
            )
        )
    except Exception:
        return 200


@cache
def estimate_long_summary_merge_prompt_tokens() -> int:
    """Estimate tokens for the long summary merge prompt (without segments)."""
    try:
        return estimate_tokens(
            long_summary_merge_prompt(
                formatted_segments="",
                output_words=0,
            )
        )
    except Exception:
        return 200


@conditional_observe(name="Create Short Summary")
async def create_short_summary(
    formatted_messages: str,
    input_tokens: int,
    previous_summary: str | None = None,
) -> HonchoLLMCallResponse[str]:
    # input_tokens indicates how many tokens the message list + previous summary take up
    # we want to optimize short summaries to be smaller than the actual content being summarized
    # so we ask the model for a conservative word count that will fit comfortably
    # within the configured output-token limit across languages/providers.
    output_words = max(
        MIN_SHORT_SUMMARY_WORDS,
        int(
            min(input_tokens, settings.SUMMARY.MAX_TOKENS_SHORT)
            * SHORT_SUMMARY_WORDS_PER_OUTPUT_TOKEN
        ),
    )

    if previous_summary:
        previous_summary_text = previous_summary
    else:
        previous_summary_text = "There is no previous summary -- the messages are the beginning of the conversation."

    prompt = short_summary_prompt(
        formatted_messages, output_words, previous_summary_text
    )

    return await honcho_llm_call(
        llm_settings=settings.SUMMARY,
        prompt=prompt,
        max_tokens=settings.SUMMARY.MAX_TOKENS_SHORT,
    )


@conditional_observe(name="Create Long Summary")
async def create_long_summary(
    formatted_messages: str,
    previous_summary: str | None = None,
) -> HonchoLLMCallResponse[str]:
    # Use a conservative words-per-token ratio so the requested target length
    # fits under the actual generation budget.
    output_words = int(
        settings.SUMMARY.MAX_TOKENS_LONG * LONG_SUMMARY_WORDS_PER_OUTPUT_TOKEN
    )

    if previous_summary:
        previous_summary_text = previous_summary
    else:
        previous_summary_text = "There is no previous summary -- the messages are the beginning of the conversation."

    prompt = long_summary_prompt(
        formatted_messages, output_words, previous_summary_text
    )

    return await honcho_llm_call(
        llm_settings=settings.SUMMARY,
        prompt=prompt,
        max_tokens=settings.SUMMARY.MAX_TOKENS_LONG,
    )


@conditional_observe(name="Merge Short Summary Segments")
async def merge_short_summary_segments(
    formatted_segments: str,
    input_tokens: int,
) -> HonchoLLMCallResponse[str]:
    output_words = max(
        MIN_SHORT_SUMMARY_WORDS,
        int(
            min(input_tokens, settings.SUMMARY.MAX_TOKENS_SHORT)
            * SHORT_SUMMARY_WORDS_PER_OUTPUT_TOKEN
        ),
    )
    prompt = short_summary_merge_prompt(formatted_segments, output_words)
    return await honcho_llm_call(
        llm_settings=settings.SUMMARY,
        prompt=prompt,
        max_tokens=settings.SUMMARY.MAX_TOKENS_SHORT,
    )


@conditional_observe(name="Merge Long Summary Segments")
async def merge_long_summary_segments(
    formatted_segments: str,
) -> HonchoLLMCallResponse[str]:
    output_words = int(
        settings.SUMMARY.MAX_TOKENS_LONG * LONG_SUMMARY_WORDS_PER_OUTPUT_TOKEN
    )
    prompt = long_summary_merge_prompt(formatted_segments, output_words)
    return await honcho_llm_call(
        llm_settings=settings.SUMMARY,
        prompt=prompt,
        max_tokens=settings.SUMMARY.MAX_TOKENS_LONG,
    )


async def summarize_if_needed(
    workspace_name: str,
    session_name: str,
    message_id: int,
    message_seq_in_session: int,
    message_public_id: str,
    configuration: schemas.ResolvedConfiguration,
) -> None:
    """
    Create short/long summaries if thresholds met.

    This function checks for both short and long summary needs independently,
    without assuming any relationship between their thresholds.

    Args:
        workspace_name: The workspace name
        session_name: The session name
        message_id: The message ID
        message_seq_in_session: The sequence number of the message in the session
        message_public_id: The public ID of the message
        configuration: The resolved configuration for the message
    """
    if configuration.summary.enabled is False:
        return

    should_create_long: bool = (
        message_seq_in_session % configuration.summary.messages_per_long_summary == 0
    )
    should_create_short: bool = (
        message_seq_in_session % configuration.summary.messages_per_short_summary == 0
    )

    if should_create_long is False and should_create_short is False:
        return

    # If both summaries need to be created, run them in parallel
    if should_create_long and should_create_short:

        async def create_long_summary_task():
            await _create_and_save_summary(
                workspace_name,
                session_name,
                message_id=message_id,
                message_seq_in_session=message_seq_in_session,
                message_public_id=message_public_id,
                summary_type=SummaryType.LONG,
                configuration=configuration,
            )
            accumulate_metric(
                f"summary_{workspace_name}_{message_id}",
                "long_summary_up_to_message",
                message_seq_in_session,
                "count",
            )

        async def create_short_summary_task():
            await _create_and_save_summary(
                workspace_name,
                session_name,
                message_id=message_id,
                message_seq_in_session=message_seq_in_session,
                message_public_id=message_public_id,
                summary_type=SummaryType.SHORT,
                configuration=configuration,
            )
            accumulate_metric(
                f"summary_{workspace_name}_{message_id}",
                "short_summary_up_to_message",
                message_seq_in_session,
                "count",
            )

        await asyncio.gather(
            create_long_summary_task(),
            create_short_summary_task(),
            return_exceptions=True,
        )
    else:
        # If only one summary needs to be created, run individually
        if should_create_long:
            await _create_and_save_summary(
                workspace_name,
                session_name,
                message_id=message_id,
                message_seq_in_session=message_seq_in_session,
                message_public_id=message_public_id,
                summary_type=SummaryType.LONG,
                configuration=configuration,
            )
            accumulate_metric(
                f"summary_{workspace_name}_{message_id}",
                "long_summary_up_to_message",
                message_seq_in_session,
                "count",
            )
        elif should_create_short:
            await _create_and_save_summary(
                workspace_name,
                session_name,
                message_id=message_id,
                message_seq_in_session=message_seq_in_session,
                message_public_id=message_public_id,
                summary_type=SummaryType.SHORT,
                configuration=configuration,
            )
            accumulate_metric(
                f"summary_{workspace_name}_{message_id}",
                "short_summary_up_to_message",
                message_seq_in_session,
                "count",
            )


async def _create_and_save_summary(
    workspace_name: str,
    session_name: str,
    *,
    message_id: int,
    message_seq_in_session: int,
    message_public_id: str,
    summary_type: SummaryType,
    configuration: schemas.ResolvedConfiguration,
) -> None:
    """
    Create a new summary and save it to the database.
    1. Get the latest summary
    2. Get the messages since the latest summary
    3. Generate a new summary using the messages and the previous summary
    4. Save the new summary to the database
    """

    logger.debug("Creating new %s summary", summary_type.name)
    summary_start = time.perf_counter()

    async with tracked_db("summary.fetch_data") as db:
        latest_summary = await get_summary(
            db, workspace_name, session_name, summary_type
        )
        if latest_summary:
            latest_summary_message_id = latest_summary["message_id"]
            # Skip if latest summary already covers message.
            if latest_summary_message_id >= message_id:
                return

        previous_summary_text = latest_summary["content"] if latest_summary else None

        # Calculate the sequence range for messages to summarize
        # We want to get the last N messages where N is the configured summary interval
        messages_per_summary = (
            configuration.summary.messages_per_long_summary
            if summary_type == SummaryType.LONG
            else configuration.summary.messages_per_short_summary
        )
        start_seq = max(message_seq_in_session - messages_per_summary + 1, 1)

        messages: list[Message] = await crud.get_messages_by_seq_range(
            db,
            workspace_name,
            session_name,
            start_seq=start_seq,
            end_seq=message_seq_in_session,
        )
        if not messages:
            logger.warning("No messages to summarize for message %s", message_id)
            return

        # Extract values before closing session
        formatted_messages = _format_messages(messages)
        last_message_id = messages[-1].id
        last_message_content_preview = messages[-1].content[:30]
        message_count = len(messages)

        messages_tokens = sum([message.token_count for message in messages])
        previous_summary_tokens = latest_summary["token_count"] if latest_summary else 0
        input_tokens = messages_tokens + previous_summary_tokens

    (
        new_summary,
        is_fallback,
        llm_input_tokens,
        llm_output_tokens,
    ) = await _create_summary(
        formatted_messages=formatted_messages,
        previous_summary_text=previous_summary_text,
        summary_type=summary_type,
        input_tokens=input_tokens,
        message_public_id=message_public_id,
        last_message_id=last_message_id,
        last_message_content_preview=last_message_content_preview,
        message_count=message_count,
    )

    # Step 3: Save to database with new transaction
    if not is_fallback:
        # Get base prompt tokens based on summary type
        if summary_type == SummaryType.SHORT:
            prompt_tokens = estimate_short_summary_prompt_tokens()
        else:
            prompt_tokens = estimate_long_summary_prompt_tokens()

        track_deriver_input_tokens(
            task_type=DeriverTaskTypes.SUMMARY,
            components={
                DeriverComponents.PROMPT: prompt_tokens,
                DeriverComponents.MESSAGES: messages_tokens,
                DeriverComponents.PREVIOUS_SUMMARY: previous_summary_tokens,
            },
        )

        # Track output tokens
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_deriver_tokens(
                count=new_summary["token_count"],
                task_type=DeriverTaskTypes.SUMMARY.value,
                token_type=TokenTypes.OUTPUT.value,
                component=DeriverComponents.OUTPUT_TOTAL.value,
            )

        # Save summary to database with new transaction
        async with tracked_db("summary.save") as db:
            await _save_summary(
                db,
                new_summary,
                workspace_name,
                session_name,
            )

    accumulate_metric(
        f"summary_{workspace_name}_{message_id}",
        f"{summary_type.name}_summary_text",
        new_summary["content"],
        "blob",
    )
    accumulate_metric(
        f"summary_{workspace_name}_{message_id}",
        f"{summary_type.name}_summary_size",
        new_summary["token_count"],
        "tokens",
    )

    summary_duration = (time.perf_counter() - summary_start) * 1000
    accumulate_metric(
        f"summary_{workspace_name}_{message_id}",
        f"{summary_type.name}_summary_creation",
        summary_duration,
        "ms",
    )

    # Emit telemetry event (only for non-fallback summaries)
    # Note: Using AgentToolSummaryCreatedEvent with dummy run_id/iteration since
    # this is called from the deriver, not from an agentic loop
    if not is_fallback:
        emit(
            AgentToolSummaryCreatedEvent(
                run_id="deriver",  # Placeholder - not from an agentic run
                iteration=0,  # Placeholder - not from an agentic loop
                parent_category="deriver",
                agent_type="summarizer",
                workspace_name=workspace_name,
                session_name=session_name,
                message_id=message_public_id,
                message_count=len(messages),
                message_seq_in_session=message_seq_in_session,
                summary_type="short" if summary_type == SummaryType.SHORT else "long",
                input_tokens=llm_input_tokens,
                output_tokens=llm_output_tokens,
            )
        )


async def _create_summary(
    formatted_messages: str,
    previous_summary_text: str | None,
    summary_type: SummaryType,
    input_tokens: int,
    message_public_id: str,
    last_message_id: int,
    last_message_content_preview: str,
    message_count: int,
) -> tuple[Summary, bool, int, int]:
    """
    Generate a summary of the provided messages using an LLM.

    Args:
        formatted_messages: Pre-formatted message string
        previous_summary_text: Optional previous summary to provide context
        summary_type: Type of summary to create ("short" or "long")
        input_tokens: Token count for input
        message_public_id: Public ID of the last message
        last_message_id: ID of the last message
        last_message_content_preview: Preview of last message content for fallback
        message_count: Number of messages for fallback

    Returns:
        A tuple of (Summary, is_fallback, llm_input_tokens, llm_output_tokens)
        where is_fallback indicates if the summary was generated using a
        fallback instead of an LLM call, and the token counts are from the LLM call
        (0 if fallback was used)
    """

    response: HonchoLLMCallResponse[str] | None = None
    is_fallback = False
    llm_input_tokens = 0
    llm_output_tokens = 0
    try:
        if summary_type == SummaryType.SHORT:
            response = await create_short_summary(
                formatted_messages, input_tokens, previous_summary_text
            )
        else:
            response = await create_long_summary(
                formatted_messages, previous_summary_text
            )

        summary_text = response.content
        summary_tokens = response.output_tokens
        llm_input_tokens = response.input_tokens
        llm_output_tokens = response.output_tokens

        if _hit_output_length_limit(response.finish_reasons):
            summary_text = _trim_partial_summary(summary_text)
            summary_tokens = estimate_tokens(summary_text) if summary_text else 0

        # Detect potential issues with the summary
        if not summary_text.strip():
            logger.error(
                "Generated summary is empty (finish_reasons=%s). Falling back to basic summary.",
                response.finish_reasons,
            )
            is_fallback = True
            summary_text = (
                f"Conversation with {message_count} messages about {last_message_content_preview}..."
                if message_count > 0
                else ""
            )
            summary_tokens = estimate_tokens(summary_text) if summary_text else 0
            llm_input_tokens = 0
            llm_output_tokens = 0
    except Exception:
        logger.exception("Error generating summary!")
        # Fallback to a basic summary in case of error
        summary_text = (
            f"Conversation with {message_count} messages about {last_message_content_preview}..."
            if message_count > 0
            else ""
        )
        summary_tokens = 0
        is_fallback = True

    return (
        Summary(
            content=summary_text,
            message_id=last_message_id,
            summary_type=summary_type.value,
            created_at=utc_now_iso(),
            token_count=summary_tokens,
            message_public_id=message_public_id,
        ),
        is_fallback,
        llm_input_tokens,
        llm_output_tokens,
    )


async def _merge_summary_segments(
    summary_type: SummaryType,
    segments: list[SummarySegment],
) -> SummarySegment:
    """Merge a set of already-compressed chronological summary segments."""
    formatted_segments = "\n".join(
        f"segment_{index}: {segment['content']}"
        for index, segment in enumerate(segments, start=1)
    )
    input_tokens = estimate_tokens(formatted_segments)

    response: HonchoLLMCallResponse[str] | None = None
    summary_text = ""
    summary_tokens = 0
    try:
        if summary_type == SummaryType.SHORT:
            response = await merge_short_summary_segments(formatted_segments, input_tokens)
        else:
            response = await merge_long_summary_segments(formatted_segments)

        summary_text = response.content
        summary_tokens = response.output_tokens
        if _hit_output_length_limit(response.finish_reasons):
            summary_text = _trim_partial_summary(summary_text)
            summary_tokens = estimate_tokens(summary_text) if summary_text else 0
    except Exception:
        logger.exception("Error merging summary segments!")
        summary_text = "\n\n".join(segment["content"] for segment in segments)
        summary_tokens = estimate_tokens(summary_text) if summary_text else 0

    if not summary_text.strip():
        summary_text = "\n\n".join(segment["content"] for segment in segments)
        summary_tokens = estimate_tokens(summary_text) if summary_text else 0

    return SummarySegment(
        content=summary_text,
        token_count=summary_tokens,
        message_id=segments[-1]["message_id"],
        message_public_id=segments[-1]["message_public_id"],
    )


def _refresh_generation_content_budget(summary_type: SummaryType) -> int:
    """Token budget for raw-message chunks during manual refresh."""
    max_output = (
        settings.SUMMARY.MAX_TOKENS_SHORT
        if summary_type == SummaryType.SHORT
        else settings.SUMMARY.MAX_TOKENS_LONG
    )
    prompt_tokens = (
        estimate_short_summary_prompt_tokens()
        if summary_type == SummaryType.SHORT
        else estimate_long_summary_prompt_tokens()
    )
    ratio = 1.5 if summary_type == SummaryType.SHORT else 2.0
    return max(int(max_output * ratio), max_output) - prompt_tokens


def _refresh_merge_content_budget(summary_type: SummaryType) -> int:
    """Token budget for merging summary segments during manual refresh."""
    max_output = (
        settings.SUMMARY.MAX_TOKENS_SHORT
        if summary_type == SummaryType.SHORT
        else settings.SUMMARY.MAX_TOKENS_LONG
    )
    prompt_tokens = (
        estimate_short_summary_merge_prompt_tokens()
        if summary_type == SummaryType.SHORT
        else estimate_long_summary_merge_prompt_tokens()
    )
    ratio = 2.0 if summary_type == SummaryType.SHORT else 3.0
    return max(int(max_output * ratio), max_output) - prompt_tokens


def _refresh_message_line_token_count(message: SummaryRefreshMessage) -> int:
    """Estimate token count for one formatted message line."""
    return estimate_tokens(f"{message['peer_name']}: {message['content']}")


def _refresh_segment_line_token_count(segment: SummarySegment) -> int:
    """Estimate token count for one formatted summary-segment line."""
    return estimate_tokens(f"segment: {segment['content']}")


def _chunk_messages_for_refresh(
    messages: list[SummaryRefreshMessage],
    summary_type: SummaryType,
) -> list[list[SummaryRefreshMessage]]:
    """Split raw session messages into token-budgeted chunks."""
    content_budget = max(_refresh_generation_content_budget(summary_type), 1)
    chunks: list[list[SummaryRefreshMessage]] = []
    current_chunk: list[SummaryRefreshMessage] = []
    current_tokens = 0

    for message in messages:
        line_tokens = _refresh_message_line_token_count(message)
        if current_chunk and current_tokens + line_tokens > content_budget:
            chunks.append(current_chunk)
            current_chunk = [message]
            current_tokens = line_tokens
        else:
            current_chunk.append(message)
            current_tokens += line_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _chunk_segments_for_merge(
    segments: list[SummarySegment],
    summary_type: SummaryType,
) -> list[list[SummarySegment]]:
    """Split summary segments into token-budgeted merge groups."""
    if len(segments) <= 1:
        return [segments]

    segment_line_tokens = [_refresh_segment_line_token_count(segment) for segment in segments]
    dynamic_floor = max(segment_line_tokens) * 2
    content_budget = max(_refresh_merge_content_budget(summary_type), dynamic_floor)

    groups: list[list[SummarySegment]] = []
    current_group: list[SummarySegment] = []
    current_tokens = 0

    for segment, line_tokens in zip(segments, segment_line_tokens, strict=False):
        if current_group and current_tokens + line_tokens > content_budget:
            groups.append(current_group)
            current_group = [segment]
            current_tokens = line_tokens
        else:
            current_group.append(segment)
            current_tokens += line_tokens

    if current_group:
        groups.append(current_group)

    if len(groups) == len(segments):
        forced_groups: list[list[SummarySegment]] = []
        for index in range(0, len(segments), 2):
            forced_groups.append(segments[index : index + 2])
        return forced_groups

    return groups


async def _save_summary(
    db: AsyncSession,
    summary: Summary,
    workspace_name: str,
    session_name: str,
) -> None:
    """
    Save a summary as metadata on a session.

    Args:
        db: Database session
        summary: The summary to save
        workspace_name: Workspace name
        session_name: Session name
    """
    from src.exceptions import ResourceNotFoundException

    # Get the label value from the enum
    label_value = summary["summary_type"]

    try:
        session = await crud.get_session(db, session_name, workspace_name)
    except ResourceNotFoundException:
        # If session doesn't exist, we can't save the summary
        logger.warning(
            f"Cannot save summary: session {session_name} not found in workspace {workspace_name}"
        )
        return

    # Use SQLAlchemy update() with PostgreSQL's || operator to properly merge JSONB
    # We need to merge the new summary into the existing summaries structure
    update_data = {}
    existing_summaries = session.internal_metadata.get(SUMMARIES_KEY, {})
    existing_summaries[label_value] = summary
    update_data[SUMMARIES_KEY] = existing_summaries

    stmt = (
        update(models.Session)
        .where(models.Session.workspace_name == workspace_name)
        .where(models.Session.name == session_name)
        .values(
            internal_metadata=models.Session.internal_metadata.op("||")(update_data)
        )
    )

    await db.execute(stmt)
    await db.commit()

    cache_key = session_cache_key(workspace_name, session_name)
    await cache_client.delete(cache_key)


async def _rebuild_summary_from_scratch(
    workspace_name: str,
    session_name: str,
    *,
    summary_type: SummaryType,
    messages: list[SummaryRefreshMessage],
) -> None:
    """Regenerate a summary slot from the full session history using balanced merges."""
    leaf_segments: list[SummarySegment] = []

    for chunk in _chunk_messages_for_refresh(messages, summary_type):
        if not chunk:
            continue

        formatted_messages = "\n".join(
            f"{message['peer_name']}: {message['content']}" for message in chunk
        )
        messages_tokens = estimate_tokens(formatted_messages)

        latest_summary, _, _, _ = await _create_summary(
            formatted_messages=formatted_messages,
            previous_summary_text=None,
            summary_type=summary_type,
            input_tokens=messages_tokens,
            message_public_id=chunk[-1]["public_id"],
            last_message_id=chunk[-1]["id"],
            last_message_content_preview=chunk[-1]["content"][:30],
            message_count=len(chunk),
        )

        leaf_segments.append(
            SummarySegment(
                content=latest_summary["content"],
                token_count=latest_summary["token_count"],
                message_id=latest_summary["message_id"],
                message_public_id=latest_summary["message_public_id"],
            )
        )

    if not leaf_segments:
        raise ValueError("Cannot refresh summaries for an empty session")

    current_segments = leaf_segments
    while len(current_segments) > 1:
        next_segments: list[SummarySegment] = []
        for group in _chunk_segments_for_merge(current_segments, summary_type):
            next_segments.append(await _merge_summary_segments(summary_type, group))
        current_segments = next_segments

    final_segment = current_segments[0]
    final_summary = Summary(
        content=final_segment["content"],
        message_id=final_segment["message_id"],
        summary_type=summary_type.value,
        created_at=utc_now_iso(),
        token_count=final_segment["token_count"],
        message_public_id=final_segment["message_public_id"],
    )

    async with tracked_db("summary.refresh.save") as db:
        await _save_summary(
            db,
            final_summary,
            workspace_name,
            session_name,
        )


async def refresh_session_summaries(
    workspace_name: str,
    session_name: str,
    *,
    refresh_short: bool = True,
    refresh_long: bool = True,
) -> None:
    """
    Force-regenerate stored summaries for a session using current settings.

    This bypasses threshold/existence checks and rebuilds from the full message
    history so changes to prompts, models, or token limits are reflected.
    """
    if not refresh_short and not refresh_long:
        return

    async with tracked_db("summary.refresh.fetch") as db:
        await crud.get_session(db, session_name, workspace_name)

        result = await db.execute(
            select(models.Message)
            .where(models.Message.workspace_name == workspace_name)
            .where(models.Message.session_name == session_name)
            .order_by(models.Message.seq_in_session.asc())
        )
        messages = [
            SummaryRefreshMessage(
                id=message.id,
                public_id=message.public_id,
                peer_name=message.peer_name,
                content=message.content,
                token_count=message.token_count,
            )
            for message in result.scalars().all()
        ]

    if not messages:
        raise ValueError("Cannot refresh summaries for an empty session")

    tasks = []
    if refresh_short:
        tasks.append(
            _rebuild_summary_from_scratch(
                workspace_name,
                session_name,
                summary_type=SummaryType.SHORT,
                messages=messages,
            )
        )
    if refresh_long:
        tasks.append(
            _rebuild_summary_from_scratch(
                workspace_name,
                session_name,
                summary_type=SummaryType.LONG,
                messages=messages,
            )
        )

    await asyncio.gather(*tasks)


async def get_summarized_history(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    cutoff: int | None = None,
    summary_type: SummaryType = SummaryType.SHORT,
) -> str:
    """
    Get a summarized version of the chat history by combining the latest summary
    with all messages since that summary.

    Note: history is exclusive of the cutoff message.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        cutoff: (Optional) message ID to cutoff at
        summary_type: Type of summary to get ("short" or "long")

    Returns:
        A string formatted history text with summary and recent messages
    """
    # Get messages since the latest summary and the summary itself
    summary = await get_summary(db, workspace_name, session_name, summary_type)

    # Check if we have a valid summary with a message_id
    if summary:
        messages = await crud.get_messages_id_range(
            db,
            workspace_name,
            session_name,
            start_id=summary["message_id"],
            end_id=cutoff,
        )
    else:
        messages = await crud.get_messages_id_range(
            db, workspace_name, session_name, end_id=cutoff
        )

    # Format messages
    messages_text = _format_messages(messages)

    if summary:
        # Combine summary with recent messages
        return f"""
<summary>
{summary["content"]}
</summary>
<recent_messages>
{messages_text}
</recent_messages>
"""

    # No summary available, return just the messages
    return messages_text


async def get_summary(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    summary_type: SummaryType = SummaryType.SHORT,
) -> Summary | None:
    """
    Get summary for a given session.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        summary_type: Type of summary to retrieve ("short" or "long")

    Returns:
        The summary data dictionary, or None if no summary exists
    """
    try:
        session = await crud.get_session(db, session_name, workspace_name)
    except ResourceNotFoundException:
        # If session doesn't exist, there's no summary to retrieve
        return None

    summaries: dict[str, Summary] = session.internal_metadata.get(SUMMARIES_KEY, {})
    if not summaries or summary_type.value not in summaries:
        return None
    return summaries[summary_type.value]


async def get_both_summaries(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
) -> tuple[Summary | None, Summary | None]:
    """
    Get both short and long summaries for a given session.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name

    Returns:
        A tuple of the short and long summaries, or None if no summary exists
    """
    try:
        session = await crud.get_session(db, session_name, workspace_name)
    except ResourceNotFoundException:
        # If session doesn't exist, there's no summary to retrieve
        return None, None

    summaries: dict[str, Summary] = session.internal_metadata.get(SUMMARIES_KEY, {})
    return summaries.get(SummaryType.SHORT.value), summaries.get(SummaryType.LONG.value)


async def get_session_context(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    token_limit: int,
    *,
    cutoff: int | None = None,
    include_summary: bool = True,
) -> tuple[schemas.Summary | None, list[models.Message]]:
    """
    Get session context similar to the API endpoint but for internal use.

    Args:
        db: Database session
        workspace_name: The workspace name
        session_name: The session name
        token_limit: Maximum tokens for the context
        cutoff: Optional message ID to stop at (exclusive)
        include_summary: Whether to include summary if available

    Returns:
        Tuple of (summary, messages) where summary is a Summary pydantic model (or None)
        and messages is the list of message objects
    """
    if token_limit <= 0:
        return None, []

    summary = None
    messages_tokens = token_limit
    messages_start_id = 0

    if include_summary:
        # Allocate 40% of tokens to summary, 60% to messages
        summary_tokens_limit = int(token_limit * 0.4)

        latest_short_summary, latest_long_summary = await get_both_summaries(
            db, workspace_name, session_name
        )

        long_len = latest_long_summary["token_count"] if latest_long_summary else 0
        short_len = latest_short_summary["token_count"] if latest_short_summary else 0

        # Return the longest summary that fits within the token limit
        if (
            latest_long_summary
            and long_len <= summary_tokens_limit
            and long_len > short_len
        ):
            summary = schemas.Summary(
                content=latest_long_summary["content"],
                message_id=latest_long_summary["message_id"],
                summary_type=latest_long_summary["summary_type"],
                created_at=latest_long_summary["created_at"],
                token_count=latest_long_summary["token_count"],
                message_public_id=latest_long_summary.get("message_public_id", ""),
            )
            messages_tokens = token_limit - latest_long_summary["token_count"]
            messages_start_id = latest_long_summary["message_id"]
        elif (
            latest_short_summary and short_len <= summary_tokens_limit and short_len > 0
        ):
            summary = schemas.Summary(
                content=latest_short_summary["content"],
                message_id=latest_short_summary["message_id"],
                summary_type=latest_short_summary["summary_type"],
                created_at=latest_short_summary["created_at"],
                token_count=latest_short_summary["token_count"],
                message_public_id=latest_short_summary.get("message_public_id", ""),
            )
            messages_tokens = token_limit - latest_short_summary["token_count"]
            messages_start_id = latest_short_summary["message_id"]
        else:
            logger.debug(
                "No summary available for get_context call with token limit %s, returning empty string. Normal if brand-new session. long_summary_len: %s, short_summary_len: %s",
                token_limit,
                long_len,
                short_len,
            )

    # Get recent messages after summary
    messages = await crud.get_messages_id_range(
        db,
        workspace_name,
        session_name,
        start_id=messages_start_id,
        end_id=cutoff,
        token_limit=messages_tokens,
    )

    return summary, messages


async def get_session_context_formatted(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    token_limit: int,
    *,
    cutoff: int | None = None,
    include_summary: bool = True,
) -> str:
    """
    Get formatted session context as a string for internal use (e.g., deriver).

    This is a convenience wrapper around get_session_context that formats
    the output as a string.
    """
    if token_limit <= 0:
        return ""

    summary, messages = await get_session_context(
        db,
        workspace_name,
        session_name,
        token_limit,
        cutoff=cutoff,
        include_summary=include_summary,
    )

    # Format the messages
    messages_text = _format_messages(messages)
    summary_content = summary.content if summary else ""

    if summary_content and messages_text:
        return f"""<summary>
{summary_content}
</summary>

<recent_messages>
{messages_text}
</recent_messages>"""
    elif summary_content:
        return f"""<summary>
{summary_content}
</summary>"""
    elif messages_text:
        return messages_text
    else:
        return ""


def _format_messages(messages: list[models.Message]) -> str:
    """
    Format a list of messages into a string by concatenating their content and
    prefixing each with the peer name.
    """
    if len(messages) == 0:
        return ""
    return "\n".join([f"{msg.peer_name}: {msg.content}" for msg in messages])

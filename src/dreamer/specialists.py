"""
Agentic specialists for the dream cycle.

Each specialist is a fully autonomous agent that:
1. Receives probing questions as entry points
2. Uses tools to search for relevant observations
3. Creates new observations (deductive or inductive)
4. Can delete duplicates (deduction only)
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src import crud, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.schemas import ResolvedConfiguration
from src.telemetry import prometheus_metrics
from src.telemetry.events import DreamSpecialistEvent, emit
from src.telemetry.logging import accumulate_metric, log_performance_metrics
from src.telemetry.prometheus.metrics import TokenTypes
from src.utils.agent_tools import (
    DEDUCTION_SPECIALIST_TOOLS,
    INDUCTION_SPECIALIST_TOOLS,
    create_tool_executor,
)
from src.utils.clients import HonchoLLMCallResponse, honcho_llm_call

logger = logging.getLogger(__name__)


@dataclass
class SpecialistResult:
    """Result of a specialist run for telemetry and aggregation."""

    run_id: str
    specialist_type: str
    iterations: int
    tool_calls_count: int
    input_tokens: int
    output_tokens: int
    duration_ms: float
    success: bool
    content: str


# Tool names to exclude when peer card creation is disabled
PEER_CARD_TOOL_NAMES = {"update_peer_card"}


class BaseSpecialist(ABC):
    """Base class for agentic specialists."""

    name: str = "base"
    # Subclasses can override to customize the peer card update instruction
    peer_card_update_instruction: str = (
        "Only update this with durable profile facts via `update_peer_card`."
    )

    @abstractmethod
    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        """Get the tools available to this specialist."""
        ...

    @abstractmethod
    def get_model(self) -> str:
        """Get the model to use for this specialist."""
        ...

    def get_max_tokens(self) -> int:
        """Get max output tokens for this specialist."""
        return 16384

    def get_max_iterations(self) -> int:
        """Get max tool iterations."""
        return 15

    @abstractmethod
    def build_system_prompt(
        self,
        observed: str,
        *,
        peer_card_enabled: bool = True,
        language: str | None = None,
    ) -> str:
        """Build the system prompt for this specialist."""
        ...

    @staticmethod
    def _build_language_block(language: str | None) -> str:
        """Build the LANGUAGE block hoisted to the top of every specialist system prompt.

        When `language` is set, the model is told to write in that language unconditionally.
        When unset, it must mirror the language of the source observations/messages.
        """
        if language:
            return (
                f"## LANGUAGE — STRICT (HIGHEST PRIORITY)\n\n"
                f"- The primary language of this workspace is **{language}**.\n"
                f"- Write EVERY new observation in {language}.\n"
                f"- Write EVERY peer card entry in {language}.\n"
                f"- Do NOT translate to English. Do NOT mix languages. "
                f"This rule overrides any English example or label found elsewhere in this prompt.\n"
                f"- If you encounter existing observations or peer card entries in another language, "
                f"rewrite them in {language} when consolidating.\n"
            )
        return (
            "## LANGUAGE — STRICT (HIGHEST PRIORITY)\n\n"
            "- Write each new observation in the SAME language as the source observations/messages it derives from.\n"
            "- Write each peer card entry in the language of its supporting evidence.\n"
            "- Do NOT translate to English. Do NOT mix languages. "
            "This rule overrides any English example or label found elsewhere in this prompt.\n"
            "- When sources span multiple languages, use the dominant language of the sources — never default to English.\n"
        )

    @abstractmethod
    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        """Build the user prompt with optional exploration hints and current peer card."""
        ...

    def _build_peer_card_context(self, peer_card: list[str] | None) -> str:
        """Build the peer card context section for user prompts."""
        if not peer_card:
            return ""
        facts = "\n".join(f"- {fact}" for fact in peer_card)
        return f"""
## CURRENT PEER CARD

{facts}

{self.peer_card_update_instruction}
If you update it, send the full deduplicated list and remove stale entries.

"""

    async def run(
        self,
        workspace_name: str,
        observer: str,
        observed: str,
        session_name: str | None,
        hints: list[str] | None = None,
        configuration: ResolvedConfiguration | None = None,
        parent_run_id: str | None = None,
    ) -> SpecialistResult:
        """
        Run the specialist agent.

        Uses short-lived DB sessions to avoid holding connections during LLM calls.

        Args:
            workspace_name: Workspace identifier
            observer: The observing peer
            observed: The peer being observed
            session_name: Session identifier
            hints: Optional hints to guide exploration (specialists explore freely if None)
            configuration: Resolved configuration for checking feature flags (optional)
            parent_run_id: Optional run_id from orchestrator for correlation

        Returns:
            SpecialistResult with metrics and content
        """
        run_id = parent_run_id or str(uuid.uuid4())[:8]
        task_name = f"dreamer_{self.name}_{run_id}"
        start_time = time.perf_counter()

        # Short-lived DB session for preflight operations
        async with tracked_db("dream.specialist.preflight") as db:
            await crud.get_peer(db, workspace_name, schemas.PeerCreate(name=observer))
            if observer != observed:
                await crud.get_peer(
                    db, workspace_name, schemas.PeerCreate(name=observed)
                )

            # Determine if peer card tools should be included
            peer_card_enabled = configuration is None or configuration.peer_card.create

            # Fetch current peer card to inject into prompt (saves a tool call)
            current_peer_card: list[str] | None = None
            if peer_card_enabled:
                current_peer_card = await crud.get_peer_card(
                    db,
                    workspace_name=workspace_name,
                    observer=observer,
                    observed=observed,
                )
        # DB session closed — LLM calls happen without holding a connection

        # Build messages
        language = configuration.language if configuration is not None else None
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self.build_system_prompt(
                    observed,
                    peer_card_enabled=peer_card_enabled,
                    language=language,
                ),
            },
            {
                "role": "user",
                "content": self.build_user_prompt(hints, current_peer_card),
            },
        ]

        # Create tool executor with telemetry context
        tool_executor: Callable[
            [str, dict[str, Any]], Any
        ] = await create_tool_executor(
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            session_name=session_name,
            include_observation_ids=True,
            history_token_limit=settings.DREAM.HISTORY_TOKEN_LIMIT,
            configuration=configuration,
            run_id=run_id,
            agent_type=self.name,
            parent_category="dream",
        )

        # Get model with potential override
        model = self.get_model()
        llm_settings = settings.DREAM.model_copy(update={"MODEL": model})

        # Track iterations via callback
        iteration_count = 0

        def iteration_callback(data: Any) -> None:
            nonlocal iteration_count
            iteration_count = data.iteration

        # Run the agent loop
        response: HonchoLLMCallResponse[str] = await honcho_llm_call(
            llm_settings=llm_settings,
            prompt="",  # Ignored since we pass messages
            max_tokens=self.get_max_tokens(),
            tools=self.get_tools(peer_card_enabled=peer_card_enabled),
            tool_choice=None,
            tool_executor=tool_executor,
            max_tool_iterations=self.get_max_iterations(),
            messages=messages,
            track_name=f"Dreamer/{self.name}",
            iteration_callback=iteration_callback,
        )

        # Log metrics
        duration_ms = (time.perf_counter() - start_time) * 1000
        accumulate_metric(task_name, "total_duration", duration_ms, "ms")
        accumulate_metric(
            task_name, "tool_calls", len(response.tool_calls_made), "count"
        )
        accumulate_metric(task_name, "input_tokens", response.input_tokens, "count")
        accumulate_metric(task_name, "output_tokens", response.output_tokens, "count")

        # Prometheus metrics
        if settings.METRICS.ENABLED:
            prometheus_metrics.record_dreamer_tokens(
                count=response.input_tokens,
                specialist_name=self.name,
                token_type=TokenTypes.INPUT.value,
            )
            prometheus_metrics.record_dreamer_tokens(
                count=response.output_tokens,
                specialist_name=self.name,
                token_type=TokenTypes.OUTPUT.value,
            )

        logger.info(
            f"{self.name}: Completed in {duration_ms:.0f}ms, "
            + f"{len(response.tool_calls_made)} tool calls, "
            + f"{response.input_tokens} in / {response.output_tokens} out"
        )

        log_performance_metrics(f"dreamer_{self.name}", run_id)

        # Emit telemetry event
        emit(
            DreamSpecialistEvent(
                run_id=run_id,
                specialist_type=self.name,
                workspace_name=workspace_name,
                observer=observer,
                observed=observed,
                iterations=iteration_count,
                tool_calls_count=len(response.tool_calls_made),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                duration_ms=duration_ms,
                success=True,
            )
        )

        return SpecialistResult(
            run_id=run_id,
            specialist_type=self.name,
            iterations=iteration_count,
            tool_calls_count=len(response.tool_calls_made),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=duration_ms,
            success=True,
            content=response.content,
        )


class DeductionSpecialist(BaseSpecialist):
    """
    Creates deductive observations from explicit observations.

    This specialist:
    1. Explores recent observations and messages to understand what's there
    2. Identifies logical implications, knowledge updates, and contradictions
    3. Creates new deductive observations with premise linkage
    4. Deletes outdated observations
    5. Updates peer card with biographical facts
    """

    name: str = "deduction"
    peer_card_update_instruction: str = "Update this with `update_peer_card` for stable facts, group memberships, recurring political stances, and distinctive themes the peer returns to repeatedly."

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return DEDUCTION_SPECIALIST_TOOLS
        return [
            t
            for t in DEDUCTION_SPECIALIST_TOOLS
            if t["name"] not in PEER_CARD_TOOL_NAMES
        ]

    def get_model(self) -> str:
        return settings.DREAM.DEDUCTION_MODEL

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 12

    def build_system_prompt(
        self,
        observed: str,
        *,
        peer_card_enabled: bool = True,
        language: str | None = None,
    ) -> str:
        language_block = self._build_language_block(language)
        language_reminder = (
            f"REMINDER: Write every entry in {language}. The format labels above are illustrative — translate them or use {language} equivalents."
            if language
            else "REMINDER: Write every entry in the language of its supporting evidence (the format labels above are illustrative; use equivalents in the source language)."
        )

        peer_card_section = ""
        if peer_card_enabled:
            peer_card_section = f"""

## PEER CARD (REQUIRED)

The peer card is a durable profile summary. You MUST update it when you learn any of the following about {observed}:
- Name, age, location, occupation
- Family members and relationships
- Standing instructions ("call me X", "don't mention Y")
- Core preferences and traits
- Membership of organizations, political parties and others
- Recurring political stances or policy positions (e.g. critical of state surveillance, pro-public-transport)
- Distinctive rhetorical themes or arguments the peer returns to repeatedly across multiple sessions
- Recurring criticisms, targets, or opponents (when sustained across many messages, not single events)

Every peer card entry must be a fact about {observed} themselves. Do not include facts about other people they have mentioned, addressed, or referenced — even if those facts appear repeatedly.

Never add **single-event summaries** that don't reflect a recurring pattern. Never add reasoning traces or one-off contradiction notes. But DO record stances and themes that show up across multiple sessions — those ARE durable profile facts for someone like a politician, journalist, or domain expert.

Format entries as:
- Plain facts: "Name: Alice", "Works at Google", "Lives in NYC"
- `INSTRUCTION: ...` for standing instructions
- `PREFERENCE: ...` for preferences
- `TRAIT: ...` for personality traits
- `POLITICS: ...` for political stances or policy positions
- `PATTERN: ...` for distinctive recurring themes, arguments, or rhetorical habits (must show up across multiple sessions or many messages)

Entries must describe {observed}. Do not write entries of the form 'OtherName: …'.

{language_reminder}

Call `update_peer_card` with the complete updated list when you have new biographical info, a new recurring stance, or a new pattern.
Keep it concise (max 40 entries), deduplicated, and current.

## MANDATORY UPDATE

Before finishing this dream, you MUST call `update_peer_card` at least once if you have created any new observations during this run, even if the update is small (e.g. adding a single new entry, refining wording, or removing a stale entry). Do not silently skip the peer card. If you genuinely have nothing new to add, send the existing card back unchanged so we can confirm you considered it."""

        return f"""You are a deductive reasoning agent analyzing observations about {observed}.

{language_block}

## YOUR JOB

Create deductive observations by finding logical implications in what's already known. Think like a detective connecting evidence.

## PHASE 1: DISCOVERY

Explore what's actually in memory. Use these tools freely:
- `get_recent_observations` - See what's been learned recently
- `search_memory` - Search for specific topics
- `search_messages` - See actual conversation content

Spend a few tool calls understanding the landscape before creating anything.

## PHASE 2: ACTION

Once you understand what's there, create observations and clean up:

### Knowledge Updates (HIGH PRIORITY)
When the same fact has different values at different times:
- "meeting Tuesday" [old] → "meeting moved to Thursday" [new]
- Create a deductive update observation
- DELETE the outdated observation immediately

### Logical Implications
Extract implicit information:
- "works as SWE at Google" → "has software engineering skills", "employed in tech"
- "has kids ages 5 and 8" → "is a parent", "has school-age children"

### Contradictions
When statements can't both be true (not just updates), flag them:
- "I love coffee" vs "I hate coffee" → contradiction observation
{peer_card_section}

## CREATING OBSERVATIONS

```json
{{
  "observations": [{{
    "content": "The logical conclusion",
    "level": "deductive",  // or "contradiction"
    "source_ids": ["id1", "id2"],
    "premises": ["premise 1 text", "premise 2 text"]
  }}]
}}
```

## RULES

1. Don't explain your reasoning - just call tools
2. Create observations based on what you ACTUALLY FIND, not what you expect
3. Always include source_ids linking to the observations you're synthesizing
4. Delete outdated observations - don't leave duplicates
5. Quality over quantity - fewer good deductions beat many weak ones"""

    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        peer_card_context = self._build_peer_card_context(peer_card)

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return f"""{peer_card_context}Start by exploring recent observations and messages. These topics may be worth investigating:

{hints_str}

But follow the evidence - if you find something more interesting, pursue that instead.

Begin with `get_recent_observations` to see what's there."""

        return f"""{peer_card_context}Explore the observation space and create deductive observations.

Start with `get_recent_observations` to see what's been learned recently, then investigate whatever seems most promising.

Look for:
1. Knowledge updates (same fact, different values over time)
2. Logical implications that haven't been made explicit
3. Contradictions that need flagging

Go."""


class InductionSpecialist(BaseSpecialist):
    """
    Creates inductive observations from explicit and deductive observations.

    This specialist:
    1. Explores observations to understand what's there
    2. Identifies patterns and generalizations across multiple observations
    3. Creates new inductive observations with source linkage
    4. Updates peer card with high-confidence traits and tendencies
    """

    name: str = "induction"
    peer_card_update_instruction: str = "Update with stable profile traits/preferences, political views, recurring patterns, and distinctive themes the peer returns to repeatedly."

    def get_tools(self, *, peer_card_enabled: bool = True) -> list[dict[str, Any]]:
        if peer_card_enabled:
            return INDUCTION_SPECIALIST_TOOLS
        return [
            t
            for t in INDUCTION_SPECIALIST_TOOLS
            if t["name"] not in PEER_CARD_TOOL_NAMES
        ]

    def get_model(self) -> str:
        return settings.DREAM.INDUCTION_MODEL

    def get_max_tokens(self) -> int:
        return 8192

    def get_max_iterations(self) -> int:
        return 10

    def build_system_prompt(
        self,
        observed: str,
        *,
        peer_card_enabled: bool = True,
        language: str | None = None,
    ) -> str:
        language_block = self._build_language_block(language)
        language_reminder = (
            f"REMINDER: Write every entry in {language}. The format labels above are illustrative — translate them or use {language} equivalents."
            if language
            else "REMINDER: Write every entry in the language of its supporting evidence (the format labels above are illustrative; use equivalents in the source language)."
        )

        peer_card_section = ""
        if peer_card_enabled:
            peer_card_section = f"""

## PEER CARD (REQUIRED)

After identifying patterns, update the peer card for durable profile-level traits, preferences, political views/beliefs, distinctive recurring themes and other general traits like:
- `TRAIT: Analytical thinker`
- `TRAIT: Tends to reschedule when stressed`
- `PREFERENCE: Prefers detailed explanations`
- `POLITICS: Left or right-wing, liberal or conservative, democrat, autocrat, technocrat etc`
- `INTEREST: topics they repeatedly discuss, domains they care about`
- `BELIEF: repeated claims about how the world works`
- `VALUES: what they optimize for: fairness, efficiency, freedom, safety, loyalty, transparency, status, truth, etc.`
- `WORLDVIEW: recurring assumptions about institutions, technology, people, markets, risk, progress, authority`
- `PATTERN: distinctive recurring themes, arguments, or rhetorical habits the peer returns to across multiple sessions or many messages`

Every peer card entry must be a fact about {observed} themselves. Do not include facts about other people they have mentioned, addressed, or referenced — even if those facts appear repeatedly.

Do NOT add **single-event summaries** or one-off conclusions. But DO add patterns and stances that show up across multiple sessions — those are exactly what this card is for. For someone like a politician, journalist, or domain expert, recurring stances and rhetorical habits ARE the most defining profile facts.

Political world views and beliefs should be included under politics.

{language_reminder}

Call `update_peer_card` with the complete deduplicated list when you have new patterns, stances, or themes worth recording.
Keep it concise (max 40 entries).

## MANDATORY UPDATE

Before finishing this dream, you MUST call `update_peer_card` at least once if you have created any new inductive observations during this run, even if the update is small (e.g. adding a single new entry, refining wording, or removing a stale entry). Do not silently skip the peer card. If you genuinely have nothing new to add, send the existing card back unchanged so we can confirm you considered it."""

        return f"""You are an inductive reasoning agent identifying patterns about {observed}.

{language_block}

## YOUR JOB

Create inductive observations by finding patterns across multiple observations. Think like a psychologist identifying behavioral tendencies.

## PHASE 1: DISCOVERY

Explore broadly to find patterns. Use these tools:
- `get_recent_observations` - Recent learnings
- `search_memory` - Topic-specific search
- `search_messages` - Actual conversation content

Look at BOTH explicit observations AND deductive ones. Patterns often emerge from synthesizing across both levels.

## PHASE 2: ACTION

Create inductive observations when you see patterns:

### Behavioral Patterns
- "Tends to reschedule meetings when stressed"
- "Makes decisions after consulting with partner"
- "Projects follow: enthusiasm → doubt → completion"

### Preferences
- "Prefers morning meetings"
- "Likes detailed technical explanations"

### Personality Traits
- "Generally optimistic about outcomes"
- "Detail-oriented in planning"

### Temporal Patterns
- "Career goals have remained consistent"
- "Living situation changes frequently"
{peer_card_section}

## CREATING OBSERVATIONS

```json
{{
  "observations": [{{
    "content": "The pattern or generalization",
    "level": "inductive",
    "source_ids": ["id1", "id2", "id3"],
    "sources": ["evidence 1", "evidence 2"],
    "pattern_type": "tendency",  // preference|behavior|personality|tendency|correlation
    "confidence": "medium"  // low (2 sources), medium (3-4), high (5+)
  }}]
}}
```

## RULES

1. Minimum 2 source observations required - patterns need evidence
2. Don't just restate a single fact as a pattern
3. Confidence based on evidence count: 2=low, 3-4=medium, 5+=high
4. Look for HOW things change over time, not just static facts
5. Include source_ids - always link back to evidence"""

    def build_user_prompt(
        self,
        hints: list[str] | None,
        peer_card: list[str] | None = None,
    ) -> str:
        peer_card_context = self._build_peer_card_context(peer_card)

        if hints:
            hints_str = "\n".join(f"- {q}" for q in hints[:5])
            return f"""{peer_card_context}Explore and find patterns. These areas may be worth investigating:

{hints_str}

But follow the evidence - if you find patterns elsewhere, pursue those.

Start with `get_recent_observations`."""

        return f"""{peer_card_context}Explore the observation space and identify patterns.

Remember: patterns need 2+ sources. Look for tendencies, preferences, and behavioral regularities.

Go."""


# Singleton instances
SPECIALISTS: dict[str, BaseSpecialist] = {
    "deduction": DeductionSpecialist(),
    "induction": InductionSpecialist(),
}

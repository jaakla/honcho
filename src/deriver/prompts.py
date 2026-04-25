"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
    language: str | None = None,
) -> str:
    """
    Generate minimal prompt for fast observation extraction.

    Args:
        peer_id: The ID of the user being analyzed.
        messages: All messages in the range (interleaving messages and new turns combined).
        language: Optional workspace/session primary language. When provided, the model
            is instructed to write all observations in this language.

    Returns:
        Formatted prompt string for observation extraction.
    """
    if language:
        language_block = c(
            f"""
[LANGUAGE — STRICT]
- The primary language of this workspace is **{language}**.
- Write EVERY observation in {language}.
- Do NOT translate to English. Do NOT mix languages. Do NOT use this prompt's language as a cue for output language.
- Direct quotes from the source messages must be reproduced verbatim in their original language.
"""
        )
    else:
        language_block = c(
            """
[LANGUAGE — STRICT]
- Write each observation in the SAME language as the source message it is derived from.
- Do NOT translate observations into English (or any other language) regardless of how this prompt is written.
- If a message is in Estonian, write the observation in Estonian. If in Spanish, write in Spanish. And so on.
- Direct quotes must be reproduced verbatim in their original language.
"""
        )

    return c(
        f"""
Analyze messages from {peer_id} to extract **explicit atomic facts** about them.

{language_block}

[EXPLICIT] DEFINITION: Facts about {peer_id} that can be derived directly from their messages.
   - Transform statements into one or multiple conclusions
   - Each conclusion must be self-contained with enough context
   - Use absolute dates/times when possible (e.g. "June 26, 2025" not "yesterday")

RULES:
- Properly attribute observations to the correct subject: if it is about {peer_id}, say so. If {peer_id} is referencing someone or something else, make that clear.
- Observations should make sense on their own. Each observation will be used in the future to better understand {peer_id}.
- Extract ALL observations from {peer_id} messages, using others as context.
- Contextualize each observation sufficiently (e.g. name the subject, the topic, and any constraint that makes the fact meaningful on its own).

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate base prompt tokens (cached)."""
    try:
        prompt = minimal_deriver_prompt(
            peer_id="",
            messages="",
        )
        return estimate_tokens(prompt)
    except Exception:
        return 300

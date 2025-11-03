"""
Image prompt rewriter for combining historical context with new instructions.
Uses LLM-based rewriting to intelligently merge previous image context with new modifications.
Works with prompt_history array similar to chat history.
"""

import json
import logging
from typing import Callable, List, Tuple


class ImagePromptRewriter:
    """Rewrites image prompts by combining historical context with new instructions using LLM."""

    def _llm_rewrite(
        self,
        base_prompt: str,
        instruction: str,
        llm_call: Callable[[str], str],
    ) -> Tuple[str, str]:
        """
        LLM-based prompt rewriting that intelligently merges context with new instruction.

        Args:
            base_prompt: Original full prompt from history
            instruction: User's new instruction
            llm_call: Callable that takes combined prompt and returns LLM response

        Returns:
            Tuple of (rewritten_prompt, context_type) where context_type is "modification" or "new_request"

        Raises:
            Exception: If LLM call fails
        """
        # Construct a minimal-diff full-rewriter prompt (no enhancement)
        user_prompt = (
            f"You are a minimal-diff prompt rewriter. Decide if the new request modifies the previous "
            f"image or is a new request.\n\n"
            f'PREVIOUS IMAGE PROMPT:\n"{base_prompt}"\n\n'
            f'USER\'S NEW REQUEST:\n"{instruction}"\n\n'
            f"DECISION CRITERIA:\n"
            f"A) MODIFICATION: Changes an attribute of the existing scene/subject (color, style, add/remove element).\n"
            f"B) NEW REQUEST: Describes a different scene/subject.\n\n"
            f"YOUR TASK:\n"
            f"1) If MODIFICATION: Output a FULL rewritten prompt. Apply ONLY the requested change.\n"
            f"   - Preserve original wording/order as much as possible.\n"
            f"   - No adjectives or new details unless explicitly requested.\n"
            f"   - Prefer using only words from the base and the user's request plus minimal glue words.\n"
            f"2) If NEW REQUEST: Output the user's request EXACTLY as given.\n\n"
            f"IMPORTANT:\n"
            f"- No enhancement, no style elaboration, no extra descriptors.\n"
            f"- Do NOT introduce new nouns or locations unless explicitly requested.\n"
            f"- Output JSON only.\n\n"
            f"OUTPUT FORMAT (JSON):\n"
            f'{{\n  "decision": "modification" or "new_request",\n'
            f'  "final_prompt": "the final prompt text"\n}}\n'
        )

        # Call LLM
        response = llm_call(user_prompt)

        # Parse JSON response
        try:
            # Clean up response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(
                    line for line in lines if not line.strip().startswith("```")
                )

            # Parse JSON
            result = json.loads(response)
            decision = result.get("decision", "new_request")

            # Always parse final_prompt from the model
            final_prompt = result.get("final_prompt", instruction)
            final_prompt = final_prompt.strip('"').strip("'").strip()
            logging.info(
                f"LLM analysis: decision={decision}, output_len={len(final_prompt)}"
            )
            return final_prompt, decision

        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to parse LLM JSON response: {e}. Response: {response[:200]}"
            )
            # Fallback: treat as new request and use instruction as-is
            return instruction, "new_request"

    def rewrite_prompt(
        self,
        prompt_history: List[str],
        current_prompt: str,
        llm_call: Callable[[str], str],
    ) -> Tuple[str, str, str]:
        """
        Rewrite prompt by combining historical context with new instruction using LLM.

        Args:
            prompt_history: List of previous prompts (excluding current)
            current_prompt: Current user prompt/instruction
            llm_call: LLM callable for prompt rewriting

        Returns:
            Tuple of (rewritten_prompt, method_used, context_type) where:
            - rewritten_prompt: The final prompt to use
            - method_used: "llm", "none", or "error"
            - context_type: "modification", "new_request", or "none"
        """
        if not prompt_history or len(prompt_history) == 0:
            logging.info("No prompt history, using current prompt as-is")
            return current_prompt, "none", "none"

        # Get the last prompt from history as base
        base_prompt = prompt_history[-1]

        if not llm_call:
            logging.error("No LLM call provided, cannot rewrite prompt")
            return current_prompt, "none", "none"

        try:
            # Use LLM for classification and get a minimal-diff full rewrite when modifying
            rewritten, context_type = self._llm_rewrite(
                base_prompt, current_prompt, llm_call
            )
            return rewritten, "llm", context_type
        except Exception as e:
            logging.error(f"LLM rewrite failed: {str(e)}, using current prompt as-is")
            return current_prompt, "error", "none"

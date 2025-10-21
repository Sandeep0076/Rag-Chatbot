"""
Image prompt rewriter for combining historical context with new instructions.
Uses LLM-based rewriting to intelligently merge previous image context with new modifications.
Works with prompt_history array similar to chat history.
"""

import logging
import re
from typing import Callable, List


class ImagePromptRewriter:
    """Rewrites image prompts by combining historical context with new instructions using LLM."""

    def _llm_rewrite(
        self,
        base_prompt: str,
        instruction: str,
        llm_call: Callable[[str], str],
    ) -> str:
        """
        LLM-based prompt rewriting that intelligently merges context with new instruction.

        Args:
            base_prompt: Original full prompt from history
            instruction: User's new instruction
            llm_call: Callable that takes combined prompt and returns LLM response

        Returns:
            Rewritten prompt from LLM

        Raises:
            Exception: If LLM call fails
        """
        # Construct prompt with explicit decision-making while avoiding content filter triggers
        user_prompt = (
            f"You are helping refine an image generation prompt.\n\n"
            f"CONTEXT - Previous image was generated with this prompt:\n"
            f'"{base_prompt}"\n\n'
            f"USER'S NEW REQUEST:\n"
            f'"{instruction}"\n\n'
            f"YOUR TASK:\n"
            f"Step 1: Analyze the user's request and decide:\n"
            f"  • Is this a MODIFICATION of the previous image? "
            f"(Examples: 'make it blue', 'add a tree', 'change background to sunset', 'remove the hat')\n"
            f"  • OR is this a COMPLETELY NEW image request? "
            f"(Examples: 'create a forest scene', 'draw a robot', 'generate a logo', 'make a portrait')\n\n"
            f"Step 2: Based on your decision:\n"
            f"  • If MODIFICATION: Merge the previous prompt with the new request. "
            f"Preserve all original details (scene, objects, style, mood, lighting) "
            f"and apply only the specific change requested.\n"
            f"  • If NEW REQUEST: Use the new request as-is. "
            f"The previous prompt is not relevant.\n\n"
            f"IMPORTANT:\n"
            f"  • For modifications, create a complete standalone prompt with all details explicitly stated\n"
            f"  • For new requests, output the new request directly\n"
            f"  • Keep prompts clear and detailed\n"
            f"  • Output ONLY the final prompt text, no explanations\n\n"
            f"FINAL PROMPT:"
        )

        # Call LLM
        rewritten = llm_call(user_prompt)

        # Clean up response
        rewritten = rewritten.strip()

        # Remove common prefixes if LLM added them
        rewritten = re.sub(
            r"^(final prompt:|prompt:|here's the prompt:|the final prompt is:)\s*",
            "",
            rewritten,
            flags=re.IGNORECASE,
        ).strip()

        logging.info(
            f"LLM rewrite completed: input={len(instruction)} chars, output={len(rewritten)} chars"
        )
        return rewritten

    def rewrite_prompt(
        self,
        prompt_history: List[str],
        current_prompt: str,
        llm_call: Callable[[str], str],
    ) -> tuple[str, str, str]:
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
            # Always use LLM for rewriting
            rewritten = self._llm_rewrite(base_prompt, current_prompt, llm_call)

            # Determine if this was treated as a modification or new request
            # by checking if the rewritten prompt is significantly different from current_prompt
            context_type = self._determine_context_type(
                base_prompt, current_prompt, rewritten
            )

            return rewritten, "llm", context_type
        except Exception as e:
            logging.error(f"LLM rewrite failed: {str(e)}, using current prompt as-is")
            return current_prompt, "error", "none"

    def _determine_context_type(
        self, base_prompt: str, current_prompt: str, rewritten_prompt: str
    ) -> str:
        """
        Determine if the LLM treated this as a modification or new request.

        Args:
            base_prompt: Original prompt from history
            current_prompt: User's current instruction
            rewritten_prompt: LLM's output

        Returns:
            "modification" if treated as modification, "new_request" if treated as new
        """
        # Simple heuristic: if rewritten prompt is very similar to current_prompt,
        # it was likely treated as a new request
        current_words = set(current_prompt.lower().split())
        rewritten_words = set(rewritten_prompt.lower().split())

        # Calculate word overlap
        overlap = len(current_words.intersection(rewritten_words))
        total_current = len(current_words)

        # If most words from current prompt are in rewritten prompt, likely new request
        if total_current > 0 and overlap / total_current > 0.7:
            return "new_request"
        else:
            return "modification"

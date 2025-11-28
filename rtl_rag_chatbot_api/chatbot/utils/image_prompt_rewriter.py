"""
Image prompt rewriter for combining historical context with new instructions.
Uses LLM-based rewriting to intelligently merge previous image context with new modifications.
Works with prompt_history array similar to chat history.
"""

import json
import logging
from typing import Callable, List, Tuple

from rtl_rag_chatbot_api.common.prompts_storage import Image_prompt_rewriter_prompt


class ImagePromptRewriter:
    """Rewrites image prompts by combining historical context with new instructions using LLM."""

    @staticmethod
    def _clean_response(response: str) -> str:
        """Remove markdown code blocks and normalize JSON from response."""
        response = response.strip()

        # Remove markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            )
            response = response.strip()

        # Remove 'json' language identifier if present after backticks
        if response.startswith("json"):
            response = response[4:].strip()

        # If response starts with a field name but no opening brace, add it
        if not response.startswith("{") and (
            '"decision"' in response or '"final_prompt"' in response
        ):
            response = "{" + response

        # If response doesn't end with closing brace but has JSON fields, add it
        if not response.endswith("}") and (
            '"decision"' in response or '"final_prompt"' in response
        ):
            response = response + "}"

        return response.strip()

    @staticmethod
    def _generate_json_candidates(response: str) -> List[str]:
        """Generate multiple JSON parsing candidates from LLM response."""
        import re

        json_candidates = []

        # Strategy 1: Direct JSON if starts with {
        if response.startswith("{"):
            json_candidates.append(response)

        # Strategy 2: Extract from first { to last }
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = response[start_idx : end_idx + 1]
            if candidate not in json_candidates:
                json_candidates.append(candidate)

        # Strategy 3: Reconstruct if missing opening brace but has fields
        if '"decision"' in response and '"final_prompt"' in response:
            if not response.strip().startswith("{"):
                reconstructed = "{" + response.strip().lstrip("{").strip()
                if not reconstructed.endswith("}"):
                    reconstructed += "}"
                json_candidates.append(reconstructed)

                # Also try extracting fields via regex
                decision_match = re.search(
                    r'"decision"\s*:\s*"([^"]+)"', response, re.DOTALL
                )
                final_prompt_match = re.search(
                    r'"final_prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL
                )
                reasoning_match = re.search(
                    r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL
                )

                if decision_match and final_prompt_match:
                    reconstructed_obj = {
                        "decision": decision_match.group(1),
                        "final_prompt": final_prompt_match.group(1),
                    }
                    if reasoning_match:
                        reconstructed_obj["reasoning"] = reasoning_match.group(1)
                    json_candidates.append(json.dumps(reconstructed_obj))

        # Strategy 4: Regex to find complete JSON object
        pattern = re.search(
            r'\{[^{}]*"decision"[^{}]*"final_prompt"[^{}]*\}', response, re.DOTALL
        )
        if pattern:
            cand = pattern.group()
            if cand not in json_candidates:
                json_candidates.append(cand)

        # Strategy 5: Clean up formatting issues
        if response.count("{") >= 1 and response.count("}") >= 1:
            cleaned = re.sub(r"\n\s*", " ", response)
            cleaned = re.sub(r",\s*}", "}", cleaned)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            if cleaned not in json_candidates:
                json_candidates.append(cleaned)

        return json_candidates

    @staticmethod
    def _try_parse_json(candidates: List[str], original_response: str) -> dict:
        """Try to parse JSON from candidates, raise error if all fail."""
        parse_errors = []
        for i, candidate in enumerate(candidates):
            try:
                result = json.loads(candidate)
                logging.info(
                    f"Successfully parsed JSON candidate #{i + 1}/{len(candidates)}"
                )
                return result
            except json.JSONDecodeError as je:
                parse_errors.append(f"Candidate {i + 1}: {str(je)[:100]}")

        # All parsing attempts failed
        error_summary = "; ".join(parse_errors[:3])
        logging.error(
            f"All {len(candidates)} JSON parsing attempts failed. Errors: {error_summary}"
        )
        logging.error(
            f"Original response (first 500 chars): '{original_response[:500]}'"
        )
        raise json.JSONDecodeError("No valid JSON in candidates", original_response, 0)

    @staticmethod
    def _fallback_parse(response: str, instruction: str) -> Tuple[str, str]:
        """Fallback regex-based parsing when JSON parsing fails."""
        import re

        lower_resp = response.lower()
        has_modification = (
            "modification" in lower_resp and "new_request" not in lower_resp
        )
        has_new_request = "new_request" in lower_resp
        decision = (
            "modification"
            if has_modification
            else ("new_request" if has_new_request else "new_request")
        )

        # Try to extract final_prompt
        fp_match = re.search(r'"final_prompt"\s*:\s*"(.*?)"', response, re.DOTALL)
        if fp_match:
            final_prompt = fp_match.group(1).strip()
        else:
            # Attempt to capture after final_prompt key without braces
            fp_line = None
            for line in response.splitlines():
                if "final_prompt" in line:
                    fp_line = line
                    break
            if fp_line and '"' in fp_line:
                after_colon = fp_line.split(":", 1)[-1]
                qparts = after_colon.split('"')
                final_prompt = qparts[1].strip() if len(qparts) > 2 else instruction
            else:
                final_prompt = instruction

        final_prompt = final_prompt.strip().strip('"').strip("'") or instruction
        logging.info(
            f"Fallback parsing succeeded: decision={decision}, final_prompt='{final_prompt[:120]}'"
        )
        return final_prompt, decision

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
        # Call LLM
        user_prompt = Image_prompt_rewriter_prompt.format(
            base_prompt=base_prompt, instruction=instruction
        )

        response = llm_call(user_prompt)
        original_response = response
        response = self._clean_response(response)

        # Try primary JSON parsing
        try:
            json_candidates = self._generate_json_candidates(response)
            result = self._try_parse_json(json_candidates, original_response)

            decision = result.get("decision", "new_request")
            final_prompt = result.get("final_prompt", instruction) or instruction
            final_prompt = final_prompt.strip().strip('"').strip("'")
            return final_prompt, decision

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logging.error(f"Primary JSON parsing failed: {str(e)[:200]}")
            logging.error(
                f"Raw response (first 500 chars): '{original_response[:500]}'"
            )
            # Try fallback regex extraction
            try:
                return self._fallback_parse(response, instruction)
            except Exception as fb_err:
                logging.error(
                    f"Fallback parsing failed: {fb_err}. Using instruction as new request."
                )
                return instruction.strip(), "new_request"

    def rewrite_prompt(
        self,
        prompt_history: List[str],
        current_prompt: str,
        llm_call: Callable[[str], str],
    ) -> Tuple[str, str, str, bool]:
        """
        Rewrite prompt by combining historical context with new instruction using LLM.

        Args:
            prompt_history: List of previous prompts (excluding current)
            current_prompt: Current user prompt/instruction
            llm_call: LLM callable for prompt rewriting

        Returns:
            Tuple of (rewritten_prompt, method_used, context_type, is_edit_operation) where:
            - rewritten_prompt: The final prompt to use
            - method_used: "llm", "none", or "error"
            - context_type: "modification", "new_request", or "none"
            - is_edit_operation: True if this should use image-to-image editing (modification context)
        """
        if not prompt_history or len(prompt_history) == 0:
            logging.info("No prompt history, using current prompt as-is")
            return current_prompt, "none", "none", False

        # Get the last prompt from history as base
        base_prompt = prompt_history[-1]

        if not llm_call:
            logging.error("No LLM call provided, cannot rewrite prompt")
            return current_prompt, "none", "none", False

        # Use LLM for classification and get a minimal-diff full rewrite when modifying
        rewritten, context_type = self._llm_rewrite(
            base_prompt, current_prompt, llm_call
        )

        # Determine if this should be an edit operation (modification of previous image)
        is_edit_operation = context_type == "modification"

        return rewritten, "llm", context_type, is_edit_operation

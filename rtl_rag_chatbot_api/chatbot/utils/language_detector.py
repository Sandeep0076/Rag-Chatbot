"""
Language detection module using the lingua library.

This module provides a class-based approach to detect the language of text.
Supports multiple languages with English and German as primary languages.
"""

from typing import Optional

from lingua import Language, LanguageDetectorBuilder


class LanguageDetector:
    """
    Language detector class that uses lingua library to detect text language.

    Supports multiple languages with English and German as primary languages.
    The detector is built once at initialization for optimal performance.
    """

    def __init__(self, languages: Optional[list[Language]] = None):
        """
        Initialize the language detector.

        Args:
            languages: Optional list of Language enums to support.
                     If None, defaults to [Language.ENGLISH, Language.GERMAN]
        """
        if languages is None:
            languages = [Language.ENGLISH, Language.GERMAN]

        self.languages = languages
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()

        # Map Language enum to full language names
        self.language_name_map = {
            Language.ENGLISH: "English",
            Language.GERMAN: "German",
            Language.FRENCH: "French",
            Language.SPANISH: "Spanish",
            Language.ITALIAN: "Italian",
            Language.PORTUGUESE: "Portuguese",
            Language.DUTCH: "Dutch",
            Language.POLISH: "Polish",
            Language.RUSSIAN: "Russian",
            Language.CHINESE: "Chinese",
            Language.JAPANESE: "Japanese",
            Language.KOREAN: "Korean",
        }

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text and return full language name.

        Uses confidence scores and German structure markers to improve accuracy,
        especially for mixed-language text.

        Args:
            text: The text to detect language for

        Returns:
            Full language name (e.g., "English", "German", "French")
            Defaults to "German" if language cannot be detected or is not in the map
        """
        if not text or not text.strip():
            return "German"  # Default to German for empty text

        # Check for strong German sentence structure markers
        text_lower = text.lower()
        german_markers = [
            "was ist",
            "wie ist",
            "wo ist",
            "wann ist",
            "warum ist",
            "was sind",
            "wie sind",
            "wo sind",
            "wann sind",
            "warum sind",
            "der ",
            "die ",
            "das ",
            "und ",
            "für ",
            "mit ",
            "von ",
            "zu ",
            "ist ein",
            "ist eine",
            "sind ",
            "haben ",
            "hat ",
            "kann ",
        ]
        has_german_structure = any(marker in text_lower for marker in german_markers)

        # Use confidence scores for more accurate detection
        detected_lang = None
        confidence = 0.0

        try:
            confidence_values = self.detector.compute_language_confidence_values(text)

            if confidence_values:
                # Get the language with the highest confidence
                detected_lang = confidence_values[0].language
                confidence = confidence_values[0].value
        except (AttributeError, IndexError, TypeError):
            # Fallback to simple detection if confidence method is not available
            pass

        # If confidence method didn't work, use simple detection
        if detected_lang is None:
            detected_lang = self.detector.detect_language_of(text)

        # Override detection if German structure is detected but lingua says English
        # This handles cases like "Was ist ein Manatee" where structure is German
        if (
            has_german_structure
            and detected_lang == Language.ENGLISH
            and confidence < 0.8
        ):
            detected_lang = Language.GERMAN

        if detected_lang is None:
            return "German"  # Default to German if detection fails

        # Return the language name if found in map, otherwise default to German
        return self.language_name_map.get(detected_lang, "German")

    def detect_language_object(self, text: str) -> Optional[Language]:
        """
        Detect the language of the given text and return the Language enum object.

        Args:
            text: The text to detect language for

        Returns:
            Language enum object or None if detection fails
        """
        if not text or not text.strip():
            return None

        return self.detector.detect_language_of(text)


# Create a default instance for convenience
_default_detector = LanguageDetector()


def detect_lang(text: str) -> str:
    """
    Convenience function to detect language using the default detector.

    Args:
        text: The text to detect language for

    Returns:
        Full language name (e.g., "English", "German")
    """
    return _default_detector.detect_language(text)


# # Test cases
# if __name__ == "__main__":
#     test_texts = [
#         "Was ist ein Manatee",
#         "Was ist ein Seetier",
#         "Hello, how are you?",
#         "Ich liebe dich.",
#         "Wie geht's?",
#         "Das ist gut.",
#         "Ich verstehe nicht.",
#         "Wo ist die Toilette?",
#         "Guten Tag, wie geht es dir?",
#     ]
#     for text in test_texts:
#         detected_obj = _default_detector.detector.detect_language_of(text)
#         result = detect_lang(text)
#         print(f"Text: '{text}'")
#         print(f"  Detected Language object: {detected_obj}")
#         print(f"  Result: {result}")
#         print()

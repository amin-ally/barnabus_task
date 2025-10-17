# File: src/data_processing/preprocessor.py

import re
import string
from typing import List, Optional, Dict, Tuple, Set
import emoji
import numpy as np
from parsivar import Normalizer, Tokenizer


class MultilingualTextPreprocessor:
    """Preprocess text for classification and embedding (English + Farsi)"""

    def __init__(
        self,
        lower_case: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_emojis: bool = False,
        normalize_farsi: bool = True,
        enable_pii_masking: bool = True,
        max_length: int = 512,
    ):
        self.lower_case = lower_case
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_emojis = remove_emojis
        self.normalize_farsi = normalize_farsi
        self.enable_pii_masking = enable_pii_masking
        self.max_length = max_length

        if self.normalize_farsi:
            self.farsi_normalizer = Normalizer()
            self.farsi_tokenizer = Tokenizer()

        self.farsi_chars = (
            r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
        )

    def detect_language(self, text: str) -> str:
        """Simple language detection based on character sets"""
        farsi_count = len(re.findall(self.farsi_chars, text))
        total_chars = len(text)
        if total_chars == 0:
            return "en"
        return "fa" if farsi_count / total_chars > 0.2 else "en"

    def clean_english_text(self, text: str) -> str:
        """Clean English text"""
        if self.remove_urls:
            text = re.sub(r"http\S+|www.\S+", "[URL]", text)

        if self.remove_mentions:
            text = re.sub(r"@\w+", "[USER_MENTION]", text)

        if self.remove_hashtags:
            text = re.sub(r"#\w+", "", text)
        else:
            text = re.sub(r"#(\w+)", r"\1", text)

        if self.remove_emojis:
            text = emoji.replace_emoji(text, "")
        else:
            text = emoji.demojize(text)

        text = " ".join(text.split())
        if self.lower_case:
            text = text.lower()
        return text.strip()

    def clean_farsi_text(self, text: str) -> str:
        """Clean Farsi/Persian text"""
        if self.normalize_farsi and hasattr(self, "farsi_normalizer"):
            text = self.farsi_normalizer.normalize(text)

        if self.remove_urls:
            text = re.sub(r"http\S+|www.\S+", "[URL]", text)

        if self.remove_mentions:
            text = re.sub(r"@[\w\u0600-\u06FF]+", "[USER_MENTION]", text)

        if self.remove_hashtags:
            text = re.sub(r"#[\w\u0600-\u06FF]+", "", text)
        else:
            text = re.sub(r"#([\w\u0600-\u06FF]+)", r"\1", text)

        if self.remove_emojis:
            text = emoji.replace_emoji(text, "")
        else:
            text = emoji.demojize(text)

        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
        text = re.sub(r"\u200c+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean_text(self, text: str, language: Optional[str] = None) -> str:
        """Clean text based on detected or specified language"""
        if language is None:
            language = self.detect_language(text)

        cleaned_text = (
            self.clean_farsi_text(text)
            if language == "fa"
            else self.clean_english_text(text)
        )

        words = cleaned_text.split()
        if len(words) > self.max_length:
            cleaned_text = " ".join(words[: self.max_length])
        return cleaned_text.strip()

    @staticmethod
    def normalize_numerals(text: str) -> str:
        """Normalize Persian/Arabic numerals to ASCII for easier matching."""
        numeral_map = {
            ord("۰"): "0",
            ord("۱"): "1",
            ord("۲"): "2",
            ord("۳"): "3",
            ord("۴"): "4",
            ord("۵"): "5",
            ord("۶"): "6",
            ord("۷"): "7",
            ord("۸"): "8",
            ord("۹"): "9",
        }
        return text.translate(numeral_map)

    def mask_pii(self, text: str) -> Tuple[str, List[str]]:
        """
        Masks common PII (emails, phones, credit cards, user mentions) in the text.

        This function uses regular expressions to find and replace PII. It is designed
        to be a lightweight, first-pass filter.

        Args:
            text: The input string to be masked.

        Returns:
            A tuple containing:
            - masked_text (str): The text with PII replaced by placeholders.
            - detected_types (List[str]): A sorted list of unique PII types found.
        """
        if not self.enable_pii_masking:
            return text, []

        masked_text = self.normalize_numerals(text)
        detected_types: Set[str] = set()

        # 1. Email Masking
        email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        if re.search(email_pattern, masked_text):
            detected_types.add("EMAIL")
        masked_text = re.sub(email_pattern, "[EMAIL]", masked_text)

        # 2. Phone Number Masking (EN/Persian)
        phone_patterns = [
            r"\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})?[-. )]*(\d{3})[-. ]*(\d{4})\b",
            r"\b09\d{9}\b",
            r"\b(?:\+98|0)?\d{10}\b",
            r"\b(?:\+98|0)?\s?\d{2,3}\s?\d{3,4}\s?\d{4}\b",
        ]
        phone_regex = re.compile("|".join(phone_patterns))
        if phone_regex.search(masked_text):
            detected_types.add("PHONE")
        masked_text = phone_regex.sub("[PHONE]", masked_text)

        # 3. Credit Card Masking
        cc_pattern = r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
        if re.search(cc_pattern, masked_text):
            detected_types.add("CREDIT_CARD")
        masked_text = re.sub(cc_pattern, "[CREDIT_CARD]", masked_text)

        # 4. User Mention Masking (additional PII protection)
        mention_pattern = r"@\w+"
        if re.search(mention_pattern, masked_text):
            detected_types.add("USER_MENTION")
        masked_text = re.sub(mention_pattern, "[USER_MENTION]", masked_text)

        return masked_text, sorted(list(detected_types))

    def preprocess_for_service(
        self, text: str, language: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Preprocess text for service use: clean, mask PII, and return structured dict.

        Args:
            text: Input text to preprocess
            language: Optional language hint ('en' or 'fa')

        Returns:
            Dictionary with keys: 'original', 'cleaned', 'masked', 'lang', 'pii_detected'

        Raises:
            ValueError: If input exceeds max_length threshold
        """
        # Rough character-based check to prevent processing overly long inputs
        if len(text) > self.max_length * 6:
            raise ValueError(
                f"Input character count is too high. Max words: {self.max_length}"
            )

        if language is None:
            language = self.detect_language(text)

        cleaned = self.clean_text(text, language)
        masked, pii_types = self.mask_pii(cleaned)

        return {
            "original": text,
            "cleaned": cleaned,
            "masked": masked,
            "lang": language,
            "pii_detected": pii_types,
        }

    def process_batch(
        self, texts: List[str], languages: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Process a batch of texts, returning list of dicts for service use."""
        if languages is None:
            languages = [None] * len(texts)

        return [
            self.preprocess_for_service(text, lang)
            for text, lang in zip(texts, languages)
        ]

    def get_text_stats(
        self, texts: List[str], languages: Optional[List[str]] = None
    ) -> Dict:
        """Get statistics about the text data"""
        if languages is None:
            languages = [self.detect_language(text) for text in texts]

        stats = {
            "total_samples": len(texts),
            "language_distribution": {},
        }

        lengths = [len(text.split()) for text in texts]
        stats["overall"] = {
            "avg_length": np.mean(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "std_length": np.std(lengths) if lengths else 0,
        }

        for lang in set(languages):
            lang_texts = [t for t, l in zip(texts, languages) if l == lang]
            lang_lengths = [len(text.split()) for text in lang_texts]
            stats[f"{lang}_stats"] = {
                "count": len(lang_texts),
                "avg_length": np.mean(lang_lengths) if lang_lengths else 0,
                "max_length": max(lang_lengths) if lang_lengths else 0,
                "min_length": min(lang_lengths) if lang_lengths else 0,
                "std_length": np.std(lang_lengths) if lang_lengths else 0,
            }
        return stats

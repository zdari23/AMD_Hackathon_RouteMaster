import unittest

from src.optimizer.parser import (
    asks_for_explanation,
    asks_for_steps,
    detect_code_language,
    detect_output_format,
    extract_allowed_sentiment_labels,
    extract_bullet_count,
    extract_code_block,
    extract_requested_entity_types,
    extract_sentence_limit,
    extract_word_limit,
    get_instruction_view,
)


class InstructionParserTests(unittest.TestCase):
    def test_explicit_limits(self):
        self.assertEqual(extract_word_limit("Summarize in no more than 120 words."), (120, False, False))
        self.assertEqual(extract_word_limit("Use exactly 50 words."), (50, True, False))
        self.assertEqual(extract_sentence_limit("Use exactly 3 sentences."), (3, True))
        self.assertEqual(extract_bullet_count("Return exactly 4 bullet points."), 4)

    def test_output_requirements(self):
        prompt = "Explain your reasoning and show your work. Return a JSON object."
        self.assertTrue(asks_for_explanation(prompt))
        self.assertTrue(asks_for_steps(prompt))
        self.assertEqual(detect_output_format(prompt), "json")

    def test_negated_explanation_and_steps_are_not_requested(self):
        prompt = "Return the final answer only. Do not include any explanation or reasoning."
        self.assertFalse(asks_for_explanation(prompt))
        self.assertFalse(asks_for_steps(prompt))

    def test_identify_bug_counts_as_explanation_request(self):
        self.assertTrue(asks_for_explanation("Identify the bug and provide corrected code."))

    def test_language_and_code_block(self):
        prompt = "Fix this JavaScript function.\n```javascript\nfunction x() { return 1; }\n```"
        self.assertEqual(detect_code_language(prompt), "javascript")
        self.assertIn("function x", extract_code_block(prompt))
        self.assertNotIn("function x", get_instruction_view(prompt))

    def test_labels_and_entity_types(self):
        self.assertEqual(
            extract_allowed_sentiment_labels("Use Positive, Negative, Neutral, or Mixed."),
            ("Positive", "Negative", "Neutral", "Mixed"),
        )
        self.assertEqual(
            extract_requested_entity_types("Extract persons, organizations, locations, and dates."),
            ("person", "organization", "location", "date"),
        )

import unittest

from src.optimizer import validate_output


class CategoryValidatorTests(unittest.TestCase):
    def test_sentiment_valid_and_invalid(self):
        prompt = "Classify as Positive, Negative, or Neutral. Return the label only."
        self.assertTrue(validate_output("sentiment_analysis", prompt, "Positive", "stop").valid)
        self.assertFalse(validate_output("sentiment_analysis", prompt, "Happy", "stop").valid)

    def test_ner_malformed_and_hallucinated(self):
        prompt = "Extract persons as JSON.\n\nText:\nAlice met Bob."
        self.assertFalse(validate_output("entity_extraction", prompt, "{bad", "stop").valid)
        result = validate_output("entity_extraction", prompt, '{"person":["Mallory"]}', "stop")
        self.assertFalse(result.valid)
        self.assertTrue(any("not found" in error for error in result.errors))

    def test_fenced_json_is_repaired_locally(self):
        prompt = "Extract persons as JSON.\n\nText:\nAlice met Bob."
        result = validate_output("entity_extraction", prompt, '```json\n{"person":["Alice","Alice"]}\n```', "stop")
        self.assertTrue(result.valid)
        self.assertEqual(result.repaired_output, '{"person":["Alice"]}')

    def test_python_syntax_error(self):
        prompt = "Fix this Python code.\n```python\ndef add(a, b):\n return a + b\n```"
        result = validate_output("bug_fixing", prompt, "def add(a, b)\n    return a + b", "stop")
        self.assertFalse(result.valid)
        self.assertTrue(any("syntax error" in error for error in result.errors))

    def test_wrong_fenced_language(self):
        prompt = "Implement this C++ method `add`."
        result = validate_output("code_authoring", prompt, "```python\ndef add(a, b): return a + b\n```", "stop")
        self.assertFalse(result.valid)
        self.assertTrue(any("wrong programming language" in error for error in result.errors))

    def test_truncated_code(self):
        result = validate_output("code_authoring", "Write a Python function `f()`.", "def f():\n    return 1", "length")
        self.assertFalse(result.valid)

    def test_math_requires_numeric_result(self):
        result = validate_output("math_solving", "Calculate the total.", "I could not calculate it.", "stop")
        self.assertFalse(result.valid)

    def test_wrong_bullet_and_word_counts(self):
        bullets = validate_output("summarization", "Summarize in exactly 3 bullet points.", "- one\n- two", "stop")
        self.assertFalse(bullets.valid)
        words = validate_output("summarization", "Summarize in no more than 3 words.", "one two three four", "stop")
        self.assertFalse(words.valid)

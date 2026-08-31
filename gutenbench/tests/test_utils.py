import unittest

from gutenbench.utils import convert_osis_to_plain


class ConvertOsisToPlainTests(unittest.TestCase):
    def test_single_verse(self):
        self.assertEqual(convert_osis_to_plain("GEN.2.8", "GEN.2.8"), "Genesis 2:8")

    def test_same_chapter(self):
        self.assertEqual(convert_osis_to_plain("GEN.2.8", "GEN.2.25"), "Genesis 2:8-25")

    def test_multiple_chapters(self):
        self.assertEqual(
            convert_osis_to_plain("GEN.1.1", "GEN.2.7"),
            "Genesis 1:1-31; Genesis 2:1-7",
        )

    def test_raises_for_cross_book_range(self):
        with self.assertRaisesRegex(ValueError, "must stay within one book"):
            convert_osis_to_plain("GEN.1.1", "EXO.1.3")

    def test_raises_for_reversed_range(self):
        with self.assertRaisesRegex(ValueError, "Invalid verse range"):
            convert_osis_to_plain("GEN.2.8", "GEN.2.7")

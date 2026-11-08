import unittest
from solution import is_balanced


class BalancedTests(unittest.TestCase):
    def test_plain(self):
        self.assertTrue(is_balanced(""))
        self.assertTrue(is_balanced("abc"))
        self.assertTrue(is_balanced("()"))
        self.assertTrue(is_balanced("()[]{}"))
        self.assertTrue(is_balanced("([{}])"))
        self.assertTrue(is_balanced("a(b[c]{d}e)f"))

    def test_unbalanced(self):
        for text in ["(", ")", ")(", "(]", "([)]", "{[}]", "((", "(()", "())", "]"]:
            with self.subTest(text=text):
                self.assertFalse(is_balanced(text))

    def test_strings_hide_brackets(self):
        self.assertTrue(is_balanced('f(a[1], {"k": "]"})'))
        self.assertTrue(is_balanced('")"'))
        self.assertTrue(is_balanced('("(")'))
        self.assertTrue(is_balanced('"(" + ")"'))
        self.assertTrue(is_balanced('"" ()'))
        self.assertFalse(is_balanced('("a)'))
        self.assertFalse(is_balanced('(")")('))

    def test_escapes(self):
        self.assertTrue(is_balanced(r'("a\"b)")'))
        self.assertTrue(is_balanced(r'"\\"'))
        self.assertTrue(is_balanced(r'("\\")'))
        self.assertTrue(is_balanced(r'("\\" ")")'))
        self.assertFalse(is_balanced(r'"abc\"'))
        self.assertFalse(is_balanced(r'"abc\\\"'))
        self.assertFalse(is_balanced(r'("\\")")'))

    def test_unterminated_string(self):
        self.assertFalse(is_balanced('"abc'))
        self.assertFalse(is_balanced('()"'))
        self.assertFalse(is_balanced('"\\'))

    def test_single_quotes_are_ordinary(self):
        self.assertFalse(is_balanced("'('"))
        self.assertTrue(is_balanced("'()'"))


if __name__ == "__main__":
    unittest.main()

import unittest
from solution import parse_csv_line


class CsvLineTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_csv_line("a,b,c"), ["a", "b", "c"])
        self.assertEqual(parse_csv_line("abc"), ["abc"])
        self.assertEqual(parse_csv_line(" a , b "), [" a ", " b "])

    def test_empty_fields(self):
        self.assertEqual(parse_csv_line(""), [""])
        self.assertEqual(parse_csv_line("a,,b"), ["a", "", "b"])
        self.assertEqual(parse_csv_line("a,"), ["a", ""])
        self.assertEqual(parse_csv_line(",a"), ["", "a"])
        self.assertEqual(parse_csv_line(","), ["", ""])
        self.assertEqual(parse_csv_line(",,"), ["", "", ""])

    def test_quoted(self):
        self.assertEqual(parse_csv_line('"a,b",c'), ["a,b", "c"])
        self.assertEqual(parse_csv_line('x,"y z",w'), ["x", "y z", "w"])
        self.assertEqual(parse_csv_line('""'), [""])
        self.assertEqual(parse_csv_line('"",""'), ["", ""])
        self.assertEqual(parse_csv_line('"a",'), ["a", ""])

    def test_escaped_quotes(self):
        self.assertEqual(parse_csv_line('"say ""hi"""'), ['say "hi"'])
        self.assertEqual(parse_csv_line('"a""b",c'), ['a"b', "c"])
        self.assertEqual(parse_csv_line('""""'), ['"'])

    def test_quote_inside_unquoted_field_is_literal(self):
        self.assertEqual(parse_csv_line('ab"cd,e'), ['ab"cd', "e"])
        self.assertEqual(parse_csv_line('5" pipe,x'), ['5" pipe', "x"])

    def test_errors(self):
        for text in ['"abc', '"a,b', 'a,"b', '"', '"a"b', '"a" ,b', '"a""']:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_csv_line(text)

    def test_type(self):
        with self.assertRaises(TypeError):
            parse_csv_line(None)


if __name__ == "__main__":
    unittest.main()

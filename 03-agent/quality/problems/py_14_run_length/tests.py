import unittest
from solution import rle_decode, rle_encode


class RunLengthTests(unittest.TestCase):
    def test_encode(self):
        self.assertEqual(rle_encode("aaabcc"), "3a1b2c")
        self.assertEqual(rle_encode("x"), "1x")
        self.assertEqual(rle_encode(""), "")
        self.assertEqual(rle_encode("abc"), "1a1b1c")
        self.assertEqual(rle_encode("aaaaaaaaaaaa"), "12a")
        self.assertEqual(rle_encode("aabbaa"), "2a2b2a")
        self.assertEqual(rle_encode("  !!"), "2 2!")
        self.assertEqual(rle_encode("a" * 100 + "b"), "100a1b")

    def test_encode_rejects_digits(self):
        for text in ["a1", "1", "abc9def", "007"]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    rle_encode(text)

    def test_decode(self):
        self.assertEqual(rle_decode("3a1b2c"), "aaabcc")
        self.assertEqual(rle_decode("1x"), "x")
        self.assertEqual(rle_decode(""), "")
        self.assertEqual(rle_decode("12x"), "x" * 12)
        self.assertEqual(rle_decode("2a2a"), "aaaa")
        self.assertEqual(rle_decode("2 1!"), "  !")
        self.assertEqual(rle_decode("10a2b"), "a" * 10 + "bb")
        self.assertEqual(rle_decode("01a"), "a")

    def test_decode_rejects_malformed(self):
        for text in ["0a", "abc", "a3", "3", "2a3", "00a", "a", "1", "3a0b", "x1y"]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    rle_decode(text)

    def test_round_trip(self):
        for text in ["", "a", "aaaa", "abab", "hello world", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzyyx", "..--..", "\n\n\t"]:
            with self.subTest(text=text):
                self.assertEqual(rle_decode(rle_encode(text)), text)

    def test_type(self):
        for fn in (rle_encode, rle_decode):
            for bad in (None, 5, b"3a", ["a"]):
                with self.subTest(fn=fn.__name__, bad=bad):
                    with self.assertRaises(TypeError):
                        fn(bad)


if __name__ == "__main__":
    unittest.main()

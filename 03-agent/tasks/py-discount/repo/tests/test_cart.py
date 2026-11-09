import unittest
from decimal import Decimal

from shop.cart import Cart


class CartTests(unittest.TestCase):
    def test_total_of_plain_lines(self):
        c = Cart()
        c.add("pen", "1.50", 2)
        c.add("book", 12)
        self.assertEqual(c.total(), Decimal("15.00"))

    def test_bad_quantity(self):
        with self.assertRaises(ValueError):
            Cart().add("pen", 1, 0)

    def test_discount_reduces_the_line(self):
        c = Cart()
        c.add("pen", "2.00", 3, discount=10)
        c.add("book", "10.00")
        self.assertEqual(c.total(), Decimal("15.40"))
        self.assertEqual(c.lines[0]["discount"], Decimal("10"))

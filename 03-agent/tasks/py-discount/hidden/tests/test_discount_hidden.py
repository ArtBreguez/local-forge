import unittest
from decimal import Decimal

from shop.cart import Cart
from shop.receipt import render


class DiscountHiddenTests(unittest.TestCase):
    def test_line_total_rounds_half_up(self):
        c = Cart()
        c.add("gum", "0.05", 1, discount=10)        # 0.045 -> 0.05
        self.assertEqual(c.line_total(c.lines[0]), Decimal("0.05"))
        c.add("tea", "9.99", 1, discount="12.5")    # 8.74125 -> 8.74
        self.assertEqual(c.line_total(c.lines[1]), Decimal("8.74"))
        self.assertEqual(c.total(), Decimal("8.79"))

    def test_zero_and_full_discount(self):
        c = Cart()
        c.add("a", "4.00", 2, discount=0)
        c.add("b", "4.00", 2, discount=100)
        self.assertEqual(c.line_total(c.lines[0]), Decimal("8.00"))
        self.assertEqual(c.line_total(c.lines[1]), Decimal("0.00"))
        self.assertEqual(c.total(), Decimal("8.00"))

    def test_discount_stored_as_decimal(self):
        c = Cart()
        c.add("a", 1, 1, discount=5)
        c.add("b", 1, 1, discount="2.5")
        c.add("c", 1)
        self.assertEqual([l["discount"] for l in c.lines], [Decimal("5"), Decimal("2.5"), Decimal("0")])
        self.assertTrue(all(isinstance(l["discount"], Decimal) for l in c.lines))

    def test_discount_out_of_range(self):
        for bad in (-1, 100.5, 101, "-0.1"):
            with self.assertRaises(ValueError):
                Cart().add("x", 1, 1, discount=bad)

    def test_receipt_percent_formatting(self):
        c = Cart()
        c.add("Tea", "9.99", 1, discount="12.5")
        c.add("Gum", "0.05", 1, discount=10)
        c.add("Pen", "1.80", 1)
        self.assertEqual(render(c), "1 x Tea -12.5%  8.74\n1 x Gum -10%  0.05\n1 x Pen  1.80\nTOTAL  10.59")

    def test_whole_cart_without_discounts_unchanged(self):
        c = Cart()
        c.add("Pen", "1.80", 2)
        c.add("Ink", "3.10")
        self.assertEqual(render(c), "2 x Pen  3.60\n1 x Ink  3.10\nTOTAL  6.70")

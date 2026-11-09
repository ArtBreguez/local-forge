import unittest

from shop.cart import Cart
from shop.receipt import render


class ReceiptTests(unittest.TestCase):
    def test_plain_receipt(self):
        c = Cart()
        c.add("Pen", "1.80", 2)
        self.assertEqual(render(c), "2 x Pen  3.60\nTOTAL  3.60")

    def test_receipt_shows_the_discount(self):
        c = Cart()
        c.add("Pen", "2.00", 2, discount=10)
        self.assertEqual(render(c), "2 x Pen -10%  3.60\nTOTAL  3.60")

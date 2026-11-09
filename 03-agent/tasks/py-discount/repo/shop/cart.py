"""A shopping cart."""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")


class Cart:
    def __init__(self):
        self.lines = []

    def add(self, name, price, qty=1):
        """Add `qty` units of `name` at `price` (a number or a string such as "9.99")."""
        if qty < 1:
            raise ValueError("qty must be >= 1")
        self.lines.append({"name": name, "price": Decimal(str(price)), "qty": qty})

    def line_total(self, line):
        """Price x quantity of one line."""
        return line["price"] * line["qty"]

    def total(self):
        """Sum of all line totals, rounded half up to 2 decimals."""
        raw = sum((self.line_total(line) for line in self.lines), Decimal("0"))
        return raw.quantize(CENT, rounding=ROUND_HALF_UP)

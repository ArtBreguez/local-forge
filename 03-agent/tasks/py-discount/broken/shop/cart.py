"""A shopping cart."""
from decimal import Decimal, ROUND_HALF_UP

CENT = Decimal("0.01")
HUNDRED = Decimal(100)


class Cart:
    def __init__(self):
        self.lines = []

    def add(self, name, price, qty=1, discount=0):
        """Add `qty` units of `name` at `price`; `discount` is a percentage from 0 to 100."""
        if qty < 1:
            raise ValueError("qty must be >= 1")
        discount = Decimal(str(discount))
        if discount < 0 or discount > 100:
            raise ValueError("discount must be between 0 and 100")
        self.lines.append({"name": name, "price": Decimal(str(price)), "qty": qty, "discount": discount})

    def line_total(self, line):
        """Price x quantity of one line, less its discount, rounded half up to 2 decimals."""
        gross = line["price"] * line["qty"]
        net = gross * (HUNDRED - line["discount"]) / HUNDRED
        return net.quantize(CENT, rounding=ROUND_HALF_UP)

    def total(self):
        """Sum of all line totals."""
        return sum((self.line_total(line) for line in self.lines), Decimal("0")).quantize(CENT)

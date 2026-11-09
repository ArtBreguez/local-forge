Add per-line percentage discounts to the small shop package: `shop/cart.py` and `shop/receipt.py` both need a change.

- `Cart.add(name, price, qty=1, discount=0)`: `discount` is a percentage from 0 to 100 (an int, a float or a string such as `"12.5"`); anything outside that range raises `ValueError`. Store it on the line under the key `"discount"` as a `Decimal` (convert with `Decimal(str(discount))`).
- `Cart.line_total(line)` applies the discount to price x qty and rounds the result half up to 2 decimals (so a line without a discount is also rounded to 2 decimals). `Cart.total()` is the sum of the line totals.
- `receipt.render(cart)`: a line with a discount shows it after the name as `-<percent>%`, written without trailing zeros: `-10%`, `-12.5%`. A line without a discount (0) looks as before. Example: `2 x Pen -10%  3.60`.

The visible tests are in `tests/` (run them with `python -m unittest`); more tests that you cannot see check the same behaviour. Do not modify anything under `tests/`.

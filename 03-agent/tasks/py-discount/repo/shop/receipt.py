"""Plain text receipts."""


def render(cart):
    """One line per cart line ("<qty> x <name>  <line total>"), then "TOTAL  <total>"."""
    out = []
    for line in cart.lines:
        out.append(f"{line['qty']} x {line['name']}  {cart.line_total(line):.2f}")
    out.append(f"TOTAL  {cart.total():.2f}")
    return "\n".join(out)

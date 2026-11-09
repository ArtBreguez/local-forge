"""Plain text receipts."""


def _percent(value):
    text = f"{value:f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def render(cart):
    """One line per cart line ("<qty> x <name>[ -<percent>%]  <line total>"), then "TOTAL  <total>"."""
    out = []
    for line in cart.lines:
        label = f"{line['qty']} x {line['name']}"
        if line["discount"]:
            label += f" -{_percent(line['discount'])}%"
        out.append(f"{label}  {cart.line_total(line):.2f}")
    out.append(f"TOTAL  {cart.total():.2f}")
    return "\n".join(out)

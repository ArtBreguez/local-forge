"""Plain text receipts."""


def render(cart):
    out = []
    for line in cart.lines:
        label = f"{line['qty']} x {line['name']}"
        if line["discount"]:
            label += f" -{line['discount']:.1f}%"
        out.append(f"{label}  {cart.line_total(line):.2f}")
    out.append(f"TOTAL  {cart.total():.2f}")
    return "\n".join(out)

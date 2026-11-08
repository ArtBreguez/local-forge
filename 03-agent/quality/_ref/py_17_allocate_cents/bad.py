def allocate(total, weights):
    if total < 0:
        raise ValueError("total must not be negative")
    if not weights:
        raise ValueError("weights must not be empty")
    if any(w < 0 for w in weights):
        raise ValueError("weights must not be negative")
    s = sum(weights)
    if s == 0:
        raise ValueError("weights must not all be zero")
    shares = [total * w // s for w in weights]
    left = total - sum(shares)
    for i in range(left):
        shares[i] += 1
    return shares

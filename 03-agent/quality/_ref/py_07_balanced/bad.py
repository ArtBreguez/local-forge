def is_balanced(code):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in code:
        if c in "([{":
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack

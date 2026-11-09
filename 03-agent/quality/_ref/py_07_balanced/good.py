def is_balanced(code):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if c == '"':
            i += 1
            while True:
                if i >= n:
                    return False
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == '"':
                    break
                i += 1
        elif c in "([{":
            stack.append(c)
        elif c in pairs:
            if not stack or stack.pop() != pairs[c]:
                return False
        i += 1
    return not stack

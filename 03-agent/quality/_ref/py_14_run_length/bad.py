def rle_encode(s):
    if not isinstance(s, str):
        raise TypeError("s must be a str")
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{j - i}{s[i]}")
        i = j
    return "".join(out)


def rle_decode(s):
    if not isinstance(s, str):
        raise TypeError("s must be a str")
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(int(ch))
        else:
            out.append(ch * out.pop() if out and isinstance(out[-1], int) else ch)
    return "".join(x for x in out if isinstance(x, str))

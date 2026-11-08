def wrap_text(text, width):
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an int")
    if width < 1:
        raise ValueError("width must be at least 1")
    lines = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

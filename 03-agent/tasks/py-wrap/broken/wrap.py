def wrap(text, width):
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("width must be an int >= 1")
    lines = []
    current = ""
    for word in text.split():
        if len(word) > width:
            if current:
                lines.append(current)
                current = ""
            for i in range(0, len(word), width):
                lines.append(word[i:i + width])
        elif not current:
            current = word
        elif len(current) + 1 + len(word) < width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

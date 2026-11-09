def parse_csv_line(line):
    if not isinstance(line, str):
        raise TypeError("line must be a str")
    fields = []
    i, n = 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1
            buf = []
            while True:
                if i >= n:
                    raise ValueError("unterminated quoted field")
                c = line[i]
                if c == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        buf.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                buf.append(c)
                i += 1
            fields.append("".join(buf))
            if i >= n:
                return fields
            if line[i] != ",":
                raise ValueError("unexpected character after closing quote")
            i += 1
        else:
            j = line.find(",", i)
            if j == -1:
                fields.append(line[i:])
                return fields
            fields.append(line[i:j])
            i = j + 1

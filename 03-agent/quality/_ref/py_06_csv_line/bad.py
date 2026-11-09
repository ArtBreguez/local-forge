def parse_csv_line(line):
    if not isinstance(line, str):
        raise TypeError("line must be a str")
    return line.split(",")

def flatten(obj, sep="."):
    if not isinstance(obj, dict):
        raise TypeError("obj must be a dict")
    out = {}

    def walk(value, prefix):
        if isinstance(value, dict) and value:
            for k, v in value.items():
                walk(v, prefix + [str(k)])
        elif isinstance(value, list) and value:
            for i, v in enumerate(value):
                walk(v, prefix + [str(i)])
        else:
            out[sep.join(prefix)] = value

    for k, v in obj.items():
        walk(v, [str(k)])
    return out

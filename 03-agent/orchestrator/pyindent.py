"""Deterministic repair of Python indentation (phase 4, arm T).

Phase 3 found that most failing Python answers were not wrong, only badly laid out: the structure was there but every nested
line sat at the same one-space (or zero) indent, so the file did not compile. `repair(code)` rebuilds the nesting from the
block openers (a line that ends with `:`) and a few rules about where a block ends. It is a heuristic, not a parser:

  * code that already compiles is returned untouched (`changed` is False);
  * otherwise the structure is rebuilt line by line and the result is returned only if it compiles;
  * it never invents or removes a statement: only the leading whitespace of lines changes.

The rules (a "logical line" is a statement start; lines inside brackets or a triple-quoted string are carried along):
  1. the line after a block opener is one level deeper;
  2. `elif`, `else`, `except`, `finally` go back to the level of the construct they continue;
  3. `def`, `class` and decorators go to the level of the nearest enclosing `class` + 1, or to 0;
  4. after `return`, `raise`, `break` or `continue` the next statement leaves the block (one level out);
  5. a `return` that is the direct child of a `for`/`while` loop, or the last statement of the function, leaves the loop
     (the usual shape is "loop, then return the result");
  6. any other line stays at the level of the line before it.
"""
from __future__ import annotations

import re

OPENERS = ("if", "elif", "else", "for", "while", "try", "except", "finally", "with", "def", "class", "async", "match", "case")
CONTINUERS = ("elif", "else", "except", "finally")
TERMINAL = ("return", "raise", "break", "continue")
UNIT = 4


def compiles(code: str) -> bool:
    try:
        compile(code, "<repair>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def _first_word(text: str) -> str:
    m = re.match(r"[A-Za-z_]+", text)
    return m.group(0) if m else ""


def _logical_lines(lines: list[str]) -> list[dict]:
    """Group physical lines into logical ones: a line opened bracket or triple quote swallows the lines that follow it."""
    out: list[dict] = []
    depth = 0
    triple: str | None = None
    for raw in lines:
        if depth > 0 or triple:                       # a continuation line: keep it with its statement, text unchanged
            out[-1]["cont"].append(raw)
        else:
            out.append({"raw": raw, "text": raw.strip(), "cont": []})
        i, text = 0, raw
        while i < len(text):
            if triple:
                if text.startswith(triple, i):
                    triple = None
                    i += 3
                    continue
                i += 1
                continue
            if text.startswith(('"""', "'''"), i):
                triple = text[i:i + 3]
                i += 3
                continue
            ch = text[i]
            if ch in "\"'":
                j = i + 1
                while j < len(text) and text[j] != ch:
                    j += 2 if text[j] == "\\" else 1
                i = j + 1
                continue
            if ch == "#":
                break
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            i += 1
    return out


def _opens_block(text: str) -> bool:
    body = text.split("#", 1)[0].rstrip() if "#" in text and not re.search(r"['\"].*#.*['\"]", text) else text.rstrip()
    return body.endswith(":") and _first_word(text) in OPENERS + ("lambda",)[:0]


def rebuild(code: str) -> str:
    """Re-indent `code` from its block structure (see the module docstring). Raises nothing; the caller checks that it compiles."""
    lines = code.replace("\t", " " * UNIT).split("\n")
    logical = _logical_lines(lines)
    stack: list[tuple[int, str]] = []                  # (level of the opener, its keyword): the blocks we are inside
    prev_level, prev_text, prev_opens = 0, "", False
    out: list[str] = []
    pending_comments: list[str] = []
    levels: list[int | None] = []
    for idx, item in enumerate(logical):
        text = item["text"]
        if not text or text.startswith("#"):
            levels.append(None)
            continue
        word = _first_word(text)
        orig = len(item["raw"]) - len(item["raw"].lstrip(" "))
        if prev_opens:
            level = prev_level + 1
            if word in ("def", "class") and prev_text.startswith(("def ", "async def ")) is False and orig == 0 and prev_level == 0 \
                    and not prev_text.startswith("class "):
                level = 0
        elif word in CONTINUERS:
            kinds = {"elif": ("if", "elif"), "else": ("if", "elif", "for", "while", "try", "except"),
                     "except": ("try", "except"), "finally": ("try", "except", "else")}[word]
            level = 0
            for lvl, kw in reversed(stack):
                if kw in kinds and lvl <= prev_level:
                    level = lvl
                    break
        elif word in ("def", "class", "async") or text.startswith("@"):
            cls_levels = [lvl for lvl, kw in stack if kw == "class" and lvl < prev_level + 1]
            level = cls_levels[-1] + 1 if cls_levels and orig > 0 else 0
        elif orig == 0 and not stack:
            level = 0
        elif orig == 0 and word not in TERMINAL and not prev_text.startswith(TERMINAL):
            level = 0 if not stack or stack[0][1] in ("def", "class") and orig == 0 and prev_level == 0 else prev_level
        elif _first_word(prev_text) in TERMINAL:
            level = max(0, prev_level - 1)
        else:
            level = prev_level
        # rule 5: a `return` directly inside a loop leaves the loop; the last statement of a function sits in the function body
        if word == "return" and not prev_opens and stack:
            kw_top = stack[-1][1]
            nxt = next((x for x in logical[idx + 1:] if x["text"] and not x["text"].startswith("#")), None)
            last = nxt is None or _first_word(nxt["text"]) in ("def", "class") or nxt["text"].startswith("@") \
                or (len(nxt["raw"]) - len(nxt["raw"].lstrip(" "))) == 0
            fn = [lvl for lvl, kw in stack if kw == "def"]
            if kw_top in ("for", "while") and level > stack[-1][0]:
                level = stack[-1][0]
            elif last and fn and kw_top not in ("def",) and level > fn[-1] + 1:
                level = fn[-1] + 1
        # drop blocks we have left, then record this line as an opener if it is one
        while stack and stack[-1][0] >= level and not (word in CONTINUERS and stack[-1][0] == level and False):
            stack.pop()
        opens = _opens_block(text)
        if opens:
            kw = word if word != "async" else _first_word(text[5:])
            stack.append((level, kw))
        levels.append(level)
        prev_level, prev_text, prev_opens = level, text, opens
    # emit; comments and blank lines take the level of the next statement (or stay as they are for blank lines)
    next_levels: list[int] = [0] * len(logical)
    nxt = 0
    for i in range(len(logical) - 1, -1, -1):
        if levels[i] is not None:
            nxt = levels[i]
        next_levels[i] = nxt
    for i, item in enumerate(logical):
        if levels[i] is None:
            out.append(" " * (UNIT * next_levels[i]) + item["text"] if item["text"] else "")
        else:
            out.append(" " * (UNIT * levels[i]) + item["text"])
        out.extend(item["cont"])
    return "\n".join(out)


def repair(code: str) -> tuple[str, bool, bool]:
    """(code, changed, repaired): `changed` tells that the text differs from the input; `repaired` that the input did NOT compile
    and the result does. A failed repair returns the input and (False, False)."""
    if compiles(code):
        return code, False, False
    tabs = code.replace("\t", " " * UNIT)
    if tabs != code and compiles(tabs):
        return tabs, True, True
    try:
        fixed = rebuild(code)
    except Exception:                                  # noqa: BLE001 - a heuristic must never break the run
        return code, False, False
    if compiles(fixed):
        return fixed, True, True
    return code, False, False

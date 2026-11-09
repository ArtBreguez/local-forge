Write a function `top_words(text, k)` that returns the `k` most frequent words in a text.

A word is a maximal run of ASCII letters that may contain single apostrophes between letters, i.e. it matches the pattern `[A-Za-z]+(?:'[A-Za-z]+)*`; everything else (digits, punctuation, other whitespace, a leading or trailing apostrophe) only separates words. Words are compared case-insensitively and reported in lower case, so `"Don't"` and `"don't"` are the same word and `"'tis"` is the word `"tis"`.

Return a list of `(word, count)` tuples for the `k` most frequent words, ordered by count descending and, for equal counts, by word ascending (plain string order). If there are fewer than `k` distinct words, return all of them. `k == 0` gives `[]`.

Raise `ValueError` if `k` is negative and `TypeError` if `k` is not an `int` or `text` is not a `str`.

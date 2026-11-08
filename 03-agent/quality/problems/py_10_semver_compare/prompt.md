Write a function `compare_versions(a, b)` that compares two Semantic Versioning 2.0.0 strings and returns `-1` if `a` has lower precedence than `b`, `0` if they have equal precedence and `1` if `a` has higher precedence.

A version is `MAJOR.MINOR.PATCH`, optionally followed by `-PRERELEASE` and/or `+BUILD`. MAJOR, MINOR and PATCH are non-negative integers without leading zeros. PRERELEASE is one or more dot-separated identifiers made of `[0-9A-Za-z-]`; an identifier that is entirely digits must not have leading zeros, and no identifier may be empty. BUILD is one or more dot-separated non-empty identifiers of `[0-9A-Za-z-]`.

Precedence: compare MAJOR, MINOR, PATCH numerically in that order. A version with a PRERELEASE has lower precedence than the same version without one. Two prerelease parts are compared identifier by identifier from the left: identifiers that are all digits compare numerically; other identifiers compare lexically in ASCII order; an all-digit identifier has lower precedence than an alphanumeric one; if all the shared identifiers are equal, the prerelease with more identifiers has higher precedence. BUILD metadata is ignored completely.

Raise `ValueError` if either string is not a valid version (including a leading `v`, missing or extra numeric parts, or an empty PRERELEASE after `-`).

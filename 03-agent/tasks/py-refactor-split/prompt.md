Refactor `util.py`, which has grown into two unrelated groups of functions, without changing behaviour:

- Create `parsing.py` holding `DEFAULT_SEP`, `UNITS`, `parse_kv`, `parse_kv_block` and `parse_size`.
- Create `formatting.py` holding `format_size`, `format_table` and `format_kv`. It must import what it needs (`UNITS`, `DEFAULT_SEP`) from `parsing`, not copy it.
- `util.py` keeps working for existing callers: it contains no definitions of its own any more, only imports that re-export all of those names, so that `util.parse_kv is parsing.parse_kv`.

The visible tests in `tests/test_util.py` pass now and must still pass (run them with `python -m unittest`); more tests that you cannot see check the new modules. Do not modify anything under `tests/`.

| Arm | thinking | n | pass | rate | Wilson 95% CI | only A / only arm | McNemar p | vs A | same text as A | median tok/s | median tokens | cap hits | thinking budget used up | server errors left out | failures (kind: count) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | off | 40 | 9 | 22% | 12%–38% | — | — | baseline | — | 8.61 | 248 | 3 | 0 | 0 | syntax: 4, test_fail: 26, timeout: 1 |
| A2 | off | 25 | 4 | 16% | 6%–35% | 0 / 0 | 1.000 | within noise; only 25 common problems | 25/25 | 9.72 | 266 | 3 | 0 | 15 | syntax: 3, test_fail: 17, timeout: 1 |
| B | on | 37 | 7 | 19% | 9%–34% | 4 / 3 | 1.000 | within noise; only 37 common problems | 0/37 | 9.69 | 1357 | 6 | 27 | 3 | syntax: 19, test_fail: 11 |
| C | off | 40 | 10 | 25% | 14%–40% | 1 / 2 | 1.000 | within noise | 13/40 | 9.90 | 246 | 2 | 0 | 0 | syntax: 5, test_fail: 24, timeout: 1 |
| D | off | 40 | 10 | 25% | 14%–40% | 0 / 1 | 1.000 | within noise | 21/40 | 8.17 | 244 | 1 | 0 | 0 | syntax: 2, test_fail: 27, timeout: 1 |
| E | off | 40 | 9 | 22% | 12%–38% | 0 / 0 | 1.000 | within noise | 37/40 | 9.71 | 248 | 3 | 0 | 0 | syntax: 4, test_fail: 26, timeout: 1 |

Graded view: the share of each problem's hidden tests that passed (a syntax error, a timeout or no code counts as 0), averaged over the problems.
The sign test looks only at problems where the share differs from arm A.

| Arm | mean share of tests passed | problems where A scored higher / arm scored higher | sign test p | vs A |
|---|---|---|---|---|
| A | 62.2% | — | — | baseline |
| A2 | 59.4% | 0 / 0 | 1.000 | within noise |
| B | 35.9% | 21 / 7 | 0.013 | worse (p<0.05) |
| C | 63.7% | 5 / 9 | 0.424 | within noise |
| D | 66.7% | 1 / 7 | 0.070 | within noise |
| E | 62.2% | 0 / 0 | 1.000 | within noise |

Pass rate by language and by difficulty (k/n):

| Arm | py | js | easy | medium | hard | subtle |
|---|---|---|---|---|---|---|
| A | 6/20 | 3/20 | 3/9 | 4/21 | 2/10 | 2/15 |
| A2 | 1/5 | 3/20 | 2/7 | 2/12 | 0/6 | 1/12 |
| B | 2/20 | 5/17 | 2/7 | 3/20 | 2/10 | 1/15 |
| C | 6/20 | 4/20 | 3/9 | 5/21 | 2/10 | 2/15 |
| D | 7/20 | 3/20 | 3/9 | 5/21 | 2/10 | 2/15 |
| E | 6/20 | 3/20 | 3/9 | 4/21 | 2/10 | 2/15 |

Per problem (P pass, . fail, C = fail with the token cap hit, x = server error, left out of the statistics):

| problem | A | A2 | B | C | D | E |
|---|---|---|---|---|---|---|
| js_01_camel_to_snake | C | C | P | C | . | C |
| js_02_deep_equal | . | . | x | . | . | . |
| js_03_chunk | . | . | x | . | . | . |
| js_04_group_by | P | P | x | P | P | P |
| js_05_parse_query | . | . | P | . | . | . |
| js_06_format_bytes | . | . | . | . | . | . |
| js_07_add_days | . | . | . | . | . | . |
| js_08_age_on | . | . | . | . | . | . |
| js_09_luhn | . | . | . | . | . | . |
| js_10_flatten | . | . | . | . | . | . |
| js_11_memoize | P | P | P | P | P | P |
| js_12_retry | . | . | . | . | . | . |
| js_13_render | . | . | C | P | . | . |
| js_14_emitter | . | . | . | . | . | . |
| js_15_lower_bound | P | P | P | P | P | P |
| js_16_common_prefix | . | . | . | . | . | . |
| js_17_topo_sort | . | . | . | . | . | . |
| js_18_expand_ranges | . | . | . | . | . | . |
| js_19_format_table | . | . | P | . | . | . |
| js_20_token_bucket | . | . | . | . | . | . |
| py_01_parse_duration | . | . | C | . | . | . |
| py_02_roman | C | C | . | . | . | C |
| py_03_iso_week | C | C | . | C | C | C |
| py_04_add_months | . | . | C | . | . | . |
| py_05_business_days | P | P | . | P | P | P |
| py_06_csv_line | . | x | C | . | . | . |
| py_07_balanced | . | x | . | . | . | . |
| py_08_merge_intervals | . | x | . | . | . | . |
| py_09_lru_cache | . | x | . | . | . | . |
| py_10_semver_compare | . | x | C | . | . | . |
| py_11_wrap_text | . | x | . | . | . | . |
| py_12_validate_username | P | x | . | P | P | P |
| py_13_flatten_dict | . | x | . | . | . | . |
| py_14_run_length | P | x | . | P | P | P |
| py_15_rotated_search | P | x | P | P | P | P |
| py_16_top_words | P | x | P | P | P | P |
| py_17_allocate_cents | P | x | . | . | P | P |
| py_18_spiral | . | x | . | . | . | . |
| py_19_time_12h | . | x | C | . | . | . |
| py_20_maze_steps | . | x | . | P | P | . |



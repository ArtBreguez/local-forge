Write a function `camelToSnake(str)` that converts a camelCase or PascalCase identifier to snake_case.

Insert an underscore (1) between a lower-case letter or a digit and a following upper-case letter, and (2) between an upper-case letter and a following upper-case letter that is itself followed by a lower-case letter (the end of an acronym). Then lower-case everything. Characters other than letters and digits are left alone, and a string without upper-case letters is returned unchanged.

Examples: `"parseHTTPResponse"` -> `"parse_http_response"`, `"userID"` -> `"user_id"`, `"XMLHttpRequest"` -> `"xml_http_request"`, `"getV2Value"` -> `"get_v2_value"`, `"HTTP2Server"` -> `"http2_server"`, `"already_snake"` -> `"already_snake"`.

Throw a `TypeError` if the argument is not a string.

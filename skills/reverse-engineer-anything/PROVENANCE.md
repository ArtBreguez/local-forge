# Vendored skill: reverse-engineer-anything (REA)

- Source: https://github.com/morluto/rea, `.agents/skills/reverse-engineer-anything/` at commit
  `25d3c7237ffeffc4f9e6ab87be2f83a643cf5cbb` (2026-11-09), skill metadata version 61.
- Tool it drives: npm `rea-agents` 6.8.0, integrity
  `sha512-9QDeRnGqEDzABZyr04eLnBXBv/c6I9/TWu/6M0MD4pEfkySI/BIgwTODIbAReX3C1x6hjw9+Jlx4A2yUSPZ9ww==`.
- License: MIT, copyright notice in `LICENSE-REA` (copied unchanged).
- Copied unchanged so the offline agent has the instructions without the internet. Update by re-copying from a
  newer commit and re-running `tools/rea-tests.ps1`.

Why it is here (docs/VISION.md): with no internet and no vendor documentation, the agent still needs to understand
software it did not write: a firmware image, a binary whose source is gone, an app's file format, a model file.
REA inspects those locally and returns evidence with stated limitations, which fits the rule that answers must be
checked by tools, not claimed by the model.

What was installed on the host and how: `tools/rea-install.ps1` (portable Node 24.22.2 with the official SHA-256
checked, `npm install --ignore-scripts --omit=optional rea-agents@6.8.0` into `C:\Users\arthu\llm\rea\app`, no global
install, `rea setup` NOT run, no agent configuration touched). Audit notes and test results: docs/JOURNAL.md.

# Vision: the end-of-the-world agent

Goal: an agent that keeps working on this PC with **no internet at all**, for as long as the power is on, and can help with anything a person would otherwise look up or ask: code, repair, medicine, food, water, power, maps, language, maths.

The model is small (9B, half on the CPU, ~9 tok/s) and cannot be trusted to *know* things. So the design moves knowledge and correctness out of the model:

- **Knowledge lives on disk, not in the weights.** The agent answers from retrieved text and cites it. No source found means it says so.
- **Procedures live in skills, not in the model's judgement.** A skill is a short markdown checklist plus the few tools it needs (the Hermes pattern). Small models follow explicit steps far better than they plan.
- **Correctness is checked by the harness, not claimed by the model.** Code is run against tests, maths through a calculator or Python, unit conversions through a converter, answers against the retrieved article.
- **Narrow tool sets per step.** A router picks one skill; the skill exposes 2–5 tools. Fewer choices is the main lever for a small model's tool-calling accuracy.

## Layers

| layer | offline component | why |
|---|---|---|
| inference | llama.cpp `llama-server` (single binary, Vulkan + CPU), main model + small fast model + embedding model | no Python stack, no service that phones home |
| knowledge | Kiwix ZIM archives served by `kiwix-serve` (Wikipedia, WikiMed, Wikibooks, Wikivoyage, iFixit, Stack Exchange sites, DevDocs, Gutenberg, Appropedia), plus public-domain field manuals | the bulk of human reference text in a few compressed files with built-in full-text search |
| retrieval | full-text search from kiwix + a local vector index (embedding model over chunks of the most used archives) | a small model needs the right paragraph in the prompt, not the whole article |
| maps | OpenStreetMap extract for Brazil + offline routing | getting somewhere without Google Maps |
| code | offline package mirrors (pip wheelhouse, npm cache), language docs, local git | writing and running software without PyPI/npm |
| speech and vision | whisper.cpp (speech to text), Piper (text to speech), Tesseract (OCR) | hands-free use, reading paper documents and labels |
| translation | the model itself, checked against an offline dictionary; Argos Translate as a fallback | |
| skills | `skills/<name>/SKILL.md`: medicine triage, water purification, electrical repair, food preservation, coding with tests, research-with-citations, unit and dose calculation, map routing... | the procedures |

## How it is measured

The coding suite in `PROTOCOL.md` stays. A second suite is added: questions with known answers drawn from the offline archives (medicine, repair, agriculture, physics, Portuguese and English), graded on correctness **and** on whether the cited passage actually supports the answer. The same success thresholds apply: a configuration ships only if it is right and grounded on the suite, not because it sounds confident.

## Constraints

- Disk: the host's C: has ~2 GB free. The knowledge layer needs tens of GB (English Wikipedia without images alone is ~50 GB), so it needs a dedicated drive.
- Power: a blackout of the internet may come with one of the grid. The CPU-only profile (no GPU, smaller model) is measured as a fallback for a laptop or a UPS.

"""The distillation pipeline: sample a teacher on a subject's tasks, keep the replies whose final answer code verifies, filter and de-duplicate their reasoning, write SFT records, train a student.

`distill/` (this directory's modules, stdlib only) is the part that does not know the subject; a subject is a `Domain` (`domain.py`) in `distill/domains/<name>/`; the code that needs Modal, Kaggle
or the model libraries is in `distill/backends/`. See `distill/README.md`.
"""

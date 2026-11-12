"""WRAPPER: `modal run tools/modal/modal_lora.py ...` is `distill/backends/modal_train.py` (the LoRA fine-tune, the merge and the GGUF of base and trained model; the student is `--student`, default the model REA's
LoRA is trained on). The file moved there with its behaviour; this one keeps every command of the READMEs working and keeps the names (`run_tag`, `train`, `main`, ...) that were importable from here.
"""
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2] if len(pathlib.Path(__file__).resolve().parents) > 2 else pathlib.Path(__file__).resolve().parent     # inside the Modal container the file sits at /root
if (_ROOT / "distill").is_dir() and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from distill.backends.modal_train import *  # noqa: E402,F401,F403  (`app`, `train`, `to_gguf`, `main` are the ones `modal run` uses: they are registered on the app of that module)

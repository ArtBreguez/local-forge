"""REA task generator: families of small C programs with automatic ground truth. See cli.py for the commands."""
import pathlib
import sys

# The generic half of the pipeline (`distill/`, at the repository root) is imported by this package; a checkout has it two directories above tools/rea-bench, the Modal containers next to it.
_ROOT = pathlib.Path(__file__).resolve().parents[3] if len(pathlib.Path(__file__).resolve().parents) > 3 else None
if _ROOT is not None and (_ROOT / "distill").is_dir() and str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

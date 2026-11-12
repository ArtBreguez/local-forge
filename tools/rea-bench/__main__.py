"""`python3 tools/rea-bench <command>`: runs the rea_bench package that sits next to this file."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rea_bench.cli import main  # noqa: E402

main()

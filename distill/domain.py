"""What a subject has to provide so that the rest of the pipeline can distil it. Stdlib only.

A subject (reverse engineering, arithmetic, ...) is one `Domain` object. The pipeline asks it for tasks, for the messages a teacher is asked, for a yes/no verdict on a final answer and
for what is true about a task; everything else (sampling, trace filters, agreement, records, training) is the same for every subject and lives in the rest of `distill/`.

What each member must guarantee is in `distill/README.md` ("How to add a subject"); the short version:

  verify             computed by CODE from the task's own ground truth, never by a model. It sees only the final answer, whatever the teacher wrote before it.
  heldout_ids        the ids of the tasks a student is evaluated on; they never enter a training set.
  truth_target       the answer text rendered from the ground truth alone (not from a reply), or None when the subject has none.
  needs_agreement    True for a task a lucky guess solves often: it is kept only when `records.AGREE_MIN` samples are right.

The optional members have a default that is right for a subject that does not need them. `contradicts` and `canon_word` exist because the REA filters read the answer's numbers, `task_words` because its task records
are not always complete when a filter runs; `check_task`, `kinds` and `tasks_prefix` because the REA run fails fast on a task it cannot score, validates `--quota` against its families and names its tasks files after its decompile stage.
"""
from __future__ import annotations

import abc
import re
from typing import Container, Iterable

MODES = ("plain", "reasoning", "thinking")
# plain      the teacher answers directly; the reply itself is what `verify` sees, and what is stored is the domain's `truth_target`, never the teacher's text
# reasoning  the teacher writes its steps and a last `Answer: <value>` line; `verify` sees that value; the steps are stored as the teacher wrote them (beside the truth record)
# thinking   as `reasoning`, with the model's native reasoning (`<think>...</think>`) kept as the trace


def mode_of(reasoning: bool, thinking: bool) -> str:
    """The mode of the `--reasoning` / `--thinking` flags (a thinking run is a reasoning run)."""
    return "thinking" if thinking else "reasoning" if reasoning else "plain"


class Domain(abc.ABC):
    name: str = ""
    tasks_prefix: str = ""          # the leading part of a tasks file's name that is not the run's tag (`decompiled-` for REA): default output names are made from the rest

    @abc.abstractmethod
    def load_tasks(self, source) -> list:
        """The tasks of `source` (a path to a file or directory the subject's own task stage wrote, a list of such paths, or a seed for a subject that generates them): a list of dicts, each with a unique `id` and a `kind`
        (the family the quota counts by), in a stable order. A task keeps its ground truth next to the question: `verify` and `truth_target` read it."""

    @abc.abstractmethod
    def messages(self, task: dict, mode: str) -> list:
        """The chat messages a teacher is asked for `task` in `mode` (one of MODES): a system message and a user message. The same messages are stored in the records. They must not contain the
        ground truth."""

    @abc.abstractmethod
    def verify(self, final_answer: str, task: dict) -> bool:
        """True when `final_answer` is the right answer to `task`. Computed by code from the task's ground truth, never by a model. In `plain` mode the argument is the whole reply, otherwise the
        text after the last `Answer:` line. It must be False for anything it cannot parse, and must not raise on a reply (a task it cannot score raises in `check_task`, before anything is asked)."""

    @abc.abstractmethod
    def needs_agreement(self, task: dict) -> bool:
        """True when a right sample is weak evidence about `task` (a guess is right with a probability that is not small), so that the task is kept only when two samples are right."""

    @abc.abstractmethod
    def heldout_ids(self) -> Container:
        """The held-out task ids: anything that supports `id in container`. A task whose id is in it is never sampled or scored into training records."""

    def truth_target(self, task: dict):
        """The answer text of `task` rendered from its ground truth (the record of a `plain` run, and the truth record next to the teacher's traces), or None when the subject has none. `verify`
        must accept it."""
        return None

    # ---- optional hooks ----

    def check_task(self, task: dict) -> None:
        """Raise before a GPU is rented when `task` cannot be used: the default is that its `truth_target`, when there is one, passes `verify`."""
        text = self.truth_target(task)
        if text is not None and not self.verify(text, task):
            raise ValueError(f"the truth target of task {task.get('id', '?')} is rejected by verify: {text!r}")

    def contradicts(self, analysis: str, answer: str, task: dict) -> bool:
        """True when the analysis in front of the `Answer:` line concludes a value other than `answer` and does not correct itself (a trace filter). Default: never."""
        return False

    def task_words(self, task: dict) -> set:
        """The words of the task's prompt: a word with a non-ASCII letter that a reply copies from it is not language mixing (the `mixed_language` filter). Default: the words of its messages."""
        return set(re.findall(r"\w+", " ".join(m["content"] for m in self.messages(task, "reasoning"))))

    def canon_word(self, word: str) -> str:
        """The form of a word in which two spellings of one value are one word (the `unreadable` filter compares the analysis with the answer). Default: `0x20` and `32` are one word."""
        m = re.fullmatch(r"0x[0-9a-f]+|[0-9]+", word)
        return str(int(word, 16 if word.startswith("0x") else 10)) if m and len(word) <= 42 else word

    def kinds(self) -> Iterable | None:
        """Every `kind` a task can have, to validate `--quota`; None: not checked."""
        return None

"""Round 13: the thinking path, ready end to end. Nothing here rents anything, downloads anything or needs a GPU; Modal, the model libraries and the sampler's server are stubbed.

  Rendering        a thinking record is trained as the base model's chat template renders it: prompt ending in `<think>\\n`, target `{reasoning}\\n</think>\\n\\n{answer}` + EOS, the tokens of the whole conversation;
                   a plain record keeps phase A's path byte for byte, and in a mixed run it equals what the template renders for an answer with no reasoning. Run on the REAL template
                   (`.base_chat_template.jinja`, saved from HiddenTerps/LocalForge-9B-REA-heretic, not committed: skipped where the file is absent) and on a verbatim slice of it kept below
                   (always, when jinja2 is installed), with a stub tokenizer that splits like a BPE pre-tokenizer.
  Parsing          which assistant texts are thinking, plain or malformed (the template's own split, and nothing looser)
  Training         `modal_lora.train` with the model libraries faked: which records are trained, skipped, dropped; thinking only by default, `--mix` on request; the output names
  Sampling         Qwen3's thinking settings for `--thinking`, never greedy, overrides, the stats; the non-thinking defaults unchanged
  Length           `--max-tokens` 4096 by default for thinking, the slot holds prompt + max_tokens, prompts that do not fit are not asked, the length report counts 4096 and 8192
  Docs             the README section and the modal_lora docstring say what the code does
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import pathlib
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

from .helpers import ROOT
from .rea_tasks import meta as real_task, right
from .test_rea_modal import ModalCase, StubbedSampleStage
from .test_rea_reasoning import STEPS, answer_body, reply
from .test_rea_traces import think_body

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import outputs  # noqa: E402
from distill.backends import modal_sample  # noqa: E402
from .rea_bound import rsn, sft  # noqa: E402

try:
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment
except ImportError:         # the rendering tests are skipped, with this reason
    jinja2 = None

TEMPLATE_PATH = ROOT / ".base_chat_template.jinja"
TEMPLATE_SHA256 = "f7e41c826a3eea86dbd6ed2e204f0fc03262fcf720144e1dbb9b6d3e44e333a1"     # the file the facts below were read from

# The lines of that template that decide the assistant turn and the generation prompt, verbatim (the system-message merge, the last-query index, the assistant branch, `add_generation_prompt`),
# with the macro that prints a string content and without the tool-call branches (no record has one). `RealTemplateIsPinned` checks that every line is in the real file.
SLICE = r"""
{%- macro render_content(content, do_vision_count, is_system_content=false) %}{{- content }}{%- endmacro %}
{%- set num_sys = 0 %}
{%- set merged_system = '' %}
{%- if messages and (messages[0].role == 'system' or messages[0].role == 'developer') %}
    {%- set first = render_content(messages[0].content, false, true)|trim %}
    {%- if messages|length > 1 and (messages[1].role == 'system' or messages[1].role == 'developer') %}
        {%- set second = render_content(messages[1].content, false, true)|trim %}
        {%- set merged_system = first + '\n' + second %}
        {%- set num_sys = 2 %}
    {%- else %}
        {%- set merged_system = first %}
        {%- set num_sys = 1 %}
    {%- endif %}
{%- endif %}
{%- if merged_system %}
        {{- '<|im_start|>system\n' + merged_system + '<|im_end|>\n' }}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if loop.index0 >= num_sys and message.role != "system" and message.role != "developer" %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
"""
SLICE_ONLY = ("{%- macro render_content", "{%- if merged_system %}", "        {{- '<|im_start|>system\\n' + merged_system")       # lines of the slice that are not lines of the real file


def render(template: str, messages: list, **kw) -> str:
    """What `tokenizer.apply_chat_template(messages, tokenize=False, **kw)` returns: the template rendered the way transformers renders it (sandboxed, trim_blocks, lstrip_blocks, loopcontrols)."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)
    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(messages=messages, **kw)


PRETOKEN = re.compile(r"<\|im_start\|>|<\|im_end\|>|</?think>| ?\w+|\s*\n+|[^\w\s]+|\s+")


class StubTok:
    """The two methods `modal_lora` uses of a tokenizer, over a template: `apply_chat_template` renders it, a call splits the text the way a BPE pre-tokenizer does (special tokens whole, a word with the space
    in front of it, a run of newlines) and gives each piece one id. No merge ever crosses a piece, which is what makes a prompt that ends in a newline a prefix of the whole text."""
    eos_token = "<|im_end|>"

    def __init__(self, template: str):
        self.template, self.vocab = template, {}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):
        assert tokenize is False
        return render(self.template, msgs, add_generation_prompt=add_generation_prompt, **kw)

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        pieces = PRETOKEN.findall(text)
        assert "".join(pieces) == text
        return {"input_ids": [self.vocab.setdefault(p, len(self.vocab)) for p in pieces]}

    def decode(self, ids):
        back = {i: p for p, i in self.vocab.items()}
        return "".join(back[i] for i in ids)

    def save_pretrained(self, path):
        pass


REASONING = "First I read the table.\nIt has 8 entries.\nSo the loop bound is 8."
ANSWER = "The loop runs over the eight entries of the table.\nAnswer: 8"
THINK = f"<think>\n{REASONING}\n</think>\n\n{ANSWER}"
USER = "```c\nint f(void){return 8;}\n```\n\nQuestion: how many entries?"


def conv(assistant: str = THINK) -> list:
    return [{"role": "system", "content": rsn.SYSTEM_REASONING}, {"role": "user", "content": USER}, {"role": "assistant", "content": assistant}]


def lora_module():
    sys.modules.pop("distill.backends.modal_train", None)
    import distill.backends.modal_train as modal_lora
    return modal_lora


class RealTemplateIsPinned(unittest.TestCase):
    """The slice above is the real template's text, and the real template is the file the facts were read from."""

    def setUp(self):
        if not TEMPLATE_PATH.exists():
            self.skipTest(f"{TEMPLATE_PATH.name} is not here: copy the chat template of HiddenTerps/LocalForge-9B-REA-heretic to {TEMPLATE_PATH} (it is not committed)")

    def test_the_saved_template_is_the_one_the_facts_were_read_from(self):
        self.assertEqual(hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest(), TEMPLATE_SHA256, "the template changed: read the assistant branch and the generation prompt again, then update the pin")

    def test_every_line_of_the_slice_is_a_line_of_the_real_template(self):
        real = set(TEMPLATE_PATH.read_text().split("\n"))
        missing = [l for l in SLICE.strip("\n").split("\n") if l not in real and not l.startswith(SLICE_ONLY)]
        self.assertEqual(missing, [])

    @unittest.skipIf(jinja2 is None, "jinja2 is not installed")
    def test_the_slice_renders_like_the_real_template(self):
        real = TEMPLATE_PATH.read_text()
        for msgs in (conv(), conv("plain answer"), conv()[:2], [*conv(), {"role": "user", "content": "again"}, {"role": "assistant", "content": THINK}],
                     [{"role": "assistant", "content": THINK}, {"role": "user", "content": "q"}], [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}, *conv()[1:]]):
            for kw in ({}, {"add_generation_prompt": True}, {"add_generation_prompt": True, "enable_thinking": False}, {"add_generation_prompt": True, "enable_thinking": True}):
                self.assertEqual(render(SLICE, msgs, **kw), render(real, msgs, **kw), (msgs, kw))


# ---- rendering: a thinking record is the template's own conversation ---------------------------------------------------------------------------------------

class Rendering:
    """The tests; the classes below run them on a template."""

    def setUp(self):
        if jinja2 is None:
            self.skipTest("jinja2 is not installed: the chat template cannot be rendered")
        self.tok = StubTok(self.template())
        self.lora = self.__class__.lora

    @classmethod
    def setUpClass(cls):
        cls.template_text = cls.load_template()
        super().setUpClass()
        cls.lora = lora_module()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("distill.backends.modal_train", None)
        super().tearDownClass()

    def template(self):
        return self.template_text

    def whole(self, msgs) -> str:
        """The template's rendering of the whole conversation, without the newline it puts after the last end-of-turn token (training stops at the token)."""
        out = self.tok.apply_chat_template(msgs, tokenize=False)
        self.assertTrue(out.endswith("<|im_end|>\n"))
        return out[:-1]

    def test_the_generation_prompts_are_the_ones_the_task_describes(self):
        pre = conv()[:2]
        self.assertTrue(self.tok.apply_chat_template(pre, tokenize=False, add_generation_prompt=True, enable_thinking=True).endswith("<|im_start|>assistant\n<think>\n"))
        self.assertTrue(self.tok.apply_chat_template(pre, tokenize=False, add_generation_prompt=True).endswith("<|im_start|>assistant\n<think>\n"), "not given = thinking")
        self.assertTrue(self.tok.apply_chat_template(pre, tokenize=False, add_generation_prompt=True, enable_thinking=False).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_a_thinking_record_is_the_conversation_as_the_template_renders_it(self):
        prompt, full = self.lora.render_record(self.tok, conv(), "think", check=True)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"), "the prompt is the template's thinking generation prompt")
        self.assertEqual(full, self.whole(conv()), "prompt + reasoning + </think> + answer + EOS is the template's own text")
        self.assertEqual(full, prompt + REASONING + "\n</think>\n\n" + ANSWER + "<|im_end|>")
        self.assertEqual(prompt, self.tok.apply_chat_template(conv()[:2], tokenize=False, add_generation_prompt=True, enable_thinking=True))

    def test_the_tokens_are_those_of_the_whole_conversation_and_the_loss_covers_the_reasoning_and_the_answer(self):
        row, fate = self.lora.encode_record(self.tok, conv(), 8192, True, False)
        self.assertEqual(fate, "think")
        want = self.tok(self.whole(conv()))["input_ids"]
        self.assertEqual(row["input_ids"], want)
        p_ids = self.tok(self.tok.apply_chat_template(conv()[:2], tokenize=False, add_generation_prompt=True, enable_thinking=True))["input_ids"]
        self.assertEqual(want[:len(p_ids)], p_ids, "the prompt's tokens are the start of the record's")
        self.assertEqual(row["labels"][:len(p_ids)], [-100] * len(p_ids))
        trained = row["labels"][len(p_ids):]
        self.assertNotIn(-100, trained)
        self.assertEqual(self.tok.decode(trained), f"{REASONING}\n</think>\n\n{ANSWER}<|im_end|>", "reasoning, the closing tag, the answer and the end-of-turn token are trained, nothing else")
        self.assertEqual(row["attention_mask"], [1] * len(want))

    def test_the_reasoning_content_field_and_the_think_text_are_the_same_conversation(self):
        field = [*conv()[:2], {"role": "assistant", "content": ANSWER, "reasoning_content": REASONING}]
        self.assertEqual(self.whole(field), self.whole(conv()))

    def test_the_template_keeps_the_reasoning_only_after_the_last_user_message_and_a_record_is_always_that(self):
        earlier = [conv()[0], conv()[1], {"role": "assistant", "content": THINK}, {"role": "user", "content": "next"}]
        self.assertNotIn("First I read the table", self.tok.apply_chat_template(earlier, tokenize=False), "an earlier assistant turn loses its reasoning")
        self.assertIn("First I read the table", self.whole(conv()), "the last turn, after the last user message, keeps it")

    def test_a_plain_record_is_phase_as_path_byte_for_byte(self):
        plain = conv("The loop runs over eight entries.\nAnswer: 8")
        old_prompt = self.tok.apply_chat_template(plain[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        old_full = old_prompt + plain[-1]["content"] + self.tok.eos_token
        self.assertEqual(self.lora.render_record(self.tok, plain, "plain"), (old_prompt, old_full))
        old_row = self.lora.encode_ids(self.tok(old_prompt, add_special_tokens=False)["input_ids"], self.tok(old_full, add_special_tokens=False)["input_ids"], 2048)
        self.assertEqual(self.lora.encode_record(self.tok, plain), (old_row, "plain"))
        self.assertEqual(self.lora.encode_record(self.tok, plain, 2048, False, False), (old_row, "plain"), "a file with no thinking record trains exactly as before")
        self.assertEqual(self.lora.encode_record(self.tok, plain, 2048, False, True), (old_row, "plain"), "--mix changes nothing there")

    def test_a_plain_record_of_a_thinking_config_is_what_the_template_renders_for_an_answer_with_no_reasoning(self):
        plain = conv("The loop runs over eight entries.\nAnswer: 8")
        prompt, full = self.lora.render_record(self.tok, plain, "plain", check=True)
        self.assertTrue(prompt.endswith("<think>\n\n</think>\n\n"), "the empty think block")
        self.assertEqual(full, self.whole(plain), "the same text the template gives the whole conversation: consistent with the thinking records")
        row, fate = self.lora.encode_record(self.tok, plain, 8192, True, True)
        self.assertEqual((fate, row["input_ids"]), ("plain", self.tok(self.whole(plain))["input_ids"]))

    def test_a_record_over_the_window_is_dropped_never_cut(self):
        row, fate = self.lora.encode_record(self.tok, conv(), 30, True, False)
        self.assertEqual((row, fate), (None, "think"))
        row, _ = self.lora.encode_record(self.tok, conv(), len(self.tok(self.whole(conv()))["input_ids"]), True, False)
        self.assertIsNotNone(row, "it fits exactly")
        row, _ = self.lora.encode_record(self.tok, conv(), len(row["input_ids"]) - 1, True, False)
        self.assertIsNone(row)

    def test_the_default_run_of_a_thinking_config_skips_the_plain_records_and_malformed_ones_are_dropped(self):
        plain, bad = conv("Answer: 8"), conv("<think>\nunclosed reasoning\nAnswer: 8")
        self.assertEqual(self.lora.encode_record(self.tok, plain, 8192, True, False), (None, "skipped_plain"))
        self.assertEqual(self.lora.encode_record(self.tok, bad, 8192, True, True), (None, "malformed"))
        self.assertEqual(self.lora.encode_record(self.tok, bad, 8192, False, False), (None, "malformed"), "never trained as plain text: it would teach a literal <think>")

    def test_a_tokenizer_whose_template_does_not_end_the_prompt_that_way_is_refused_loudly(self):
        class NoThink(StubTok):
            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):
                return "<|im_start|>assistant\n" if add_generation_prompt else super().apply_chat_template(msgs, tokenize, add_generation_prompt, **kw)
        with self.assertRaisesRegex(ValueError, "ends in"):
            self.lora.encode_record(NoThink(self.template()), conv(), 8192, True, False)
        with self.assertRaisesRegex(ValueError, "ends in"):
            self.lora.encode_record(NoThink(self.template()), conv("Answer: 8"), 8192, True, True)

        class Glue(StubTok):
            def __call__(self, text, add_special_tokens=False):          # a tokenizer whose prompt does not come out as the start of the whole text (a merge across the boundary)
                ids = super().__call__(text)["input_ids"]
                return {"input_ids": ids + [999] if text.endswith("<think>\n") else ids}
        with self.assertRaisesRegex(ValueError, "not the start"):
            self.lora.encode_record(Glue(self.template()), conv(), 8192, True, False)


class OnTheRealTemplate(Rendering, ModalCase):
    @classmethod
    def load_template(cls):
        if not TEMPLATE_PATH.exists():
            raise unittest.SkipTest(f"{TEMPLATE_PATH.name} is not here (not committed): copy the chat template of HiddenTerps/LocalForge-9B-REA-heretic to {TEMPLATE_PATH} to run these on the real template")
        return TEMPLATE_PATH.read_text()


class OnThePinnedSlice(Rendering, ModalCase):
    @classmethod
    def load_template(cls):
        return SLICE


# ---- parsing --------------------------------------------------------------------------------------------------------------------------------------------

class Parsing(ModalCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.lora = lora_module()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("distill.backends.modal_train", None)
        super().tearDownClass()

    def test_a_thinking_text_is_split_the_way_the_template_splits_it(self):
        self.assertEqual(self.lora.parse_assistant(THINK), ("think", (REASONING, ANSWER)))
        self.assertEqual(self.lora.parse_assistant(f"\n  {THINK}\n\n"), ("think", (REASONING, ANSWER)), "the whole text is trimmed first, as the template does")
        self.assertEqual(self.lora.parse_assistant("<think>r</think>a"), ("think", ("r", "a")))
        self.assertEqual(self.lora.parse_assistant("<think>\n\n  r  \n\n</think>\n\n\n  a  "), ("think", ("r", "  a")), "the reasoning is trimmed, the answer loses its leading newlines only")
        self.assertEqual(self.lora.parse_assistant("<think>\nr1\n\nr2\n</think>\n\na1\na2")[1], ("r1\n\nr2", "a1\na2"))

    def test_text_without_a_think_tag_is_plain_whatever_else_it_says(self):
        for text in ("Answer: 8", "", "a < b and b > c", "I think so.\nAnswer: 8", "<thinking>x</thinking>", "x </ think> y"):
            self.assertEqual(self.lora.parse_assistant(text), ("plain", None), text)

    def test_anything_else_with_a_tag_in_it_is_malformed(self):
        for text in ("<think>\nunclosed\nAnswer: 8", "reasoning\n</think>\n\nAnswer: 8", "<think>a</think>b<think>c</think>d", "<think>a</think>b</think>", "<think>a</think>b <think>",
                     "lead-in text <think>\nr\n</think>\n\nAnswer: 8", "<think>\n\n</think>\n\nAnswer: 8", "<think>\n  \n</think>\n\nAnswer: 8", "<think>\nr\n</think>", "<think>\nr\n</think>\n\n  \n",
                     "</think><think>r"):
            self.assertEqual(self.lora.parse_assistant(text)[0], "malformed", text)

    def test_it_reads_what_the_sample_stage_stores(self):
        for think in ("Let me read it.", "Wait, the loop starts at 1.\n\nSo 8 entries."):
            stored = rsn.thinking_reply("Steps.\nAnswer: 8", think)
            self.assertEqual(self.lora.parse_assistant(stored), ("think", (think, "Steps.\nAnswer: 8")))
            self.assertEqual(self.lora.parse_assistant(rsn.trace_text(stored, True))[0], "think")
        self.assertEqual(self.lora.parse_assistant(rsn.thinking_reply("Steps.\nAnswer: 8", ""))[0], "plain", "a reply with no reasoning is a plain record")

    def test_a_dataset_is_a_thinking_config_when_one_record_is_a_thinking_record(self):
        self.assertTrue(self.lora.is_thinking_config(["Answer: 8", THINK]))
        self.assertFalse(self.lora.is_thinking_config(["Answer: 8", "More text"]))
        self.assertFalse(self.lora.is_thinking_config(["<think>unclosed"]), "a malformed record does not make a thinking config")
        self.assertFalse(self.lora.is_thinking_config([]))

    def test_the_fate_of_a_record(self):
        f = self.lora.record_fate
        self.assertEqual([f(THINK, True, False), f(THINK, True, True), f(THINK, False, False)], ["think"] * 3)
        self.assertEqual([f("Answer: 8", True, False), f("Answer: 8", True, True), f("Answer: 8", False, False), f("Answer: 8", False, True)], ["skipped_plain", "plain", "plain", "plain"])
        self.assertEqual([f("<think>x", c, m) for c in (True, False) for m in (True, False)], ["malformed"] * 4)
        self.assertEqual(self.lora.FATES, ("think", "plain", "skipped_plain", "malformed"))

    def test_the_counts(self):
        c = self.lora.fate_counts([("think", True), ("think", True), ("think", False), ("plain", True), ("plain", False), ("skipped_plain", False), ("malformed", False)])
        self.assertEqual(c, {"think": 2, "plain": 1, "skipped_plain": 1, "malformed": 1, "too_long": 2, "too_long_think": 1})


# ---- the training function, with the model libraries faked ----------------------------------------------------------------------------------------

class FakeDS:
    """The few methods of a `datasets.Dataset` that `modal_lora.train` uses."""

    def __init__(self, rows):
        self.rows = rows

    column_names = property(lambda self: list(self.rows[0]) if self.rows else [])
    __len__ = lambda self: len(self.rows)
    __getitem__ = lambda self, key: [r[key] for r in self.rows]
    select = lambda self, idx: FakeDS([self.rows[i] for i in idx])
    shuffle = lambda self, seed=None: FakeDS(list(self.rows))
    map = lambda self, fn, remove_columns=None: FakeDS([fn(dict(r)) for r in self.rows])
    filter = lambda self, fn: FakeDS([r for r in self.rows if fn(r)])
    remove_columns = lambda self, cols: FakeDS([{k: v for k, v in r.items() if k not in cols} for r in self.rows])


def dataset_rows() -> list:
    """What a thinking sample run uploads for two tasks: per task one truth-template record, then its traces; and one malformed record besides (2 plain, 3 thinking, 1 malformed)."""
    def row(msgs, tid, source):
        return {"messages": msgs, "task": tid, "kind": "x", "model": "m", "target_source": source}
    rows = []
    for tid, traces in (("t0", ("Count the entries.\nThere are 8.", "The bound is a constant.\nIt is 8.")), ("t1", ("Read the loop header.\nIt stops at 8.",))):
        rows.append(row(conv(f"The loop bound is 8 for {tid}.\nAnswer: 8"), tid, "truth-template"))
        rows += [row(conv(f"<think>\n{think}\n</think>\n\nSteps.\nAnswer: 8"), tid, "teacher-reasoning") for think in traces]
    rows.append(row(conv("<think>\nnever closed\nAnswer: 8"), "t9", "teacher-reasoning"))
    return rows


THINK_TRACES = ("Count the entries.\nThere are 8.", "The bound is a constant.\nIt is 8.", "Read the loop header.\nIt stops at 8.")


class TrainEndToEnd:
    """`modal_lora.train` up to the Trainer, on a fake dataset: which examples reach the trainer."""

    def setUp(self):
        if jinja2 is None:
            self.skipTest("jinja2 is not installed: the chat template cannot be rendered")
        self.tok = StubTok(self.template_text)
        self.captured = {}
        captured, tok = self.captured, self.tok
        model = mock.MagicMock()

        class Trainer:
            def __init__(self, model, args, train_dataset, data_collator):
                captured["ds"], captured["args"] = train_dataset, args
                self.state = types.SimpleNamespace(log_history=[{"loss": 1.0}])

            def train(self):
                return types.SimpleNamespace(global_step=3)

        def load_dataset(name, config, split):
            captured["loaded"] = (name, config, split)
            return FakeDS(self.rows)
        self.rows = dataset_rows()
        self.mods = {"torch": types.SimpleNamespace(bfloat16="bf16"), "datasets": types.SimpleNamespace(load_dataset=load_dataset),
                     "peft": types.SimpleNamespace(LoraConfig=lambda **k: types.SimpleNamespace(**k), get_peft_model=lambda m, c: m),
                     "transformers": types.SimpleNamespace(AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: model), AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda b: tok),
                                                           Trainer=Trainer, TrainingArguments=lambda **k: types.SimpleNamespace(**k), DataCollatorForSeq2Seq=lambda *a, **k: None)}
        self.lora = lora_module()
        self.addCleanup(lambda: sys.modules.pop("distill.backends.modal_train", None))

    def train(self, **kw):
        with mock.patch.dict(sys.modules, self.mods), contextlib.redirect_stdout(io.StringIO()) as printed:
            r = self.lora.train(**{"max_len": 8192, "config": "rea_thinking", **kw})
        self.printed = printed.getvalue()
        return r

    def trained(self):
        """The decoded trained part (the labels that are not -100) of every example that reached the trainer."""
        out = []
        for row in self.captured["ds"].rows:
            first = next(i for i, x in enumerate(row["labels"]) if x != -100)
            self.assertNotIn(-100, row["labels"][first:])
            out.append(self.tok.decode(row["labels"][first:]))
        return out

    def test_a_thinking_config_trains_the_thinking_records_only_by_default(self):
        r = self.train()
        self.assertEqual(self.captured["loaded"], ("HiddenTerps/local-forge-agent-sft", "rea_thinking", "train"))
        self.assertEqual((r["thinking_config"], r["mix"], r["tag"], r["n"]), (True, False, "rea_thinking", 3))
        self.assertEqual(r["records"], {"think": 3, "plain": 0, "skipped_plain": 2, "malformed": 1, "too_long": 0, "too_long_think": 0})
        self.assertEqual(self.trained(), [f"{x}\n</think>\n\nSteps.\nAnswer: 8<|im_end|>" for x in THINK_TRACES])
        self.assertIn("2 plain records skipped, 1 malformed dropped", self.printed)
        self.assertIn("no --mix", self.printed)
        for row in self.captured["ds"].rows:
            self.assertTrue(self.tok.decode(row["input_ids"]).startswith("<|im_start|>system\n"))
            self.assertIn("<|im_start|>assistant\n<think>\n", self.tok.decode(row["input_ids"]))

    def test_mix_adds_the_truth_template_records_with_an_empty_think_block_and_has_its_own_output(self):
        r = self.train(mix=True)
        self.assertEqual((r["mix"], r["tag"], r["n"]), (True, "rea_thinking-mix", 5))
        self.assertEqual(r["records"], {"think": 3, "plain": 2, "skipped_plain": 0, "malformed": 1, "too_long": 0, "too_long_think": 0})
        self.assertEqual(sum(t.startswith("The loop bound is 8 for") and t.endswith("Answer: 8<|im_end|>") for t in self.trained()), 2, "a plain record trains its text and the end of turn")
        for row in self.captured["ds"].rows:
            self.assertEqual(self.tok.decode(row["input_ids"]).count("<think>"), 1, "one think block per example: the template's empty one, or the record's own")
        plain = next(x for x in self.captured["ds"].rows if "The loop bound is 8 for" in self.tok.decode(x["input_ids"]))
        self.assertIn("<|im_start|>assistant\n<think>\n\n</think>\n\nThe loop bound", self.tok.decode(plain["input_ids"]))
        self.assertIn("--mix", self.printed)
        self.assertEqual(self.train(mix=True, out_tag="mine")["tag"], "mine", "a tag given by hand is used as it is")
        self.rows = dataset_rows() * 20
        self.assertEqual(self.train(smoke=True, mix=True)["tag"], "rea_thinking-mix-smoke")

    def test_a_smoke_run_decides_on_the_whole_file_not_on_its_slice(self):
        plain = dataset_rows()[0]
        self.rows = [plain] * 70 + dataset_rows()
        with self.assertRaisesRegex(SystemExit, "60 skipped plain"):
            self.train(smoke=True)             # the slice is 60 plain records of a thinking file: they are skipped, not trained as if the file were phase A's
        self.rows = dataset_rows() * 20
        r = self.train(smoke=True)
        self.assertEqual((r["thinking_config"], r["records"]["think"], r["records"]["skipped_plain"], r["tag"]), (True, 30, 20, "rea_thinking-smoke"))

    def test_a_file_without_a_thinking_record_trains_as_it_always_did_whatever_mix_says(self):
        self.rows = [r for r in dataset_rows() if r["target_source"] == "truth-template"]
        r = self.train(max_len=2048, config="rea_sft")
        self.assertEqual((r["thinking_config"], r["n"], r["tag"], r["records"]["plain"], r["records"]["skipped_plain"]), (False, 2, "full", 2, 0))
        first = [dict(x) for x in self.captured["ds"].rows]
        again = self.train(max_len=2048, config="rea_sft", mix=True, out_tag="x")
        self.assertEqual((again["n"], again["tag"]), (2, "x"))
        self.assertEqual(self.captured["ds"].rows, first, "the same examples with and without --mix")
        for row in self.captured["ds"].rows:
            self.assertIn("<think>\n\n</think>\n\nThe loop bound", self.tok.decode(row["input_ids"]), "phase A's prompt, with the empty think block")
        self.assertNotIn("thinking config:", self.printed, "phase A's log is its old one")
        with self.assertRaisesRegex(SystemExit, "--mix"):
            self.train(max_len=2048, config="rea_sft", mix=True)           # no tag of its own: it would write /out/full

    def test_a_record_over_the_window_is_dropped_and_counted_never_cut(self):
        plain_len = max(len(x["input_ids"]) for x in (self.train(mix=True), self.captured["ds"])[1].rows if "The loop bound" in self.tok.decode(x["input_ids"]))
        think_len = min(len(x["input_ids"]) for x in self.captured["ds"].rows if "The loop bound" not in self.tok.decode(x["input_ids"]))
        self.assertLess(plain_len, think_len, "the fixture: a thinking record is longer than a plain one")
        r = self.train(max_len=plain_len, mix=True)
        self.assertEqual((r["n"], r["dropped_too_long"], r["records"]["too_long"], r["records"]["too_long_think"], r["records"]["plain"]), (2, 3, 3, 3, 2))
        self.assertIn("3 dropped because prompt + answer is longer than", self.printed)
        self.assertTrue(all(len(x["input_ids"]) <= plain_len for x in self.captured["ds"].rows))

    def test_nothing_to_train_on_is_an_error_and_phase_as_message_is_kept_where_it_applies(self):
        self.rows = [{**dataset_rows()[1], "messages": conv("<think>\nunclosed")}]
        with self.assertRaisesRegex(SystemExit, "nothing to train on: 1 SFT pairs, 0 longer than 8192 tokens, 0 skipped plain, 1 malformed"):
            self.train()
        self.rows = [r for r in dataset_rows() if r["target_source"] == "truth-template"]
        with self.assertRaisesRegex(SystemExit, "every one of the 2 SFT pairs is longer than 5 tokens: nothing to train on"):
            self.train(max_len=5, config="rea_sft")


class TrainOnThePinnedSlice(TrainEndToEnd, ModalCase):
    template_text = SLICE


class TrainOnTheRealTemplate(TrainEndToEnd, ModalCase):
    @classmethod
    def setUpClass(cls):
        if not TEMPLATE_PATH.exists():
            raise unittest.SkipTest(f"{TEMPLATE_PATH.name} is not here (not committed): copy the chat template of HiddenTerps/LocalForge-9B-REA-heretic to {TEMPLATE_PATH}")
        cls.template_text = TEMPLATE_PATH.read_text()
        super().setUpClass()


# ---- modal_lora: the entrypoint, the output names, the batch plan -----------------------------------------------------------------------------------

class TrainingEntry(ModalCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.lora = lora_module()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("distill.backends.modal_train", None)
        super().tearDownClass()

    def go(self, remote=None, gmap=None, **kw):
        remote = remote or mock.MagicMock(return_value={"tag": "x"})
        gmap = gmap or mock.MagicMock(return_value=[])
        with tempfile.TemporaryDirectory() as d, mock.patch.object(self.lora, "ROOT", pathlib.Path(d)), mock.patch.object(self.lora, "train", types.SimpleNamespace(remote=remote)), \
                mock.patch.object(self.lora, "to_gguf", types.SimpleNamespace(map=gmap)), contextlib.redirect_stdout(io.StringIO()):
            self.lora.main(**kw)
            names = sorted(p.name for p in (pathlib.Path(d) / "results" / "phase6" / "lora").glob("*"))
        return remote, gmap, names

    def test_mix_is_a_flag_with_its_own_output_and_the_default_is_thinking_records_only(self):
        for fn in (self.lora.main, self.lora.train):
            self.assertEqual(inspect.signature(fn).parameters["mix"].default, False, fn.__name__)
        remote, gmap, names = self.go(config="rea_thinking", max_len=8192)
        remote.assert_called_once_with(False, 2.0, 1.5e-4, 32, 8192, "rea_thinking", "rea_thinking", False)
        self.assertEqual(names, ["rea_thinking-gguf.json", "rea_thinking-train.json"])
        remote, gmap, names = self.go(config="rea_thinking", max_len=8192, mix=True)
        remote.assert_called_once_with(False, 2.0, 1.5e-4, 32, 8192, "rea_thinking", "rea_thinking-mix", True)
        gmap.assert_called_once_with(["rea_thinking-mix", "rea_thinking-mix"], ["lora", "base"])
        self.assertEqual(names, ["rea_thinking-mix-gguf.json", "rea_thinking-mix-train.json"])
        remote, _, _ = self.go(config="rea_thinking", mix=True, smoke=True)
        self.assertEqual(remote.call_args.args[6], "rea_thinking-mix-smoke")

    def test_mix_on_the_default_config_is_refused_before_anything_is_rented(self):
        remote = mock.MagicMock()
        with self.assertRaisesRegex(SystemExit, "--mix"):
            self.go(remote, mix=True)
        remote.assert_not_called()
        self.assertEqual(self.lora.run_tag(False, "rea_sft", "mine", True), "mine", "with a tag of its own it is allowed (and changes nothing for that file)")
        self.assertEqual(self.lora.run_tag(), "full")
        self.assertEqual(self.lora.run_tag(True), "smoke")

    def test_the_output_names(self):
        rt = self.lora.run_tag
        self.assertEqual((rt(False, "rea_thinking", "", True), rt(True, "rea_thinking", "", True), rt(False, "rea_thinking")), ("rea_thinking-mix", "rea_thinking-mix-smoke", "rea_thinking"))
        self.assertEqual(rt(False, "rea thinking/1", "", True), "rea-thinking-1-mix")
        self.assertNotEqual(rt(False, "rea_thinking", "", True), rt(False, "rea_thinking"), "a mixed run never shares a directory with the thinking-only run")
        self.assertNotIn(rt(False, "rea_thinking", "", True), ("full", "smoke"))

    def test_the_batch_plan_for_the_thinking_window(self):
        bp = self.lora.batch_plan
        self.assertEqual([bp(w) for w in (2048, 3072, 4096, 8192)], [(4, 4), (2, 8), (2, 8), (1, 16)])
        for w in (2048, 3072, 4096, 8192):
            per, accum = bp(w)
            self.assertEqual(per * accum, 16, "16 examples per optimizer step")
            self.assertLessEqual(per * w, 8192, "the tokens of a micro-batch never exceed the 4 x 2048 that is known to fit")
        self.assertIn("8192", inspect.getsource(bp))

    def test_the_not_done_note_is_gone_and_the_docstring_says_what_the_code_does(self):
        doc = self.lora.__doc__
        self.assertNotIn("NOT DONE", doc)
        self.assertNotIn("not trainable here", doc)
        for part in ("enable_thinking=True", "{reasoning}\\n</think>\\n\\n{answer}", "per RECORD", "--mix", "empty think block", "answer directly", "thinking records ONLY", "<config>-mix"):
            self.assertIn(part, doc)


# ---- sampling ---------------------------------------------------------------------------------------------------------------------------------------

class SamplingSettings(unittest.TestCase):
    def test_the_thinking_defaults_are_qwen3s_recommendation(self):
        self.assertEqual(rsn.THINKING_SAMPLING, {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0})
        self.assertEqual(list(rsn.THINKING_SAMPLING), ["temperature", "top_p", "top_k", "min_p"], "the order of the request body")
        self.assertEqual(rsn.sampling_for(True), rsn.THINKING_SAMPLING)
        self.assertIsNot(rsn.sampling_for(True), rsn.THINKING_SAMPLING, "a copy: a caller cannot change the constant")

    def test_a_run_that_does_not_think_keeps_phase_as_one_value(self):
        self.assertEqual(rsn.DEFAULT_SAMPLING, {"temperature": 0.7})
        self.assertEqual(rsn.sampling_for(False), {"temperature": 0.7})
        self.assertEqual(rsn.sampling_for(False, None, None, None, None), {"temperature": 0.7})

    def test_overrides_replace_only_what_they_name(self):
        self.assertEqual(rsn.sampling_for(True, temperature=0.8), {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "min_p": 0.0})
        self.assertEqual(rsn.sampling_for(True, top_p=1.0, top_k=0, min_p=0.05), {"temperature": 0.6, "top_p": 1.0, "top_k": 0, "min_p": 0.05})
        self.assertEqual(rsn.sampling_for(False, top_p=0.9), {"temperature": 0.7, "top_p": 0.9})
        self.assertEqual(list(rsn.sampling_for(False, min_p=0.1, top_k=5)), ["temperature", "top_k", "min_p"])

    def test_a_thinking_run_is_never_greedy(self):
        for kw in ({"temperature": 0}, {"temperature": 0.0}, {"top_k": 1}, {"temperature": -1}):
            with self.assertRaises(SystemExit, msg=kw):
                rsn.sampling_for(True, **kw)
        with self.assertRaisesRegex(SystemExit, "never greedy"):
            rsn.sampling_for(True, temperature=0)
        self.assertEqual(rsn.sampling_for(False, temperature=0), {"temperature": 0}, "a run that does not think may be asked for it")

    def test_values_out_of_range_are_refused(self):
        for kw in ({"temperature": 2.5}, {"top_p": 0}, {"top_p": 1.5}, {"top_k": -3}, {"min_p": 1.0}, {"min_p": -0.1}):
            for thinking in (True, False):
                with self.assertRaisesRegex(SystemExit, "out of range"):
                    rsn.sampling_for(thinking, **kw)

    def test_the_record_for_the_stats(self):
        self.assertEqual(rsn.sampling_record(True, rsn.THINKING_SAMPLING, 4096, seed_base=1234),
                         {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 4096, "seed_base": 1234})

    def test_the_other_defaults(self):
        self.assertEqual((rsn.DEFAULT_THINKING_MAX_TOKENS, rsn.DEFAULT_MAX_TOKENS, rsn.THINKING_REPORT_WINDOWS, rsn.THINKING_WINDOW), (4096, 768, (4096, 8192), 8192))
        self.assertEqual([rsn.default_max_tokens(r, t) for r, t in ((False, False), (True, False), (True, True))], [300, 768, 4096])

    def test_the_slot_holds_the_prompt_and_max_tokens(self):
        self.assertEqual([rsn.slot_context(n) for n in (300, 768, 4096, 4097, 6144, 8192)], [8192, 8192, 8192, 9216, 10240, 12288])
        self.assertTrue(rsn.prompt_fits(4096, 4096, 8192))
        self.assertFalse(rsn.prompt_fits(4097, 4096, 8192))
        for n in (768, 4096, 5000, 8192):
            self.assertGreaterEqual(rsn.slot_context(n), n + rsn.SLOT_PROMPT_ROOM)

    def test_the_prompt_fingerprint(self):
        m = rsn.messages({"pseudocode": "x", "question": "q"})
        self.assertEqual(rsn.prompt_sha(m), rsn.prompt_sha(json.loads(json.dumps(m))))
        self.assertNotEqual(rsn.prompt_sha(m), rsn.prompt_sha([m[0], {**m[1], "content": m[1]["content"] + " "}]))
        self.assertRegex(rsn.prompt_sha(m), r"^[0-9a-f]{16}$")


class ThinkingRequests(StubbedSampleStage):
    """The sampler's requests, its server, its stats."""
    KW = dict(reasoning=True, thinking=True, max_len=8192)

    def setUp(self):
        super().setUp()
        self.reply = lambda tid, i, body: think_body(reply(self.tasks[tid], f"Step for {tid}.\nThe constant is read once."), f"Reasoning about {tid}: the code reads it, then returns it.")
        self.popen = []
        real = modal_sample.subprocess
        modal_sample.subprocess = types.SimpleNamespace(**{**vars(real), "Popen": lambda *a, **k: self.popen.append(a[0]) or types.SimpleNamespace(kill=lambda: None)})
        self.addCleanup(setattr, modal_sample, "subprocess", real)

    def test_a_thinking_run_asks_with_qwen3s_thinking_settings(self):
        self.run_stage(False, **self.KW)
        self.assertEqual(len(self.bodies), len(self.tasks) * self.k)
        for b in self.bodies:
            self.assertEqual({k: b[k] for k in ("temperature", "top_p", "top_k", "min_p")}, {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0})
            self.assertEqual(list(b), ["model", "messages", "temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "chat_template_kwargs", "cache_prompt"])
            self.assertEqual((b["chat_template_kwargs"], b["cache_prompt"]), ({"enable_thinking": True}, False))
        self.assertEqual({b["seed"] for b in self.bodies}, {1234, 1235, 1236})

    def test_the_other_runs_keep_their_temperature_and_get_no_other_sampling_key(self):
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid], "Step.\nRead once."))
        self.run_stage(False, reasoning=True, max_len=4096)
        for b in self.bodies:
            self.assertEqual(list(b), ["model", "messages", "temperature", "seed", "max_tokens", "chat_template_kwargs", "cache_prompt"])
            self.assertEqual((b["temperature"], b["chat_template_kwargs"]), (0.7, {"enable_thinking": False}))
        self.run_stage(False, reasoning=True, max_len=4096, sampling={"temperature": 0.7, "top_p": 0.9})
        self.assertEqual({b["top_p"] for b in self.bodies}, {0.9}, "an override is sent")

    def test_overrides_reach_the_request_and_a_greedy_thinking_run_asks_nothing(self):
        self.run_stage(False, **self.KW, sampling={"temperature": 0.8, "top_k": 40})
        self.assertEqual({(b["temperature"], b["top_p"], b["top_k"], b["min_p"]) for b in self.bodies}, {(0.8, 0.95, 40, 0.0)}, "what is not overridden keeps the thinking default")
        for bad in ({"temperature": 0}, {"top_k": 1}):
            with self.assertRaisesRegex(SystemExit, "never greedy"):
                self.run_stage(False, **self.KW, sampling=bad)
            self.assertEqual(self.requests, [])

    def test_the_stats_record_the_sampling_and_the_slot(self):
        res = self.run_stage(False, **self.KW)
        self.assertEqual(res["stats"]["sampling"], {"thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "max_tokens": 4096, "seed_base": 1234, "slot_context": 8192, "parallel": 4})
        self.assertEqual(res["stats"]["slot"], {"slot_context": 8192, "max_prompt_tokens": 4096, "prompt_too_long": 0, "by_kind": {}})
        json.dumps(res["stats"])
        res = self.run_stage(False, reasoning=True, max_len=4096)
        self.assertEqual(res["stats"]["sampling"], {"thinking": False, "temperature": 0.7, "max_tokens": 768, "seed_base": 1234, "slot_context": 8192, "parallel": 4})
        self.assertNotIn("slot", res["stats"])
        self.reply = lambda tid, i, body: {"choices": [{"message": {"content": right(self.tasks[tid])}}]}
        res = self.run_stage(False)
        self.assertNotIn("sampling", res["stats"], "the default run's stats are phase A's")

    def test_max_tokens_is_4096_for_thinking_unless_given(self):
        self.run_stage(False, **self.KW)
        self.assertEqual({b["max_tokens"] for b in self.bodies}, {4096})
        self.run_stage(False, **self.KW, max_tokens=3000)
        self.assertEqual({b["max_tokens"] for b in self.bodies}, {3000})
        self.run_stage(False, reasoning=True, max_len=4096)
        self.assertEqual({b["max_tokens"] for b in self.bodies}, {768})
        with self.assertRaisesRegex(SystemExit, "--max-tokens"):
            self.run_stage(False, **self.KW, max_tokens=-5)

    def test_the_server_slot_holds_the_prompt_and_max_tokens(self):
        self.run_stage(False, **self.KW)
        cmd = self.popen[-1]
        self.assertEqual(cmd[cmd.index("-c") + 1], str(4 * 8192))
        self.assertEqual(cmd[cmd.index("--parallel") + 1], "4")
        self.assertIn("--jinja", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning-format") + 1], "deepseek")
        self.run_stage(False, **self.KW, max_tokens=6144)
        cmd = self.popen[-1]
        self.assertEqual(int(cmd[cmd.index("-c") + 1]) // 4, rsn.slot_context(6144), "a larger budget, a larger slot")
        self.assertGreaterEqual(int(cmd[cmd.index("-c") + 1]) // 4, 6144 + rsn.SLOT_PROMPT_ROOM)

    def test_the_other_runs_start_the_server_as_they_always_did(self):
        self.reply = lambda tid, i, body: answer_body(reply(self.tasks[tid], "Step.\nRead once."))
        self.run_stage(False, reasoning=True, max_len=4096)
        self.assertEqual(self.popen[-1], ["/app/llama-server", "-m", "/tmp/fake.gguf", "--host", "127.0.0.1", "--port", "8080", "-ngl", "999", "-c", "32768", "--parallel", "4", "-fa", "off"])
        self.run_stage(False)
        self.assertEqual(self.popen[-1][-2:], ["-fa", "off"])

    def test_a_prompt_that_leaves_less_than_max_tokens_in_its_slot_is_not_asked_and_is_counted(self):
        tid = sorted(self.tasks)[0]
        self.tasks[tid]["pseudocode"] = f"/* {tid} */\n" + "word " * 5000          # the stub counts one token per word
        (self.root / "data" / outputs.decompiled_stem(self.tag) / f"{tid}.json").write_text(json.dumps(self.tasks[tid]))
        res = self.run_stage(False, **self.KW)
        self.assertNotIn(tid, {t for t, _ in self.requests}, "not asked")
        self.assertEqual(len(self.requests), (len(self.tasks) - 1) * self.k)
        self.assertEqual(res["stats"]["slot"]["prompt_too_long"], 1)
        self.assertEqual(res["stats"]["slot"]["by_kind"], {self.tasks[tid]["kind"]: 1})
        self.assertIn("1 task(s) not asked", self.printed)
        self.assertNotIn(tid, {r["task"] for r in res["sft"]})
        self.assertTrue(any("1 task(s) were not asked" in l for l in sft.reasoning_report(res["stats"])))
        self.run_stage(False, **self.KW, max_tokens=2000)
        self.assertIn(tid, {t for t, _ in self.requests}, "a smaller budget fits the same prompt")

    def test_a_reply_cut_off_at_max_tokens_is_rejected_and_counted(self):
        self.reply = lambda tid, i, body: think_body("", "Still thinking, the reasoning goes on and on", finish="length")
        res = self.run_stage(False, **self.KW)
        self.assertEqual((res["stats"]["reasoning"]["truncated"], res["stats"]["kept_tasks"]), (len(self.tasks) * self.k, 0))
        self.assertTrue(any("cut off at max_tokens" in l and "WARNING" in l for l in sft.reasoning_report(res["stats"])), "and the run warns")

    def test_the_length_report_counts_what_fits_4096_and_8192(self):
        res = self.run_stage(False, **self.KW)
        fits = res["stats"]["over_length"]["fits"]["teacher-reasoning"]
        self.assertEqual(sorted(k for k in fits if k.isdigit()), ["4096", "8192"])
        self.assertEqual(fits["records"], fits["8192"], "everything fits 8192 here")
        line = sft.fits_line(res["stats"])
        self.assertIn("4096 tokens:", line)
        self.assertIn("8192 tokens:", line)
        self.assertNotIn("2048 tokens", line)
        self.assertIn("dropped at --max-len 8192", line)
        res = self.run_stage(False, reasoning=True, max_len=4096)
        self.assertEqual(sorted(k for k in res["stats"]["over_length"]["fits"]["teacher-reasoning"] if k.isdigit()), ["2048", "3072", "4096"], "visible reasoning keeps its three windows")
        self.assertIn("2048 tokens", sft.fits_line(res["stats"]))

    def test_the_share_that_fits_is_a_projection_of_the_window(self):
        words = lambda n: " ".join(f"w{j}" for j in range(n))                  # distinct words: a run of one word is a loop and is dropped
        self.reply = lambda tid, i, body: think_body(reply(self.tasks[tid], "Step.\nRead once."), words(5000 if i == 0 else 100))        # the stub counts words: sample 0 is ~5,000 tokens
        res = self.run_stage(False, reasoning=True, thinking=True, max_len=8192, max_tokens=7000)
        fits = res["stats"]["over_length"]["fits"]["teacher-reasoning"]
        self.assertEqual(fits["8192"], fits["records"])
        self.assertLess(fits["4096"], fits["records"], "the long traces do not fit a window of 4096")
        res = self.run_stage(False, reasoning=True, thinking=True, max_len=4096, max_tokens=7000)
        self.assertGreater(res["stats"]["over_length"]["dropped"], 0, "and at --max-len 4096 they are dropped")


class ThinkingEntry(ModalCase):
    def sample(self, root, **kw):
        return self.run_main(root, stage="sample", model=self.data.MODELS[0][0], seed=7, family="crc", **kw)

    def test_thinking_needs_no_explicit_max_tokens_any_more_and_says_what_it_will_use(self):
        with tempfile.TemporaryDirectory() as d:
            remote, _ = self.sample(pathlib.Path(d), reasoning=True, thinking=True, max_len=8192, quota="crc=10", k=4)
        kw = remote.call_args.kwargs
        self.assertEqual((kw["thinking"], kw["max_tokens"], kw["sampling"]), (True, 0, rsn.THINKING_SAMPLING), "0 = the default, 4096, decided on the sampler")
        self.assertIn("max_tokens 4096", self.printed)
        self.assertIn("sampling: {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0}", self.printed)
        self.assertIn("server slot 8,192 tokens", self.printed)
        self.assertIn("one 8,192-token slot of the server", self.printed)
        self.assertNotIn("note: --max-len", self.printed, "8192 is the recommended window")

    def test_a_window_below_8192_is_noted(self):
        with tempfile.TemporaryDirectory() as d:
            self.sample(pathlib.Path(d), reasoning=True, thinking=True, max_len=4096, quota="crc=10")
        self.assertIn("note: --max-len 4096 is below the recommended 8192 for thinking records", self.printed)

    def test_the_sampling_flags_are_overrides_and_greedy_thinking_stops_before_a_gpu_is_rented(self):
        with tempfile.TemporaryDirectory() as d:
            remote, _ = self.sample(pathlib.Path(d), reasoning=True, thinking=True, max_len=8192, temperature=0.8, top_k=40)
        self.assertEqual(remote.call_args.kwargs["sampling"], {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.0})
        for kw in (dict(temperature=0.0), dict(top_k=1)):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaisesRegex(SystemExit, "never greedy"):
                    self.sample(pathlib.Path(d), reasoning=True, thinking=True, max_len=8192, remote_side_effect=AssertionError("a GPU was rented"), **kw)
                self.assertEqual(list((pathlib.Path(d) / "results" / "phase6" / "rea-data").glob("sft-*")), [])

    def test_the_other_runs_pass_their_old_temperature(self):
        with tempfile.TemporaryDirectory() as d:
            remote, _ = self.sample(pathlib.Path(d), reasoning=True, max_len=2048)
            self.assertEqual(remote.call_args.kwargs["sampling"], {"temperature": 0.7})
            self.assertNotIn("sampling:", self.printed.replace("sampling: {'temperature': 0.7}", ""))
        with tempfile.TemporaryDirectory() as d:
            remote, _ = self.sample(pathlib.Path(d))
            self.assertEqual(remote.call_args.kwargs["sampling"], {"temperature": 0.7})

    def test_the_flags_exist_with_a_default_that_means_not_given(self):
        params = inspect.signature(self.data.main).parameters
        self.assertEqual([params[k].default for k in ("temperature", "top_p", "top_k", "min_p")], [-1.0, -1.0, -1, -1.0])
        self.assertIsNone(inspect.signature(self.data.sample_answers).parameters["sampling"].default)


# ---- the README ------------------------------------------------------------------------------------------------------------------------------------

README = (ROOT / "tools" / "modal" / "README.md").read_text()


class ReadmeSection(unittest.TestCase):
    def section(self):
        m = re.search(r"^## Thinking dataset \(next run\)\n(.*?)(?=^## |\Z)", README, re.S | re.M)
        self.assertIsNotNone(m, "the section `Thinking dataset (next run)`")
        return m.group(1)

    def test_the_section_has_the_whole_recipe(self):
        s = self.section()
        for part in ("codeforge-27b-heretic-q4", "SYSTEM_REASONING", "temperature 0.6", "top_p 0.95", "top_k 20", "min_p 0", "--k 4", "--max-tokens 4096", "--max-traces 3", "--max-len 8192", "kaggle-score --replies",
                     "--thinking", "rea_thinking", "--mix", "A100", "batch", "1 x 16", "RFT", "R1", "Qwen3", "no_think", "mixed_language", "repetition", "unreadable", "contradiction", "never greedy",
                     "Not measured", "tools/kaggle/README.md", "18.4 tokens"):
            self.assertIn(part, s, part)

    def test_the_old_notes_that_said_it_was_not_trainable_or_needed_an_explicit_max_tokens_are_gone(self):
        self.assertNotIn("Not trainable yet", README)
        self.assertNotIn("no default with `--thinking`", README)
        self.assertNotIn("**no default with `--thinking`** (it must be passed)", README)
        self.assertNotIn("needs `--reasoning` and an explicit `--max-tokens`", README)

    def test_the_numbers_in_the_recipe_are_the_ones_in_the_code(self):
        s = self.section()
        for k, v in rsn.THINKING_SAMPLING.items():
            self.assertRegex(s, rf"{k.replace('_', '_')} {v}\b" if k != "min_p" else r"min_p 0\b")
        self.assertIn(str(rsn.DEFAULT_THINKING_MAX_TOKENS), s)
        self.assertIn(str(rsn.THINKING_WINDOW), s)
        self.assertEqual(rsn.DEFAULT_MAX_TRACES, 3)


if __name__ == "__main__":
    unittest.main()

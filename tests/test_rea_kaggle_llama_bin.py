"""Round 14: the kernel reuses the llama-server of an attached `llama-bin/` (a previous kernel's output, or a dataset made from it) instead of building llama.cpp for 30 minutes: it is found, copied with its libraries,
put first in `LD_LIBRARY_PATH`, tried with `--version`, and the kernel builds when it does not start. A stand-in server script stands for llama-server; nothing here touches Kaggle or a GPU.

  PrebuiltServer  find, copy, `LD_LIBRARY_PATH`, the try, the fall back
  KernelPrebuilt  `run_teacher.main()` uses an attached server that starts and builds when there is none or it does not start
  AttachDataset   build_tasks `--llama-dataset` adds the dataset to the kernel's sources
"""
from __future__ import annotations

import os
import pathlib
import tempfile
import unittest
from unittest import mock

from .test_rea_kaggle import write_tasks
from .test_rea_kaggle_kit import make_tasks
from .test_rea_thinking_kit import KernelHarness
from .thinking_pilot import KIT, build_tasks, run_teacher

import json  # noqa: E402


FAKE_SERVER = """#!/bin/sh
# a stand-in llama-server: it starts only when the FIRST entry of LD_LIBRARY_PATH holds the library it needs, as the real one finds libggml next to itself
case "${LD_LIBRARY_PATH%%:*}" in
  "") exit 3 ;;
esac
[ -f "${LD_LIBRARY_PATH%%:*}/libfake.so.0" ] && echo "version: fake" && exit 0
exit 4
"""


class PrebuiltServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)

    def attach(self, where="llama-bin", with_lib=True, mode=0o644):
        folder = self.d / "input" / "someone-llama-bin" / where if where else self.d / "input" / "someone-llama-bin"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "llama-server").write_text(FAKE_SERVER)
        (folder / "llama-server").chmod(mode)                        # a mount that did not keep the executable bit
        if with_lib:
            (folder / "libfake.so.0").write_text("lib")
        (folder / "replies.jsonl").write_text("{}\n")               # other files of the dataset
        return folder / "llama-server"

    def test_a_llama_bin_of_an_attached_dataset_or_output_is_found_first_wherever_it_is(self):
        root = self.d / "input"
        self.assertIsNone(run_teacher.find_prebuilt(root))
        (root / "elsewhere").mkdir(parents=True)
        (root / "elsewhere" / "llama-server").mkdir()                  # a directory of that name is not a server
        self.assertIsNone(run_teacher.find_prebuilt(root))
        flat = root / "aaa-flat"
        flat.mkdir()
        (flat / "llama-server").write_text("")
        self.assertEqual(run_teacher.find_prebuilt(root), flat / "llama-server", "a dataset that holds the files at its root")
        (root / "zzz-kernel-output" / "llama-bin").mkdir(parents=True)
        (root / "zzz-kernel-output" / "llama-bin" / "llama-server").write_text("")
        self.assertEqual(run_teacher.find_prebuilt(root), root / "zzz-kernel-output" / "llama-bin" / "llama-server", "`llama-bin/llama-server` comes first")
        self.assertIsNone(run_teacher.find_prebuilt(self.d / "absent"))

    def test_the_copy_holds_the_server_and_its_libraries_and_nothing_else(self):
        found = self.attach()
        (found.parent / "libggml-cuda.so").write_text("lib")
        (found.parent / "notes.txt").write_text("x")
        new = run_teacher.prepare_prebuilt(found, self.d / "tmp" / "copy")
        self.assertEqual(sorted(p.name for p in new.parent.iterdir()), ["libfake.so.0", "libggml-cuda.so", "llama-server"])
        self.assertTrue(os.access(new, os.X_OK), "executable although the mount's file was not")

    def test_ld_library_path_starts_with_the_copy_and_the_server_is_tried_before_it_is_used(self):
        found = self.attach()
        env = {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64"}
        got = run_teacher.use_prebuilt(found, self.d / "copy", env)
        self.assertEqual(got, self.d / "copy" / "llama-server")
        self.assertEqual(env["LD_LIBRARY_PATH"], f"{self.d / 'copy'}:/usr/local/nvidia/lib64")
        env = {}
        self.assertIsNotNone(run_teacher.use_prebuilt(found, self.d / "copy2", env))
        self.assertEqual(env["LD_LIBRARY_PATH"], str(self.d / "copy2"), "no empty entry: an empty entry of LD_LIBRARY_PATH is the current directory")

    def test_a_server_that_does_not_start_is_not_used_and_leaves_the_environment_as_it_was(self):
        found = self.attach(with_lib=False)
        env = {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64", "OTHER": "1"}
        self.assertIsNone(run_teacher.use_prebuilt(found, self.d / "copy", env))
        self.assertEqual(env, {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64", "OTHER": "1"})
        env = {}
        self.assertIsNone(run_teacher.use_prebuilt(found, self.d / "copy2", env))
        self.assertEqual(env, {}, "unset before, unset after: nothing of the failed copy shadows the libraries of the build that follows")

    def test_the_kernel_builds_only_when_there_is_nothing_to_use(self):
        src = (KIT / "run_teacher.py").read_text()
        main = src[src.index("def main()"):]
        self.assertIn("binary = use_prebuilt(prebuilt)", main)
        self.assertIn("if binary is None:\n        binary = build_llama(cfg)", main)
        self.assertLess(main.index("find_prebuilt()"), main.index("build_llama(cfg)"))


class KernelPrebuilt(KernelHarness):
    """`run_teacher.main()` with an attached llama-server."""

    def more_patches(self) -> list:
        real_use = run_teacher.use_prebuilt
        return [mock.patch.object(run_teacher, "use_prebuilt", lambda found: real_use(found, self.d / "copy", dict(os.environ)))]

    def test_an_attached_server_that_starts_is_used_and_nothing_is_built(self):
        found = self.d / "input" / "llama-bin" / "llama-server"
        found.parent.mkdir(parents=True)
        found.write_text(FAKE_SERVER)
        (found.parent / "libfake.so.0").write_text("lib")
        self.write_dataset()
        run = self.run_main(prebuilt=found)
        self.assertEqual(self.calls.built, 0)
        self.assertEqual(self.calls.popen[0][0], str(self.d / "copy" / "llama-server"))
        self.assertEqual(run["llama_cpp"], f"prebuilt {found}")

    def test_an_attached_server_that_does_not_start_is_replaced_by_a_build(self):
        found = self.d / "input" / "llama-bin" / "llama-server"
        found.parent.mkdir(parents=True)
        found.write_text(FAKE_SERVER)                                    # no libfake.so.0 beside it: it does not start
        self.write_dataset()
        run = self.run_main(prebuilt=found)
        self.assertEqual(self.calls.built, 1)
        self.assertEqual(self.calls.popen[0][0], str(self.d / "built" / "llama-server"))
        self.assertNotIn("prebuilt", run["llama_cpp"])

    def test_without_an_attached_server_the_kernel_builds(self):
        self.write_dataset()
        self.run_main()
        self.assertEqual(self.calls.built, 1)


class AttachDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.file = write_tasks(self.d, make_tasks((("crc", 3), ("structs", 3))), "decompiled-seed9-all.jsonl")

    def build(self, out="kb", **kw):
        return build_tasks.build([self.file], self.d / out, "someone", **kw)

    def test_the_prebuilt_dataset_is_attached_to_the_kernel_next_to_the_tasks(self):
        self.build(llama_dataset="someone/llama-bin-t4")
        meta = json.loads((self.d / "kb" / "kernel" / "kernel-metadata.json").read_text())
        self.assertEqual(meta["dataset_sources"], ["someone/rea-reasoning-tasks", "someone/llama-bin-t4"])
        self.build("kb2")
        self.assertEqual(json.loads((self.d / "kb2" / "kernel" / "kernel-metadata.json").read_text())["dataset_sources"], ["someone/rea-reasoning-tasks"])
        with self.assertRaisesRegex(SystemExit, "owner/dataset-slug"):
            self.build("kb3", llama_dataset="llama bin")


if __name__ == "__main__":
    unittest.main()

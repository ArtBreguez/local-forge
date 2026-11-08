import tempfile
import unittest
from pathlib import Path

from forge import gguf

from .helpers import tiny_gguf


class GGUFParser(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_metadata_and_tensors_round_trip(self):
        p = tiny_gguf(self.dir / "m.gguf", layers=8, nextn=1, mtp_tensors=True)
        h = gguf.read_header(str(p))
        self.assertEqual(h.version, 3)
        self.assertEqual(h.architecture, "qwen35")
        self.assertEqual(h.arch_key("block_count"), 9)
        self.assertEqual(h.arch_key("embedding_length"), 64)
        self.assertEqual(h.arch_key("attention.head_count_kv"), 2)
        self.assertEqual(h.arch_key("nextn_predict_layers"), 1)
        self.assertEqual(h.quant_name, "IQ4_XS")           # general.file_type 30
        self.assertEqual(h.tensor_count, len(h.tensors))
        self.assertEqual(h.file_size, p.stat().st_size)

    def test_large_arrays_are_skipped_not_materialised(self):
        h = gguf.read_header(str(tiny_gguf(self.dir / "m.gguf")))
        v = h.metadata["tokenizer.ggml.tokens"]
        self.assertIsInstance(v, gguf.ArrayStub)
        self.assertEqual(v.length, gguf.MAX_KEPT_ARRAY + 10)

    def test_per_layer_sizes_come_from_tensor_offsets(self):
        h = gguf.read_header(str(tiny_gguf(self.dir / "m.gguf", layers=8, layer_kib=64)))
        lb = h.layer_bytes()
        self.assertEqual(sorted(lb), list(range(8)))
        for i in range(8):
            self.assertEqual(lb[i], 64 * 1024 + 256)       # ffn_up + norm, both already 32-aligned
        nl = h.non_layer_bytes()
        self.assertEqual(nl["token_embd.weight"], 16384)
        self.assertEqual(nl["output.weight"], 16384)
        # nothing lost: the tensors tile the data section exactly
        self.assertEqual(sum(t.nbytes for t in h.tensors), h.file_size - h.data_offset)

    def test_rejects_bad_magic_and_truncation(self):
        bad = self.dir / "bad.gguf"
        bad.write_bytes(b"NOPE" + b"\0" * 64)
        with self.assertRaises(gguf.GGUFError):
            gguf.read_header(str(bad))
        good = tiny_gguf(self.dir / "m.gguf")
        cut = self.dir / "cut.gguf"
        cut.write_bytes(good.read_bytes()[:300])
        with self.assertRaises(gguf.GGUFError):
            gguf.read_header(str(cut))

    def test_fingerprint_tracks_header_and_content(self):
        a = tiny_gguf(self.dir / "a.gguf", layers=8)
        b = tiny_gguf(self.dir / "b.gguf", layers=8)
        c = tiny_gguf(self.dir / "c.gguf", layers=9)
        fa, fb, fc = (gguf.read_header(str(x)).fingerprint() for x in (a, b, c))
        self.assertEqual(fa, fb)                            # same bytes -> same id regardless of path
        self.assertNotEqual(fa, fc)                         # different shape -> different id
        data = bytearray(b.read_bytes())
        data[-5] = 7                                        # flip a byte in the last sampled chunk
        b.write_bytes(bytes(data))
        self.assertNotEqual(gguf.read_header(str(b)).fingerprint(), fa)


if __name__ == "__main__":
    unittest.main()

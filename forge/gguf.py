"""Minimal GGUF header reader (stdlib only).

Reads the metadata key/values and the tensor-info table of a GGUF file without
loading any weights. The tensor sizes are derived from consecutive tensor
offsets, so no ggml type table is needed and the numbers are exact for any
quantisation (including types this file has never heard of).

GGUF layout (v2/v3, little endian):
    magic "GGUF" | u32 version | u64 tensor_count | u64 kv_count
    kv_count x (string key | u32 type | value)
    tensor_count x (string name | u32 n_dims | u64 dims[n_dims] | u32 ggml_type | u64 offset)
    padding to general.alignment (default 32), then the tensor data.
"""
from __future__ import annotations

import hashlib
import os
import re
import struct
from dataclasses import dataclass
from typing import Any, BinaryIO

MAGIC = b"GGUF"

# gguf_metadata_value_type
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(13)

_SCALARS = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h", T_UINT32: "<I", T_INT32: "<i",
    T_FLOAT32: "<f", T_BOOL: "<?", T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}

# arrays longer than this are not materialised (tokenizer vocab, merges, scores...)
MAX_KEPT_ARRAY = 4096

# llama_ftype (general.file_type) -> name. Source: llama.cpp include/llama.h.
FTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
}


class GGUFError(ValueError):
    pass


@dataclass
class ArrayStub:
    """Stands in for an array that was skipped because it is large."""
    length: int
    elem_type: int


@dataclass
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int          # relative to the start of the data section
    nbytes: int = 0      # filled in from neighbouring offsets


@dataclass
class GGUFHeader:
    path: str
    version: int
    tensor_count: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    data_offset: int
    file_size: int
    alignment: int = 32
    header_digest: str = ""

    # ---- convenience accessors -------------------------------------------------
    @property
    def architecture(self) -> str | None:
        return self.metadata.get("general.architecture")

    def arch_key(self, suffix: str, default: Any = None) -> Any:
        arch = self.architecture
        return self.metadata.get(f"{arch}.{suffix}", default) if arch else default

    @property
    def quant_name(self) -> str | None:
        ft = self.metadata.get("general.file_type")
        if ft is None:
            return None
        return FTYPE_NAMES.get(int(ft), f"ftype={ft}")

    def layer_bytes(self) -> dict[int, int]:
        """Bytes of weights per transformer block index (tensors named blk.N.*)."""
        out: dict[int, int] = {}
        for t in self.tensors:
            m = re.match(r"blk\.(\d+)\.", t.name)
            if m:
                out[int(m.group(1))] = out.get(int(m.group(1)), 0) + t.nbytes
        return out

    def non_layer_bytes(self) -> dict[str, int]:
        """Bytes of tensors outside the blocks (token_embd, output, output_norm, ...)."""
        return {t.name: t.nbytes for t in self.tensors if not t.name.startswith("blk.")}

    def fingerprint(self) -> str:
        """Fast identity of the model file: header digest + size + 3 sampled chunks.

        Not a full content hash (hashing 5 GB at every start would defeat the cache);
        a re-quantised or edited file changes the header, the size or the samples.
        """
        h = hashlib.sha256()
        h.update(self.header_digest.encode())
        h.update(str(self.file_size).encode())
        chunk = 64 * 1024
        with open(self.path, "rb") as f:
            span = max(self.file_size - self.data_offset - chunk, 0)
            for frac in (0.0, 0.5, 1.0):
                f.seek(self.data_offset + int(span * frac))
                h.update(f.read(chunk))
        return h.hexdigest()[:32]


# ---- low level readers ----------------------------------------------------------

def _read(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise GGUFError("unexpected end of file (truncated GGUF?)")
    return b


def _u32(f: BinaryIO) -> int:
    return struct.unpack("<I", _read(f, 4))[0]


def _u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", _read(f, 8))[0]


def _string(f: BinaryIO) -> str:
    n = _u64(f)
    if n > 1 << 24:
        raise GGUFError(f"implausible string length {n}")
    return _read(f, n).decode("utf-8", "replace")


def _skip_string(f: BinaryIO) -> None:
    n = _u64(f)
    f.seek(n, os.SEEK_CUR)


def _value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        fmt = _SCALARS[vtype]
        return struct.unpack(fmt, _read(f, struct.calcsize(fmt)))[0]
    if vtype == T_STRING:
        return _string(f)
    if vtype == T_ARRAY:
        etype = _u32(f)
        n = _u64(f)
        if n > MAX_KEPT_ARRAY:
            _skip_array(f, etype, n)
            return ArrayStub(n, etype)
        return [_value(f, etype) for _ in range(n)]
    raise GGUFError(f"unknown metadata value type {vtype}")


def _skip_array(f: BinaryIO, etype: int, n: int) -> None:
    if etype in _SCALARS:
        f.seek(struct.calcsize(_SCALARS[etype]) * n, os.SEEK_CUR)
    elif etype == T_STRING:
        for _ in range(n):
            _skip_string(f)
    else:
        for _ in range(n):  # nested arrays: rare, take the slow path
            _value(f, etype)


def read_header(path: str) -> GGUFHeader:
    """Parse metadata + tensor table of a GGUF file. Raises GGUFError on bad input."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        if _read(f, 4) != MAGIC:
            raise GGUFError(f"{path}: not a GGUF file (bad magic)")
        version = _u32(f)
        if version < 2:
            raise GGUFError(f"unsupported GGUF version {version} (need >= 2)")
        tensor_count = _u64(f)
        kv_count = _u64(f)
        if tensor_count > 1 << 22 or kv_count > 1 << 20:
            raise GGUFError("implausible tensor/kv count (corrupt header?)")
        meta: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _string(f)
            vtype = _u32(f)
            meta[key] = _value(f, vtype)
        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = _string(f)
            nd = _u32(f)
            if nd > 8:
                raise GGUFError(f"tensor {name}: {nd} dims")
            dims = struct.unpack(f"<{nd}Q", _read(f, 8 * nd)) if nd else ()
            gtype = _u32(f)
            off = _u64(f)
            tensors.append(TensorInfo(name, tuple(dims), gtype, off))
        end_of_info = f.tell()
        align = int(meta.get("general.alignment", 32)) or 32
        data_offset = (end_of_info + align - 1) // align * align
        f.seek(0)
        digest = hashlib.sha256(_read(f, end_of_info)).hexdigest()

    # size of each tensor = distance to the next offset (last one: to end of file)
    order = sorted(range(len(tensors)), key=lambda i: tensors[i].offset)
    data_size = file_size - data_offset
    for pos, i in enumerate(order):
        nxt = tensors[order[pos + 1]].offset if pos + 1 < len(order) else data_size
        tensors[i].nbytes = max(nxt - tensors[i].offset, 0)
    return GGUFHeader(path, version, tensor_count, meta, tensors, data_offset, file_size, align, digest)


# ---- writer (used by tests to build tiny synthetic models) ----------------------

def _w_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _w_value(v: Any) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I?", T_BOOL, v)
    if isinstance(v, int):
        return struct.pack("<II", T_UINT32, v) if 0 <= v < 1 << 32 else struct.pack("<Iq", T_INT64, v)
    if isinstance(v, float):
        return struct.pack("<If", T_FLOAT32, v)
    if isinstance(v, str):
        return struct.pack("<I", T_STRING) + _w_string(v)
    if isinstance(v, list):
        et = T_STRING if v and isinstance(v[0], str) else T_UINT32
        body = b"".join(_w_string(x) if et == T_STRING else struct.pack("<I", x) for x in v)
        return struct.pack("<IIQ", T_ARRAY, et, len(v)) + body
    raise TypeError(type(v))


def write_synthetic(path: str, metadata: dict[str, Any], tensors: list[tuple[str, tuple[int, ...], int, int]],
                    alignment: int = 32) -> None:
    """Write a GGUF with the given metadata and tensors [(name, dims, ggml_type, nbytes)].

    The data section is zero-filled. Only meant for tests; real models come from llama.cpp.
    """
    meta = dict(metadata)
    meta.setdefault("general.alignment", alignment)
    head = MAGIC + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        head += _w_string(k) + _w_value(v)
    offset = 0
    infos = b""
    layout = []
    for name, dims, gtype, nbytes in tensors:
        infos += _w_string(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        infos += struct.pack("<IQ", gtype, offset)
        layout.append((offset, nbytes))
        offset += (nbytes + alignment - 1) // alignment * alignment
    blob = head + infos
    pad = (-len(blob)) % alignment
    with open(path, "wb") as f:
        f.write(blob + b"\0" * pad)
        f.write(b"\0" * offset)

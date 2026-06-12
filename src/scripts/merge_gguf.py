"""
Merge llama.cpp split GGUF shards into a single file.

Some models on HuggingFace are split into multiple shards (e.g. model-00001-of-00002.gguf).
llama-cpp-python requires a single merged file. This script concatenates shards,
fixing tensor offsets in the header so the merged file is a valid single GGUF.

Usage: edit the shard/output paths at the bottom of the file and run directly.
Currently only supports 2-shard merges.
"""
import struct
import sys
from pathlib import Path


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3


def read_uint32(f):
    return struct.unpack("<I", f.read(4))[0]


def read_uint64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_string(f):
    length = read_uint64(f)
    return f.read(length).decode("utf-8")


def skip_value(f, vtype):
    if vtype == 8:  # string
        length = read_uint64(f)
        f.seek(length, 1)
    elif vtype == 9:  # array
        elem_type = read_uint32(f)
        count = read_uint64(f)
        for _ in range(count):
            skip_value(f, elem_type)
    elif vtype in (0, 1):  # uint8/int8
        f.seek(1, 1)
    elif vtype in (2, 3):  # uint16/int16
        f.seek(2, 1)
    elif vtype in (4, 5, 6, 10):  # uint32/int32/float32/bool
        f.seek(4, 1)
    elif vtype in (7, 11, 12):  # float64/uint64/int64
        f.seek(8, 1)
    else:
        raise ValueError(f"Unknown value type: {vtype}")


def merge(shard_paths: list[Path], output_path: Path):
    print(f"Merging {len(shard_paths)} shards -> {output_path}")

    # Read all shard data
    shards = []
    for p in shard_paths:
        shards.append(p.read_bytes())
        print(f"  Read {p.name}: {len(shards[-1]) / 1e9:.2f} GB")

    # Parse shard 0 header to get metadata and tensor info positions
    import io
    s0 = io.BytesIO(shards[0])

    magic = s0.read(4)
    assert magic == GGUF_MAGIC, f"Not a GGUF file: {magic}"
    version = read_uint32(s0)
    tensor_count_s0 = read_uint64(s0)
    kv_count = read_uint64(s0)

    print(f"  Shard 0: version={version}, tensors={tensor_count_s0}, kv_count={kv_count}")

    # Read all key-value metadata from shard 0
    kv_start = s0.tell()
    for _ in range(kv_count):
        read_string(s0)  # key
        vtype = read_uint32(s0)
        skip_value(s0, vtype)
    kv_end = s0.tell()

    # Read tensor info from shard 0
    tensor_infos_s0_start = s0.tell()
    tensor_infos_s0 = []
    for _ in range(tensor_count_s0):
        name = read_string(s0)
        n_dims = read_uint32(s0)
        dims = [read_uint64(s0) for _ in range(n_dims)]
        dtype = read_uint32(s0)
        offset = read_uint64(s0)
        tensor_infos_s0.append((name, n_dims, dims, dtype, offset))
    tensor_infos_s0_end = s0.tell()

    alignment = 32
    data_start_s0 = ((tensor_infos_s0_end + alignment - 1) // alignment) * alignment

    # Parse shard 1+
    all_tensor_infos = list(tensor_infos_s0)
    shard_data_blocks = [shards[0][data_start_s0:]]

    for idx, shard_bytes in enumerate(shards[1:], start=1):
        s = io.BytesIO(shard_bytes)
        magic = s.read(4)
        assert magic == GGUF_MAGIC
        read_uint32(s)  # version
        tensor_count_si = read_uint64(s)
        kv_count_si = read_uint64(s)
        print(f"  Shard {idx}: tensors={tensor_count_si}, kv_count={kv_count_si}")

        # Skip KV in this shard
        for _ in range(kv_count_si):
            read_string(s)
            vtype = read_uint32(s)
            skip_value(s, vtype)

        # Read tensor infos (offsets are relative to this shard's data block)
        ti_start = s.tell()
        for _ in range(tensor_count_si):
            name = read_string(s)
            n_dims = read_uint32(s)
            dims = [read_uint64(s) for _ in range(n_dims)]
            dtype = read_uint32(s)
            offset = read_uint64(s)
            all_tensor_infos.append((name, n_dims, dims, dtype, offset))
        ti_end = s.tell()
        data_start_si = ((ti_end + alignment - 1) // alignment) * alignment
        shard_data_blocks.append(shard_bytes[data_start_si:])

    # Build merged data block (concatenate shard data blocks)
    # Offsets in shard 1+ need to be shifted by cumulative data sizes
    cumulative = [0]
    for block in shard_data_blocks[:-1]:
        cumulative.append(cumulative[-1] + len(block))

    # Reassign offsets: tensors in shard i get offset += cumulative[i]
    tensor_count_per_shard = [tensor_count_s0] + [
        len(all_tensor_infos) - tensor_count_s0  # rough, only works for 2 shards
    ]

    # For 2-shard case: first tensor_count_s0 tensors -> shard 0, rest -> shard 1
    adjusted = []
    for i, ti in enumerate(all_tensor_infos):
        shard_idx = 0 if i < tensor_count_s0 else 1
        name, n_dims, dims, dtype, offset = ti
        adjusted.append((name, n_dims, dims, dtype, offset + cumulative[shard_idx]))

    # Write merged GGUF
    print(f"  Writing merged file...")
    with open(output_path, "wb") as out:
        # Header
        out.write(GGUF_MAGIC)
        out.write(struct.pack("<I", version))
        out.write(struct.pack("<Q", len(adjusted)))
        out.write(struct.pack("<Q", kv_count))

        # KV metadata (copy verbatim from shard 0)
        out.write(shards[0][kv_start:kv_end])

        # Tensor infos
        ti_section_start = out.tell()
        for name, n_dims, dims, dtype, offset in adjusted:
            name_bytes = name.encode("utf-8")
            out.write(struct.pack("<Q", len(name_bytes)))
            out.write(name_bytes)
            out.write(struct.pack("<I", n_dims))
            for d in dims:
                out.write(struct.pack("<Q", d))
            out.write(struct.pack("<I", dtype))
            out.write(struct.pack("<Q", offset))

        # Alignment padding
        cur = out.tell()
        pad = ((cur + alignment - 1) // alignment) * alignment - cur
        out.write(b"\x00" * pad)

        # Data blocks
        for block in shard_data_blocks:
            out.write(block)

    size = output_path.stat().st_size
    print(f"Done: {output_path}  ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    downloads = Path(r"C:\Users\Arseniy\Downloads")
    shard1 = downloads / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    shard2 = downloads / "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
    output = downloads / "qwen2.5-7b-instruct-q4_k_m.gguf"

    if output.exists():
        print(f"Output already exists: {output}")
        sys.exit(0)

    merge([shard1, shard2], output)

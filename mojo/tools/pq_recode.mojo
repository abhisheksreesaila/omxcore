# pq_recode <in.parquet> <out.parquet> <zstd|snappy|none> <rows per group> [threads]: reads any Parquet file and writes it back with the
# codec named, compressing on at most `threads` threads (F-187; 0 or none given: all the build has) (used by tests/test_codec_mojo.py: built with -D MXFRAME_NO_CODECS it reads and writes with the Mojo codecs only).
from std.sys import argv
from omxcore import read_parquet, write_parquet, UNCOMPRESSED, SNAPPY, ZSTD


def main() raises:
    var args = argv()
    var codec = ZSTD if String(args[3]) == "zstd" else SNAPPY if String(args[3]) == "snappy" else UNCOMPRESSED
    var r = read_parquet(String(args[1]))
    var t = r.take_table()
    write_parquet(String(args[2]), t, r.metadata, codec, Int(String(args[4])), threads=Int(String(args[5])) if len(args) > 5 else 0)

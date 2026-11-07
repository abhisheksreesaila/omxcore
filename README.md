# omxcore

**The data engine for the agent era.**

omxcore runs SQL on your CPU *and* your GPU, over the CSV and Parquet files you already have. It's one native library,
written in Mojo, with no Python inside. Apps link it like SQLite. Agents talk to it in JSON.

### ✨ [Read the story →](https://abhisheksreesaila.github.io/omxcore/story.html) · ▶ [Watch it as slides](https://abhisheksreesaila.github.io/omxcore/deck.html)
The journey, why Mojo, why it's built for agents, the numbers, and where it's going.

[![Story](https://img.shields.io/badge/the%20story-read-0e7c86)](https://abhisheksreesaila.github.io/omxcore/story.html) [![Mojo](https://img.shields.io/badge/written%20in-Mojo%201.1-0e7c86)](https://github.com/abhisheksreesaila/omxcore/tree/main/mojo) [![PyPI](https://img.shields.io/pypi/v/mxframe?label=PyPI%20(mxframe)&color=5d6571)](https://pypi.org/project/mxframe/) [![License](https://img.shields.io/badge/license-Apache--2.0-5d6571)](LICENSE)

- **Your files stay yours.** No import, no new format. Point it at a CSV or a Parquet file and ask.
- **Your hardware, all of it.** CPU or GPU, chosen per query, with the reason.
- **Answers an agent can trust.** Exact decimals, errors with a line, a column and a hint, a note for every decision.

## Try it

```sh
mx sql "SELECT region, sum(revenue) AS revenue FROM 'orders.csv' GROUP BY region"    # JSON out, no Python
```

```python
import omxcore as mx   # 0.6.0 from PyPI: pip install "mxframe[runtime,sql]", then import mxframe as mx

out = mx.sql("SELECT dept, SUM(x) AS s FROM t GROUP BY dept ORDER BY dept", t=mx.scan_parquet("t.parquet").compute())
print(out.compute())
```

On 0.6.0, all 22 TPC-H queries at scale factor 10 ran in **3.15 s** (DuckDB 5.08 s, Polars 14.40 s), every answer equal
to DuckDB's. Where it's still behind is [on the engine page](https://abhisheksreesaila.github.io/omxcore/engine.html).

## Latest numbers (November 2026, the Mojo engine)

The `mx sql` engine against DuckDB, one fresh process per query, every answer equal to DuckDB's. "Faster" counts the queries where
omxcore wins; the ratio is DuckDB's time over ours (above 1x means omxcore is faster). Random queries are 50-100 generated SQL queries
(joins, groups, filters, distinct counts), not tuned for.

| machine | random queries | TPC-H |
|---|---|---|
| Linux, 24-core CPU, SF1 | 84 / 100 faster, **1.58x** | 7 / 22, 0.76x |
| Linux, 24-core CPU, SF10 | 38 / 59 faster, **1.26x** | 3 / 22, 0.51x |
| M4 Mac, CPU, SF1 | 45 / 50 faster, **1.89x** | 4 / 22, 0.67x |
| M4 Mac, Apple GPU, SF1 | 23 / 50, 0.49x | 1 / 22, 0.23x |

Honest notes: the Linux runs were on a busy shared machine (noisy). Multi-table joins are still 2-3x behind DuckDB, which is most of
the TPC-H gap. The Apple GPU is slower than the M4's own CPU today (a new GPU kernel is planned), so AUTO uses the CPU on a Mac. The
NVIDIA RTX 3090 GPU has not been re-measured on this engine yet. Size: about 13 MB against DuckDB's 60 MB.
Details: [docs/bench/2026-11-07-status-summary.md](docs/bench/2026-11-07-status-summary.md).

## Read on

| | |
|---|---|
| ✨ [The story](https://abhisheksreesaila.github.io/omxcore/story.html) | why omxcore, why Mojo, why it's built for agents |
| ⚡ [The engine](https://abhisheksreesaila.github.io/omxcore/engine.html) | what it does and how it got fast, every number ([long form](docs/mojo-engine.md)) |
| 📊 [Benchmarks](https://abhisheksreesaila.github.io/omxcore/benchmarks.html) | every TPC-H query against DuckDB and Polars |
| 🧭 [CPU or GPU](https://abhisheksreesaila.github.io/omxcore/auto-mode.html) | how AUTO decides where a query runs |
| 📘 [Reference](docs/reference.md) | install, the Python API tables, the latest release, the Mojo example |
| 🔥 [Mojo API](docs/omxcore-mojo-api.md) | tables, frames, SQL, jobs and blocks, with no Python |
| 🧱 [Internals](https://abhisheksreesaila.github.io/omxcore/internals.html) | how a query runs, layer by layer |
| 🗺️ [Where it's going](docs/briefs/positioning.md) | the positioning and the path |

Build from source with [pixi](https://pixi.sh): `git clone https://github.com/abhisheksreesaila/omxcore && cd omxcore && pixi install && pixi run test`.
[Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · License: [Apache-2.0](LICENSE).

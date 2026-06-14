# Latest Test Results

Generated on 2026-06-14 after clearing the previous `runs` artifacts.

## Unit Tests

| Command | Result | Report |
| --- | --- | --- |
| `python3 -m pytest -q` | 128 passed, 982 warnings in 104.97s | `runs/latest/pytest-report.txt` |

## Persian Classification Benchmark

Configuration: `seed=0`, `epochs=50`, `hidden_dim=8`, `learning_rate=0.01`, `dropout=0.2`, `device=cpu`.

| Model | Validation accuracy | Test accuracy | Test macro-F1 | Report |
| --- | ---: | ---: | ---: | --- |
| `gat` | 1.00 | 1.00 | 1.00 | `runs/benchmarks/persian-classification-gat/metrics.json` |
| `gcn` | 1.00 | 0.75 | 0.60 | `runs/benchmarks/persian-classification-gcn/metrics.json` |
| `graphsage` | 1.00 | 1.00 | 1.00 | `runs/benchmarks/persian-classification-graphsage/metrics.json` |

## Graph-LM Smoke Runs

Configuration: `data/expanded_persian_lm.txt`, `seed=0`, `epochs=3`, `d_model=64`, `n_layers=1`, `block_size=64`, `device=cpu`.

| Run | Best validation loss | Best next-token loss | Best perplexity | Report |
| --- | ---: | ---: | ---: | --- |
| `baseline-s0` | 6.023 | 6.023 | 412.786 | `runs/graph-lm-smoke/baseline-s0/metrics.json` |
| `graph-gcn-s0` | 5.985 | 5.985 | 397.509 | `runs/graph-lm-smoke/graph-gcn-s0/metrics.json` |

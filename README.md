# numpy-torch-compile-benchmark
The benchmarking code between numpy and torch compiled numpy

## How to run
Install asv first:
```bash
pip instal asv
```

Then in the root folder to this repo, run:
```bash
asv run 2>&1 | tee log.txt
python parse_log.py -o vis_folder
```

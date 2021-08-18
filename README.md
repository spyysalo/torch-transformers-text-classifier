# torch-transformers-text-classifier

Simple text classifier using Transformers with the Torch backend.

## Quickstart

Clone a repository with example data in FastText format

```
git clone https://github.com/spyysalo/ylilauta-corpus.git
```

Start a Slurm run with tokenizer in directory `TOKENIZER` and model in directory `MODEL`. (The script `slurm-run.sh` assumes the CSC environment and will need to be edited to run in other environments.)

```
sbatch slurm-run.sh python train.py --tokenizer TOKENIZER --model MODEL --data ylilauta-corpus/data/10-percent
```

Once the run starts, follow the output with

```
tail -f logs/latest.*
```

Once the run completes, get the best dev set result and the test set result with

```
egrep eval_accuracy logs/latest.out | perl -pe 's/.*eval_accuracy.: (\S+),.*/$1/' | sort -rn | head -n 1
egrep test_accuracy logs/latest.out | perl -pe 's/.*test_accuracy.: (\S+),.*/$1/'
```

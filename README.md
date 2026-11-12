# ResRank clone

Reimplementation of ResRank (arXiv 2604.22180): passages are compressed to one embedding each by an
Encoder-LLM, a Reranker-LLM reads the query plus all embeddings in one pass, and passages are scored by
`cos(h_eos, h_i + e_i)`. See `../RESRANK_PLAN.md` for the plan, decisions and the list of unverified items.

## Status

165 unit tests pass on CPU (tiny random Qwen3 models + a fake tokenizer). Nothing has been run against the
real pretrained models, real tokenizers, real datasets or a GPU yet. Treat the first real run as a debugging
session, using the `checks/` scripts below.

## Layout

| Path | Purpose |
|---|---|
| `configs/` | `base.yaml` (models, losses, model options), `stage1_small.yaml`, `stage2_small.yaml` (training) |
| `src/resrank/model/` | `encoder`, `reranker`, `fusion`, `scoring`, `losses`, `sequence`, `resrank` (full model), `loading`, `diagnostics`, `prompts`, `pooling` |
| `src/resrank/data/` | `e2rank`, `pe_rank` loaders, `filters` (English), `sampling`, `beir`, `bm25`, stages `s01_download`, `s02_build_train`, `s03_bm25` |
| `src/resrank/train/` | `trainer`, `run_training` (stages s04 and s05) |
| `src/resrank/eval/` | `metrics`, `ordering`, stages `s06_encode_corpus`, `s07_rerank`, `s08_evaluate` |
| `checks/` | scripts to run first on real data/models (need a GPU or patience) |
| `tests/` | `python -m pytest` |

## Suggested first run (Kaggle)

    pip install torch transformers huggingface_hub pyyaml numpy pytest bitsandbytes
    python -m pytest -q                                           # sanity: should pass anywhere

    # 1. data
    python -m resrank.data.s01_download pe_rank --list            # confirm file names
    python -m resrank.data.s01_download pe_rank
    python -m resrank.data.s01_download e2rank --list
    python -m resrank.data.s01_download e2rank
    python checks/inspect_pe_rank.py $ROOT/data/01_raw/pe_rank/train.jsonl --n 3
    python checks/inspect_e2rank.py  $ROOT/data/01_raw/e2rank/train.jsonl --limit 20000
    python -m resrank.data.s02_build_train stage1 --input .../pe_rank/train.jsonl --n 2000 --n-candidates 16
    python -m resrank.data.s02_build_train stage2 --input .../e2rank/train.jsonl --name stage2_e2rank_en --n 2000
    # optional: drop training samples whose query is an evaluation query (off by default)
    #   ... --exclude-queries $ROOT/data/01_raw/beir/scifact/queries.jsonl
    # The English filter is a script check: Latin-script languages (Finnish, Indonesian, Swahili, ...) still pass,
    # so pick sources with --allow-sources after reading checks/inspect_e2rank.py's source table
    # (its english_words column and 'looks English' line separate them; notebooks/stage2_train.ipynb does this).

    # 2. model checks (decide inject_mode / inject_scale from the norm report)
    python checks/check_shapes.py        --train-file stage1 --n 2
    python checks/dump_intermediates.py  --train-file stage1 --n 2 --name first_look
    python checks/check_grad_flow.py     --train-file stage1 --n 2
    python checks/memory_dry_run.py      --train-file stage1 --n 1 --optimizer adamw8bit --precision fp16
    python checks/check_overfit.py       --train-file stage1 --n 8 --steps 30 --optimizer adamw8bit --precision fp16
    # if either runs out of memory, add --freeze-embeddings (and set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)

    # 3. train
    python -m resrank.train.run_training --train-config configs/stage1_small.yaml
    python -m resrank.train.run_training --train-config configs/stage2_small.yaml

    # Back up checkpoints (Kaggle wipes /kaggle/working): add --hub-repo you/resrank-ckpts to training (needs HF_TOKEN),
    # or afterwards: python -m resrank.hub push --ckpt stage1 --repo you/resrank-ckpts ; restore with 'pull'

    # 4. evaluate (BEIR-format files for the dataset)
    python -m resrank.data.s03_bm25 --dataset scifact --corpus corpus.jsonl --queries queries.jsonl
    python -m resrank.eval.s06_encode_corpus --dataset scifact --corpus corpus.jsonl --ckpt stage2_e2rank \
        --only-candidates $ROOT/data/03_candidates/scifact.bm25.run
    python -m resrank.eval.s07_rerank --dataset scifact --queries queries.jsonl --ckpt stage2_e2rank \
        --orders original reversed random
    python -m resrank.eval.s08_evaluate --dataset scifact --qrels qrels/test.tsv \
        --run bm25=$ROOT/data/03_candidates/scifact.bm25.run \
        --run resrank=$ROOT/data/05_runs/scifact.stage2_e2rank.original.run

(Run `python checks/...` from the repo root; use `PYTHONPATH=src` or `pip install -e .` for the `-m` commands.)

## Conventions

- All indices inside the code are 0-indexed; the E2Rank and PE-Rank files are 1-indexed and are converted once
  when loaded.
- Rank 0 = most relevant. RankNet pairs come from `ranks[j] < ranks[k]`; equal ranks make no pair.
- Defaults follow the paper: tau1 = tau2 = 0.05, lambda = 0.1, RankNet summed over pairs.
- Things that are assumptions rather than paper facts are marked ASSUMED in the code and listed in the plan.

## Keeping checkpoints (the Kaggle working folder is wiped when the session ends)

Two backends, use either (or both). Each checkpoint is about 4.8 GB (both 0.6B models in fp32).

**Kaggle dataset** (`train/publish.py`), set in the train config:

    publish_dataset: <kaggle-username>/<dataset-name>
    publish_every_saves: 0        # 0 = only at the end; N = also after every Nth periodic save

Token: create a Kaggle API token, store it as a notebook Secret named `KAGGLE_API_TOKEN`, and in a Python cell run

    import os
    from kaggle_secrets import UserSecretsClient
    os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")

Manual upload: `python -m resrank.train.publish --dataset <user>/<name> --names stage1`.
Each upload is a new version containing exactly the folders staged, so include every checkpoint to keep.
To resume in a new session, attach the dataset as an input and copy it back:
`cp -r /kaggle/input/.../<name>/* /kaggle/working/checkpoints/` (check the exact path with `ls /kaggle/input`).

**Hugging Face Hub** (`hub.py`): set `hub_repo: you/resrank-ckpts` and `HF_TOKEN` (see `hub.py`).

Never put a token in code or a config file.

## Starting stage 2 from a stage 1 that was stopped early

If stage 1 was stopped before its final save, `checkpoints/stage1_full/` holds only `latest/`. `init_from: stage1_full`
then uses `stage1_full/latest` (it says so). Publishing keeps a `latest/` snapshot whenever it is the only copy
of a checkpoint's weights.

## Resuming an interrupted run

Every checkpoint folder holds `train_state.json` (step, loss scale, seed, data settings). If the session dies:

1. Start a new session, restore the checkpoint folder into `/kaggle/working/checkpoints/` (from your Kaggle dataset or HF repo). It must include the `latest/` subfolder, so set `publish_every_saves` (e.g. 5) during training: with `0`, only the final checkpoint is uploaded and a mid-run disconnect loses everything.
2. Re-run the same command with the same train file and config, adding `--resume`:

       python -m resrank.train.run_training --train-config configs/stage1_small.yaml --resume

It reloads the weights from `checkpoints/<out_name>/latest`, skips the data already used, and continues the learning-rate schedule from that step. It refuses to resume if the seed, batch settings, epochs, learning rate or number of samples differ from the interrupted run.

What is not restored: the optimizer's moment estimates (they restart from zero, softened by a 10-step learning-rate ramp). Seeds: data selection and data order are deterministic given the seed (default 42), but GPU runs are not bit-for-bit repeatable.

## Starting a fresh Kaggle session (everything in /kaggle/working is gone)

Attach two datasets as inputs: the one holding this code and the checkpoints dataset. Turn Internet on, and pick a GPU
only for the training and evaluation steps. Then:

    # Python cell
    import os
    from kaggle_secrets import UserSecretsClient
    os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")

    # bash cell
    cp -r /kaggle/input/<path-to-code-dataset>/resrank /kaggle/working/resrank
    cd /kaggle/working/resrank && pip install -q -e . && pip install -q transformers huggingface_hub pyyaml pytest bitsandbytes kagglehub
    mkdir -p /kaggle/working/checkpoints
    cp -r /kaggle/input/<path-to-checkpoints-dataset>/* /kaggle/working/checkpoints/

Rebuild the training data with exactly the same arguments as before (the build is deterministic for a given file, seed and
arguments; resume checks the sample count):

    python -m resrank.data.s01_download pe_rank
    python -m resrank.data.s02_build_train stage1 --input /kaggle/working/data/01_raw/pe_rank/train.jsonl --n 20000 --n-candidates 16

Then start (first time) or continue (after an interruption, add `--resume`):

    python -m resrank.train.run_training --train-config configs/stage1_small.yaml [--resume]

Snapshots are written every `save_every_minutes` (15) to `checkpoints/<out_name>/latest` and uploaded when
`publish_dataset` is set. Use `--no-publish` for trial runs.

## The notebook

`notebooks/stage1_train.ipynb` (also supplied separately) runs everything for stage 1 in order: setup, timing run,
sample-count planning from the measured speed and the hours left, data build, training with a progress bar
(`--auto-resume`: fresh start, resume, or nothing if finished), a log summary, and a SciFact evaluation.
The planned sample count is saved to `checkpoints/stage1_full/plan.json` (and uploaded with the checkpoints) so that
a resumed session builds exactly the same training file.

## Disk and storage limits (learned on a real run)

- **Working disk:** `/kaggle/working` is limited to about 19.5 GB and one checkpoint (encoder + reranker, fp32) is about
  4.8 GB. Keep only what you need there: restore just `stage1_full`, and use `--no-save` for timing runs. Snapshots are
  swapped in atomically only if there is room for two copies; otherwise the old one is deleted first. A snapshot that
  fails to save prints a warning and training continues; the final save frees the periodic snapshot first if the disk is tight.
- **Kaggle dataset quota:** every upload is a new dataset version of about 4.8 GB that counts in full (200 GB limit; 18 GB was
  used after two versions). Hourly uploads (`publish_every_saves: 4` with 15-minute snapshots) for a 22-hour run use about
  106 GB. Do not upload every snapshot.
- GPU memory in the real training loop peaks around 15.1 GB of the T4's 15.6 GB. If `skipped=` in the progress bar keeps
  growing, set `model.freeze_embeddings: true` in `configs/base.yaml` (saves about 1.7 GiB; a departure from the paper).

## Two GPUs: train in the background, evaluate on the other

The notebook has an optional section (cells 8b to 8f): start training in the background on GPU 0 (`setsid nohup ... &`,
output to `/kaggle/working/train.log`), watch it with the monitor cell, evaluate the newest snapshot on GPU 1
(`CUDA_VISIBLE_DEVICES=1`) from a hard-linked copy of `stage1_full/latest`, and compare two runs with a paired bootstrap.
Use these instead of the blocking training cell, not in addition to it.

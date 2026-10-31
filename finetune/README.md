# Fine-tuning MiniLM on SciFact

`finetune_minilm_scifact.ipynb` fine-tunes the app's embedding model
(`sentence-transformers/all-MiniLM-L6-v2`) on the BEIR SciFact **train** split and saves the best
model as `minilm-scifact.zip`.

What the notebook does:

1. Downloads SciFact from the official BEIR zip.
2. **Leakage check:** drops train claims that are word-for-word copies of test claims (the official
   split has two: train 871 = test 870, train 1291 = test 1292), then asserts that no test query
   remains, by id or by text.
3. Holds out 10% of the train queries (seeded) as a dev set.
4. Mines 2 hard negatives per (query, relevant document) pair: BM25's top-ranked documents that are
   not labelled relevant, using the project's BM25 tokenizer.
5. Trains with `MultipleNegativesRankingLoss` (batch 32, lr 2e-5, 10% warmup, up to 3 epochs,
   no duplicate positives in a batch, seed 42), evaluating nDCG@10 on dev after each epoch.
6. Keeps the epoch with the best dev nDCG@10, prints base vs each epoch, and zips the model with a
   `training_summary.json`.

The test split is never used in the notebook.

**Result** (Colab T4, seed 42; details in the main [README](../README.md#fine-tuning)): dev nDCG@10
0.694 → 0.751 after 3 epochs. On the SciFact test set, dense nDCG@10 rose from 0.665 to 0.721
(+0.055, 95% CI [+0.035, +0.077]). On the 27 lecture-PDF questions the change was mixed and within
noise, so the app keeps the base model.

## Run it on Colab

1. Open the notebook in Colab: **File → Upload notebook**, or
   `https://colab.research.google.com/github/yashkumar172/multimodal-document-intelligence/blob/main/finetune/finetune_minilm_scifact.ipynb`
   once it is on GitHub's `main` branch.
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all.** It takes roughly 5–10 minutes; `minilm-scifact.zip` downloads at the end.

The first cell installs the pinned `sentence-transformers` and `rank-bm25` versions plus `datasets`
and `accelerate`, so nothing else is needed on Colab.

## Use the model locally

Benchmarking the model needs only the project's `requirements.txt`. Unzip it into
`finetune/models/` (gitignored), then from the project root:

```bash
# SciFact test set: adds "fine-tuned" rows and paired differences vs the base model
python benchmarks/benchmark_scifact.py --model finetune/models/minilm-scifact --label fine-tuned

# The 27 lecture-PDF questions (re-embeds the app's chunks in memory; chroma_db/ is only read)
python benchmarks/benchmark_retrieval.py --model finetune/models/minilm-scifact --label fine-tuned
```

Run `python benchmarks/benchmark_scifact.py` (no `--model`) once first, so the fine-tuned run can be
compared query by query with the base model.

The app itself still uses the base model. Switching would mean changing `MODEL_NAME` in
`src/embeddings.py` and re-indexing every PDF, and is only worth it if the lecture-PDF benchmark
improves.

## Running the notebook locally

Training needs two extra packages on top of the project's requirements, kept out of the main
`requirements.txt` so the deployed app doesn't install them:

```bash
pip install -r requirements.txt -r finetune/requirements.txt
```

Then open the notebook in VS Code or Jupyter and skip the first (`%pip`) cell. Set the environment
variable `SMOKE_TEST=1` before starting the kernel for a quick end-to-end check on a tiny subset.

Full training on a laptop CPU is not recommended: a batch of 32 rows is about 128 sequences of up to 256 tokens (query, positive,
two negatives), which ran out of memory on a 16 GB laptop (PyTorch's CPU build crashed with an
access violation rather than a clean out-of-memory error). `SMOKE_TEST=1` runs the whole notebook
on a tiny subset with batch size 8, which is how it was checked end to end on CPU.

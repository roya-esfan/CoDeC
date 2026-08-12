# *Alle mot 1* LLM research

This repository contains code for two parts of a research project using
questions from the Norwegian television game show *Alle mot 1*:

- contamination detection using CoDeC; and
- analysis of a pilot comparing Qwen3-VL predictions across prompt templates.

## CoDeC

The [`codec`](codec/) folder applies Contamination Detection via Context
(CoDeC) to question transcripts from *Alle mot 1*. Following Zawalski et al.
(2026), each season is treated as a dataset and each question transcript as one
sample.

### Notebooks

- [`codec/codec_qwen.ipynb`](codec/codec_qwen.ipynb) runs CoDeC with prompt log
  probabilities returned by Fireworks. The same pipeline can be used with Kimi
  by changing `model_id` as shown in the configuration cell.
- [`codec/codec_pythia.ipynb`](codec/codec_pythia.ipynb) runs the corresponding
  reference-model analysis with `EleutherAI/pythia-1.4b-deduped`. The Hugging
  Face model revision is pinned.

For each target question, another question from the same season is placed
before it as context. Delta is calculated as the target's mean token log
probability with context minus its mean token log probability without context.
The comparison is repeated with five randomly sampled context questions and
averaged before classification. A question is counted as a contamination signal
when its average delta is negative.

The CoDeC notebooks expect a Supabase table named `codec_questions` containing:

`question_id`, `season`, `episode`, `question_num`, `transcript`,
`include_for_codec`, and `review_status`.

Set the following environment variables before running the notebooks:

```text
SUPABASE_URL
SUPABASE_KEY
FIREWORKS_API_KEY
```

Install the CoDeC dependencies with:

```bash
pip install -r codec/requirements.txt
```

## Qwen3-VL pilot

The [`pilot`](pilot/) folder contains the analysis pipeline for a pilot using
the pinned `Qwen/Qwen3-VL-30B-A3B-Thinking` checkpoint. The pilot contains 30
questions, 10 prompt templates and 5 prespecified seeds, giving 1,500
generations.

[`pilot/pilot_prereg.ipynb`](pilot/pilot_prereg.ipynb) evaluates output
reliability, normalizes answers to each question's permitted range, compares
repetition strategies and prompt-template ensembles, and explores the
comparison between Qwen and the Norwegian viewer average.

The notebook expects a file named `pilot_results.csv` in the same working
directory. The pilot data are not currently distributed in this repository.

Install the pilot dependencies with:

```bash
pip install -r pilot/requirements.txt
```

## Data

The source transcripts, images and result tables are not distributed in this
repository.

## Reference

Zawalski, M., Boubdir, M., Balazy, K., Nushi, B., and Ribalta, P. (2026).
*Detecting Data Contamination in LLMs via In-Context Learning*. ICLR 2026.
https://openreview.net/forum?id=YlpaaYxx4t

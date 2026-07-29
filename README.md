# CoDeC for *Alle mot 1*

This repository applies Contamination Detection via Context (CoDeC) to
question transcripts from the Norwegian television game show *Alle mot 1*.
The implementation follows Zawalski et al. (2026), treating each season as a
dataset and each question transcript as one sample.

## Notebooks

- `codec_qwen.ipynb`: primary analysis using prompt logprobs returned by
  Fireworks. The recorded Qwen serverless endpoint may no longer be available.
  The same pipeline can be run with Kimi by changing the model identifier as
  shown in the configuration cell.
- `codec_pythia.ipynb`: reference-model analysis using
  `EleutherAI/pythia-1.4b-deduped` through Hugging Face Transformers. The model
  revision is pinned to the checkpoint used in the reported analysis.

## Method

For every target question, one different question from the same season is
sampled as context. The difference between the target's mean token log
probability with and without context is calculated over five reproducible
context draws. A question is counted as a contamination signal when its
five-draw mean difference is negative.

The notebooks use:

- one same-season context question;
- five context draws;
- base random seed 42; and
- target-token range `(10, -1)`.

## Data

The transcripts are not distributed in this repository. The notebooks expect a
Supabase table named `codec_questions` with these columns:

`question_id`, `season`, `episode`, `question_num`, `transcript`,
`include_for_codec`, and `review_status`.

Rows are included when `include_for_codec` is true and `review_status` is
`READY`.

## Configuration

Set these environment variables before running a notebook:

```text
SUPABASE_URL
SUPABASE_KEY
FIREWORKS_API_KEY
```

`FIREWORKS_API_KEY` is required only by the Qwen/Fireworks notebook. The Pythia
notebook instead requires a CUDA-enabled GPU.

Dependencies can be installed with:

```bash
pip install -r requirements.txt
```

## Reference

Michał Zawalski, Meriem Boubdir, Klaudia Bałazy, Besmira Nushi, and Pablo
Ribalta. *Detecting Data Contamination in LLMs via In-Context Learning*.
ICLR 2026. https://openreview.net/forum?id=YlpaaYxx4t

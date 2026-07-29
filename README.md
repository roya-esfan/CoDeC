# CoDeC for *Alle mot 1*

This repository applies Contamination Detection via Context (CoDeC) to
question transcripts from the Norwegian television game show *Alle mot 1*.
Following Zawalski et al. (2026), each season is treated as a dataset and each
question transcript as one sample.

## Notebooks

- `codec_qwen.ipynb` runs CoDeC with prompt log probabilities returned by
  Fireworks. The same pipeline can be used with Kimi by changing `model_id` as
  shown in the configuration cell.
- `codec_pythia.ipynb` runs the corresponding reference-model analysis with
  `EleutherAI/pythia-1.4b-deduped`. The Hugging Face model revision is pinned.

## Method

For each target question, another question from the same season is placed
before it as context. Delta is calculated as the target's mean token log
probability with context minus its mean token log probability without context.
The comparison is repeated with five randomly sampled context questions and
averaged before classification. A question is counted as a contamination
signal when its average delta is negative.

The analyses use:

- one same-season context question;
- five context draws;
- base random seed 42; and
- target-token range `(10, -1)`.

## Data

The transcripts and result tables are not distributed in this repository. The
analysis notebooks expect a Supabase table named `codec_questions` containing:

`question_id`, `season`, `episode`, `question_num`, `transcript`,
`include_for_codec`, and `review_status`.

## Configuration

Set the following environment variables before running the notebooks:

```text
SUPABASE_URL
SUPABASE_KEY
FIREWORKS_API_KEY
```

`FIREWORKS_API_KEY` is required only for Fireworks inference. Pythia instead
requires enough local compute to load and evaluate the model.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Reference

Zawalski, M., Boubdir, M., Balazy, K., Nushi, B., and Ribalta, P. (2026).
*Detecting Data Contamination in LLMs via In-Context Learning*. ICLR 2026.
https://openreview.net/forum?id=YlpaaYxx4t

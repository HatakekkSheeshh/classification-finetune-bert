# LSTM vs Transformer for 20 Newsgroups Text Classification

This project benchmarks a **BiLSTM + Attention** model against **BERT / DistilBERT fine-tuning** on the **20 Newsgroups** multi-class text classification task, then packages the best Transformer checkpoint into a **Streamlit inference app**.

## Highlights

- Built an end-to-end NLP comparison pipeline from preprocessing, training, evaluation, and error analysis to model serving.
- Compared a classical sequence model (**BiLSTM + Attention + GloVe**) with contextual Transformer encoders (**BERT**, **DistilBERT**).
- Evaluated multiple fine-tuning strategies: **frozen backbone**, **hybrid training**, and **full fine-tuning**.
- Deployed a lightweight **DistilBERT Streamlit demo** that loads a checkpoint from **Weights & Biases Artifacts**.

## Tech Stack

| Category | Tools |
|---|---|
| Modeling | PyTorch, Hugging Face Transformers |
| Data & Training | datasets, pandas, numpy, tqdm |
| Experiment Tracking | Weights & Biases |
| Visualization | matplotlib |
| App Demo | Streamlit |

## Repository Structure

```text
classification-finetune-bert/
|-- app.py                 # Streamlit app for DistilBERT inference
|-- bert_finetune.ipynb    # BERT / DistilBERT training and evaluation
|-- biALSTM.ipynb          # BiLSTM + Attention training and evaluation
|-- artifacts/             # Downloaded model artifact cache
|-- requirements.txt       # Python dependencies
`-- README.md
```

## Dataset

| Item | Description |
|---|---|
| Dataset | 20 Newsgroups |
| Task | 20-class news topic classification |
| Total samples | 18,846 documents |
| Train / Val / Test | 8,201 / 2,051 / 6,757 |

## Model Summary

### BiLSTM + Attention

- 300-dimension GloVe embedding
- TF-IDF weighted token representations
- 2-layer Bidirectional LSTM with hidden size 128
- Self-attention pooling and dropout regularization
- ~15.8M parameters

### BERT / DistilBERT

- `bert-base-uncased`: 12 layers, 768 hidden size, ~109.5M parameters
- `distilbert-base-uncased`: 6 layers, 768 hidden size, ~67.0M parameters
- Training strategies:
  - Freeze backbone, train classifier head only
  - Hybrid training: head warm-up then unfreeze full model
  - Full fine-tuning

## Results

| Model | Strategy | Val Accuracy | Test Accuracy | Training Time | Inference Latency | Parameters |
|---|---|---:|---:|---:|---:|---:|
| BERT | Full fine-tuning | 79.96% | 74.17% | 1,168s | 87.9ms | 109.5M |
| BERT | Hybrid | 79.86% | **74.77%** | 1,337s | 88.3ms | 109.5M |
| DistilBERT | Full fine-tuning | 79.33% | 74.06% | 644s | 44.4ms | 67.0M |
| DistilBERT | Hybrid | 78.89% | 73.88% | 735s | 44.1ms | 67.0M |
| BiLSTM + Attention | - | 69.72% | 62.90% | **85s** | - | **15.8M** |

## Key Takeaways

- Transformer fine-tuning outperformed BiLSTM + Attention by around **11.9% absolute test accuracy**.
- **DistilBERT** achieved a better speed/accuracy trade-off than BERT, with roughly **2x faster inference** and fewer parameters.
- BiLSTM trained much faster and was more compact, but static GloVe embeddings struggled with semantically overlapping categories such as politics and religion.
- Error analysis showed lower F1-score on short documents and classes with strong vocabulary overlap, especially `talk.religion.misc`, `talk.politics.misc`, and `alt.atheism`.

## Streamlit Demo

The app in `app.py` loads the fine-tuned **DistilBERT_Full** checkpoint from W&B Artifacts, preprocesses raw text, and returns **Top-K predicted classes** with probabilities.

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### W&B Artifact Note

The demo expects a W&B API key in Streamlit secrets:

```toml
WANDB_API_KEY = "your_wandb_api_key"
```

If the artifact is private, make sure your W&B account has permission to access:

```text
nguyenquochieujff7-ho-chi-minh-city-university-of-technology/bert-models/DistilBERT_Full:v0
```

Downloaded checkpoints are cached under `artifacts/DistilBERT_Full-v0/`.

## How to Reproduce

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run model experiments

- Open `biALSTM.ipynb` to train and evaluate the BiLSTM + Attention baseline.
- Open `bert_finetune.ipynb` to fine-tune BERT and DistilBERT with different training strategies.

3. Launch the inference app

```bash
streamlit run app.py
```

## Project Structure for Recruiters

- **Problem**: Fine-grained multi-class text classification on noisy user-generated newsgroup posts.
- **Approach**: Compare sequence modeling with static embeddings versus Transformer contextual representations.
- **Impact**: Quantified the accuracy-efficiency trade-off and delivered a working model demo for interactive inference.

## Future Improvements

- Add a public model checkpoint or Hugging Face Hub release to make the demo easier to run without private W&B access.
- Export core notebook logic into reusable Python modules for cleaner training pipelines.
- Add experiment config files and automated metric logging for easier reproducibility.
- Extend evaluation with macro-F1, per-class confusion analysis, and calibration metrics.

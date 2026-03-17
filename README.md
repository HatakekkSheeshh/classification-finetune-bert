# LSTM vs Transformer Comparison for 20 Newsgroups Classification

This project compares the performance of BiLSTM with Attention and Transformer models (BERT/DistilBERT) on the 20 Newsgroups text classification task.

## Project Overview

| Aspect | Description |
|--------|-------------|
| **Dataset** | 20 Newsgroups (18,846 documents, 20 classes) |
| **Task** | Multi-class text classification |
| **Models** | BiLSTM + Attention vs BERT/DistilBERT |
| **Framework** | PyTorch + HuggingFace Transformers |

## Project Structure

```
lstm_vs_transformer/
├── bert_finetune.ipynb      # BERT/DistilBERT training & evaluation
├── biALSTM.ipynb            # BiLSTM with Attention training & evaluation
├── train_lstm.py            # Standalone LSTM training script
├── requirements.txt         # Python dependencies
├── note.txt                 # BiLSTM architecture explained
├── lstm_1.png              # BiLSTM classification report
├── lstm_2.png              # BiLSTM confusion matrix
└── short_docs.png           # Short documents analysis
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Notebooks

- **`bert_finetune.ipynb`**: Trains BERT and DistilBERT with 3 strategies (freeze backbone, hybrid, full fine-tuning)
- **`biALSTM.ipynb`**: Trains BiLSTM with GloVe embeddings, TF-IDF weighting, and attention

### 3. Key Results

| Model | Strategy | Val Acc | Test Acc | Training Time | Inference | Parameters |
|-------|----------|---------|----------|---------------|-----------|------------|
| **BERT** | Full Fine-tune | 79.96% | **74.17%** | 1,168s | 87.9ms | 109.5M |
| **BERT** | Hybrid | 79.86% | 74.77% | 1,337s | 88.3ms | 109.5M |
| **DistilBERT** | Full Fine-tune | 79.33% | 74.06% | 644s | 44.4ms | 67.0M |
| **DistilBERT** | Hybrid | 78.89% | 73.88% | 735s | 44.1ms | 67.0M |
| **BiLSTM + Attention** | - | 69.72% | 62.90% | **85s** | - | **15.8M** |

### 4. Key Findings

- **Best Transformer**: DistilBERT achieves 74.06% test accuracy with full fine-tuning
- **Accuracy Gap**: BERT/DistilBERT outperforms BiLSTM by ~11% (74.1% vs 62.9%)
- **Efficiency**: BiLSTM trains 7-16x faster than transformers (85s vs 644-1,337s)
- **Inference Speed**: DistilBERT is 2x faster than BERT (44ms vs 88ms)
- **Size**: BiLSTM is 4-7x smaller than transformer models (15.8M vs 67-109M params)
- **Architecture Impact**: Contextualized representations (transformers) are essential for fine-grained classification

## Model Architectures

### BiLSTM with Attention

```
Input (token IDs)
    ↓
Embedding Layer (300d GloVe) + TF-IDF weighting
    ↓
2-layer Bidirectional LSTM (hidden=128)
    ↓
Self-Attention Layer
    ↓
Dropout (0.3)
    ↓
Linear (256 → 20)
    ↓
Output (20 classes)
```

**Parameters**: 15,840,981

### Transformer (BERT/DistilBERT)

- **BERT-base-uncased**: 12 layers, 768 hidden, 12 heads → 109.5M parameters
- **DistilBERT**: 6 layers, 768 hidden, 6 heads → 66.0M parameters
- **Training Strategies**: Freeze backbone, Hybrid (head → full), Full fine-tuning

## Dataset Details

| Split | Samples |
|-------|---------|
| Train | 8,201 |
| Validation | 2,051 |
| Test | 6,757 |

**Classes**: 20 newsgroups covering:
- `comp.*` (6 classes): hardware, graphics, OS, mac, windows, forsale
- `rec.*` (4 classes): autos, motorcycles, baseball, hockey
- `sci.*` (4 classes): cryptography, electronics, med, space
- `talk.*` (4 classes): politics (guns, mideast, misc), religion
- `soc.*` (1 class): religion.christian
- `misc.*` (1 class): forsale
- `alt.*` (1 class): atheism

## Per-Class Performance (BiLSTM)

### Best Performers
| Class | F1-Score |
|-------|----------|
| rec.sport.hockey | 0.89 |
| rec.sport.baseball | 0.78 |
| rec.motorcycles | 0.72 |

### Worst Performers
| Class | F1-Score |
|-------|----------|
| talk.religion.misc | 0.19 |
| talk.politics.misc | 0.26 |
| alt.atheism | 0.29 |

## Challenges Identified

1. **Class Overlap**: Religion/politics newsgroups share 80-90% vocabulary
2. **Short Documents**: comp.graphics, sci.electronics have 60%+ docs under 100 words
3. **Cross-posting**: Same discussions appear in multiple newsgroups
4. **Static Embeddings**: BiLSTM uses fixed GloVe vectors without context

## Dependencies

```
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0
tokenizers>=0.15.0
accelerate>=0.25.0
wandb>=0.16.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
psutil>=5.9.0
```

## Conclusion

This project demonstrates that:
1. Transformer models significantly outperform BiLSTM on fine-grained text classification
2. The ~11% accuracy gap is mainly due to contextualized representations
3. Attention mechanisms on static embeddings cannot fully compensate for lack of context
4. For this dataset, architectural choices matter more than optimization tweaks

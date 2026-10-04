# Zero-Shot Text Classification with Embedding Models

This project explores zero-shot text classification using sentence-embedding models. Instead of training a classifier, we embed the input text and each candidate label into the same vector space and pick the label whose vector is closest (cosine similarity) to the text vector.

## Models Compared

| Model | Developer | Params | Layers | Embedding dim | Max tokens | Pooling | Prefix needed |
|---|---|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Sentence-Transformers (SBERT) | ~22M | 6 | 384 | 256 | Mean | No |
| `intfloat/e5-small-v2` | Microsoft | ~33M | 12 | 384 | 512 | Mean | Yes (`query: `) |
| `BAAI/bge-small-en-v1.5` | BAAI | ~33M | 12 | 384 | 512 | CLS | Optional |
| `BAAI/bge-m3` | BAAI | ~568M | 24 | 1024 | 8192 | CLS | No |

> Figures are approximate; check each model card on Hugging Face for exact details.

---

## How These Models Work

All four are **encoder-only Transformer models** (BERT family) that turn text into a fixed-length vector.

```
text -> tokenizer -> Transformer encoder -> token vectors -> pooling -> normalize -> sentence vector
```

### 1. Tokenization
Text is split into subword tokens (WordPiece for the BERT-based models, SentencePiece for bge-m3) and mapped to integer IDs.

### 2. Transformer encoder
Each token receives a learned token embedding plus a position embedding, then passes through a stack of identical layers. Each layer contains:

- **Self-attention**: every token attends to every other token in both directions, so each token's vector becomes a context-aware mixture of the whole sentence. This is how "bank" gets different representations in "river bank" and "bank loan".
- **Feed-forward network**: a small MLP applied to each token independently.
- **Residual connections and layer normalization** around both.

The output is **one vector per token**. Attention alone does not produce a single sentence vector.

### 3. Pooling
Token vectors are collapsed into one sentence vector:

- **Mean pooling** (all-MiniLM, e5): average all token vectors.
- **CLS pooling** (bge-small, bge-m3): use the vector of the special `[CLS]` token, which the model is trained to use as a summary of the input.

### 4. Normalization
The vector is scaled to unit length so cosine similarity reduces to a dot product.

### 5. Contrastive training (what makes the vectors meaningful)
A raw pre-trained BERT produces poor sentence embeddings. These models are fine-tuned with a **contrastive objective (InfoNCE)** on large collections of related text pairs (question/answer, title/body, query/passage, paraphrases). For each batch, the loss pulls matching pairs together and pushes them away from the other items in the batch (in-batch negatives). Over billions of pairs, semantically similar texts end up close together in vector space.

This is why zero-shot classification works: a label like "sports" lands near sports-related texts without any classification-specific training.

---

## How the Models Differ

### all-MiniLM-L6-v2
- **Base**: MiniLM, a compressed Transformer created via **knowledge distillation** (a small student mimics the self-attention behavior of a larger teacher).
- **Training**: contrastive fine-tuning on roughly 1 billion sentence pairs.
- **Strengths**: very fast and light, a dependable baseline.
- **Limitations**: only 6 layers, and input is truncated at 256 tokens, so long texts lose content. Older recipe than the other three.

### e5-small-v2
- **Name**: EmbEddings from bidirEctional Encoder rEpresentations.
- **Training**: **weakly supervised contrastive pre-training** on a large web-scraped text-pair corpus, followed by fine-tuning on labeled datasets.
- **Key detail**: it was trained with role prefixes. Use `"query: "` on both the text and the labels for classification or similarity tasks. Omitting the prefix noticeably degrades quality.
- **Strengths**: good general quality for its size.

### bge-small-en-v1.5
- **Training**: **RetroMAE** pre-training (a reconstruction objective that forces the `[CLS]` vector to hold sentence-level information), then contrastive learning with hard negatives and instruction-style fine-tuning.
- **Key detail**: for retrieval, queries work best with the prefix `"Represent this sentence for searching relevant passages: "`. For short-text similarity and classification, no prefix is generally needed.
- **Strengths**: usually the strongest of the three small models on the MTEB benchmark. v1.5 improves the similarity score distribution over v1.
- **Limitations**: English only.

### bge-m3
The largest and most versatile model here. "M3" stands for **Multi-Functionality, Multi-Linguality, Multi-Granularity**.

- **Base**: XLM-RoBERTa-large (24 layers, 1024 hidden), extended to support inputs up to **8192 tokens**.
- **Multi-linguality**: supports 100+ languages, with cross-lingual matching (a label in English can match text in another language).
- **Multi-granularity**: handles inputs from short sentences to long documents.
- **Multi-functionality**: one model produces three kinds of representations:
  - **Dense**: a single 1024-d vector per text (what we use here).
  - **Sparse (lexical)**: learned per-token weights, similar to BM25-style term matching.
  - **Multi-vector (ColBERT-style)**: one vector per token with late interaction scoring.
- **Training**: RetroMAE-style pre-training adapted for long context, large-scale unsupervised contrastive pre-training, then fine-tuning. It uses **self-knowledge distillation**, where the combined score of the dense, sparse, and multi-vector outputs acts as the teacher signal for each individual output.
- **Strengths**: multilingual data, long texts, and highest accuracy potential.
- **Limitations**: roughly 17x the parameters of the small models, so it is slower and uses more memory.

### Summary of differences

| Aspect | all-MiniLM-L6-v2 | e5-small-v2 | bge-small-en-v1.5 | bge-m3 |
|---|---|---|---|---|
| Language | English | English | English | 100+ languages |
| Size | Smallest | Small | Small | Large |
| Context length | 256 | 512 | 512 | 8192 |
| Key training idea | Distillation + 1B pairs | Weakly supervised contrastive | RetroMAE + hard negatives | Self-knowledge distillation, M3 |
| Output types | Dense | Dense | Dense | Dense, sparse, multi-vector |
| Speed | Fastest | Fast | Fast | Slowest |

---


## Tips for Better Zero-Shot Results

- **Use label descriptions, not bare words.** `"A news article about sports"` generally beats `"sports"`. Try several templates and compare.
- **Apply the e5 prefix** (`query: `) to both texts and labels.
- **Watch truncation.** all-MiniLM stops at 256 tokens; use bge-m3 for long documents.
- **Use bge-m3 for non-English data**, or when labels and texts are in different languages.
- **Embedding similarity is not a trained classifier.** For subtle or overlapping labels, NLI-based zero-shot models (e.g. `facebook/bart-large-mnli`, `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`) can be more accurate, at higher compute cost.
- **Cache label embeddings.** They are computed once and reused for every input text.

---

## References

- Reimers & Gurevych, *Sentence-BERT* (2019)
- Wang et al., *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers* (2020)
- Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5, 2022)
- Xiao et al., *C-Pack: Packaged Resources To Advance General Chinese Embedding* (BGE, 2023)
- Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation* (2024)
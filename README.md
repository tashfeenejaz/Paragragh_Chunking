# Paragraph Chunking for RAG on WHO Medical Guidelines

This project evaluates **paragraph chunking** on 10 WHO / Ministry of Health medical guideline PDFs, using the raw text extracted from the PDFs with **no cleaning**. It compares 5 chunking variants across 3 retrieval methods (BM25, dense, hybrid) on 40 questions and measures Precision@5, Recall@5, Hit@5 and MRR.

## Pipeline

1. **Load PDFs** with PyMuPDF (`page.get_text("text", sort=True)`), page by page. `sort=True` is needed because the default extraction contains no blank lines to split paragraphs on.
2. **Build paragraph chunks** by splitting on blank lines, then apply optional token limits (tokens are counted with the `BAAI/bge-small-en-v1.5` tokenizer):
   - **Split (`max_tokens`)**: paragraphs longer than the limit are split at sentence boundaries and packed up to the limit. A single oversized sentence is cut into token windows.
   - **Merge (`min_tokens`)**: chunks shorter than the minimum are merged with the next paragraph of the same page, as long as the merged chunk stays under `max_tokens`.
   - Chunks never cross page boundaries, so page metadata stays correct.
3. **Retrieve** with three retrievers:
   - **BM25** (`rank_bm25`) on lowercased word tokens.
   - **Dense**: `all-MiniLM-L6-v2` (max 256 tokens), cosine similarity.
   - **Hybrid**: Reciprocal Rank Fusion of the BM25 and dense rankings (`k=60`, depth 50).
4. **Evaluate** on 40 questions (`questions.json`, 4 per document). A chunk is relevant if it comes from the correct document and contains at least one of the answer phrases (case-insensitive, whitespace-normalized). This definition does not depend on chunk ids, so every variant is evaluated on the same questions.

## Evaluation set

- 40 questions (4 per document): a specific fact, a list, a conditional question and a definition for each document.
- The questions were drafted with an LLM from the extracted text. Each question has an `answer_phrase` plus up to two alternative phrases, all copied verbatim from the extracted text, and `must_have` keywords used only for auditing.
- Every phrase was verified programmatically: it exists in the text of the correct document, and every question keeps at least one relevant chunk in every chunking variant.

## Chunking variants

| Variant | max_tokens | min_tokens | Chunks | Median tokens | < 30 tokens | > 256 tokens |
|---|---|---|---|---|---|---|
| raw | none | none | 13731 | 31 | 6667 | 531 |
| split256 | 256 | none | 14630 | 36 | 6733 | 2 |
| split512 | 512 | none | 13953 | 33 | 6680 | 633 |
| split256+merge30 | 256 | 30 | 10009 | 71 | 1248 | 2 |
| split512+merge30 | 512 | 30 | 9293 | 68 | 1164 | 647 |

Splitting removes oversized chunks but does not help with short ones. Merging cuts the chunk count by about 30% and roughly doubles the median length. Of the 1248 chunks that are still under 30 tokens in `split256+merge30`, 1120 (about 90%) are the last chunk of their page (page numbers, running headers and footers), which cannot be merged forward.

## Results

Metrics are averaged over all 40 questions. MRR is computed over the top 10 results.

| Variant | Model | Chunks | P@5 | R@5 | Hit@5 | MRR |
|---|---|---|---|---|---|---|
| raw | BM25 | 13731 | 0.155 | 0.535 | 0.550 | 0.452 |
| raw | Dense | 13731 | 0.180 | 0.601 | 0.650 | 0.507 |
| raw | Hybrid | 13731 | 0.185 | 0.668 | 0.725 | 0.565 |
| split256 | BM25 | 14630 | 0.150 | 0.497 | 0.525 | 0.443 |
| split256 | Dense | 14630 | 0.170 | 0.570 | 0.625 | 0.501 |
| split256 | Hybrid | 14630 | 0.195 | 0.680 | 0.725 | 0.556 |
| split512 | BM25 | 13953 | 0.160 | 0.560 | 0.575 | 0.452 |
| split512 | Dense | 13953 | 0.180 | 0.601 | 0.650 | 0.506 |
| split512 | Hybrid | 13953 | 0.195 | 0.686 | 0.725 | 0.562 |
| split256+merge30 | BM25 | 10009 | 0.185 | 0.627 | 0.650 | 0.513 |
| split256+merge30 | Dense | 10009 | 0.190 | 0.645 | 0.725 | 0.569 |
| split256+merge30 | Hybrid | 10009 | **0.210** | **0.728** | **0.775** | 0.635 |
| split512+merge30 | BM25 | 9293 | 0.175 | 0.622 | 0.650 | 0.513 |
| split512+merge30 | Dense | 9293 | 0.200 | 0.676 | 0.750 | 0.574 |
| split512+merge30 | Hybrid | 9293 | 0.205 | **0.728** | **0.775** | **0.642** |

Average per model across the 5 variants:

| Model | P@5 | R@5 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25 | 0.165 | 0.568 | 0.590 | 0.475 |
| Dense (MiniLM) | 0.184 | 0.619 | 0.680 | 0.531 |
| Hybrid (RRF) | 0.198 | 0.698 | 0.745 | 0.592 |

## Final verdict

- **Best overall: Hybrid + split256+merge30.** MRR 0.635, Hit@5 0.775, R@5 0.728, P@5 0.210, with about 30% fewer chunks than split256.
- **Hybrid + split512+merge30 is practically tied** (MRR 0.642, same Hit@5 and R@5). The difference is within noise, and 647 of its chunks exceed the 256-token limit of MiniLM, so `split256+merge30` is preferred.
- **Hybrid is the best retriever for every chunking variant** (MRR 0.556 to 0.642), ahead of Dense (0.501 to 0.574) and BM25 (0.443 to 0.513).
- **Merging short paragraphs (min 30 tokens) helps all three retrievers.** For Hybrid, MRR rises from about 0.56 to 0.64 and Hit@5 from 0.725 to 0.775. Splitting alone (256 or 512) changes little compared with raw.

## Limitations

- Only 40 questions: one question equals 0.025 in Hit@5, so differences below about 0.03 to 0.05 (for example `split256+merge30` vs `split512+merge30`) are within noise.
- The questions were drafted with an LLM and verified programmatically; relevance is decided by exact phrase matching, so a chunk that answers the question in different words is counted as not relevant. This affects all variants alike but lowers absolute numbers.
- Recall@5 is not perfectly comparable across variants, because the number of relevant chunks changes with chunk size. MRR and Hit@5 are the fairer metrics here.
- Some answers are repeated in several places (summary, recommendation, evidence tables), so those questions have several relevant chunks and Recall@5 cannot reach 1.0 for them.
- After merging, 1248 of 10009 chunks in `split256+merge30` are still under 30 tokens; about 90% of them are the last chunk of a page and cannot be merged forward. The text is deliberately used without cleaning.
- `diabetes.pdf` has a broken font encoding, so part of its text is garbled and all methods struggle on it.
- The corpus is imbalanced (`malaria.pdf` has 494 pages, `diabetes.pdf` has 12), and one dense model (MiniLM) was tested, so conclusions about "dense" retrieval are specific to that model.

## Project structure

```
Paragraph_Chunking.ipynb   # full pipeline and evaluation
questions.json             # 40 evaluation questions with answer phrases
data/                      # the 10 WHO PDFs (not included in the repo)
raw_text/                  # extracted text per PDF (generated)
chunks_out/                # chunks per variant as .txt and .csv (generated)
emb_minilm_*.npy           # cached embeddings (generated)
overall_results.csv        # summary table of all 15 combinations
```

## Setup

```bash
pip install pymupdf rank_bm25 sentence-transformers nltk pandas numpy
```

1. Put the WHO guideline PDFs in a `data/` folder.
2. Make sure `questions.json` is in the project root.
3. Run `Paragraph_Chunking.ipynb` from top to bottom. Embeddings are cached on disk after the first run.

## Tech stack

PyMuPDF, rank_bm25, sentence-transformers (`all-MiniLM-L6-v2`), Hugging Face tokenizer (`BAAI/bge-small-en-v1.5`, used to count tokens), pandas, NumPy.

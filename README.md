# Paragraph Chunking for RAG on WHO Medical Guidelines

This project evaluates **paragraph chunking** on 10 WHO medical guideline PDFs, using the raw text extracted from the PDFs with **no cleaning**. It compares 5 chunking variants across 3 retrieval methods and measures Precision@5, Recall@5, Hit@5 and MRR.

## Pipeline

1. **Load PDFs** with PyMuPDF (`page.get_text("text", sort=True)`), page by page. `sort=True` is needed because the default extraction contains no blank lines to split paragraphs on.
2. **Build paragraph chunks** by splitting on blank lines, then apply optional token limits:
   - **Split (`max_tokens`)**: paragraphs longer than the limit are split at sentence boundaries and packed up to the limit. A single oversized sentence is cut into token windows.
   - **Merge (`min_tokens`)**: chunks shorter than the minimum are merged with the next paragraph of the same page, as long as the merged chunk stays under `max_tokens`.
   - Chunks never cross page boundaries, so page metadata stays correct.
3. **Retrieve** with BM25, dense embeddings (`all-MiniLM-L6-v2`, cosine similarity) and a hybrid of both (Reciprocal Rank Fusion, `k=60`, depth 50).
4. **Evaluate** on 16 hand-written questions (`questions.json`). A chunk is relevant if it comes from the correct document and contains the answer phrase (case-insensitive, whitespace-normalized). This definition does not depend on chunk ids, so every variant is evaluated on the same questions.

## Chunking variants

| Variant | max_tokens | min_tokens | Chunks | Median tokens | < 30 tokens |
|---|---|---|---|---|---|
| raw | none | none | 13731 | 31 | 6667 |
| split256 | 256 | none | 14630 | 36 | 6733 |
| split512 | 512 | none | 13953 | 33 | 6680 |
| split256+merge30 | 256 | 30 | 10009 | 71 | 1248 |
| split512+merge30 | 512 | 30 | 9293 | 68 | 1164 |

Splitting removes oversized chunks but does not help with short ones. Merging cuts the chunk count by about 30% and roughly doubles the median length.

## Results

Metrics are averaged over all 16 queries. MRR is computed over the top 10 results.

| Variant | Model | Chunks | P@5 | R@5 | Hit@5 | MRR |
|---|---|---|---|---|---|---|
| raw | BM25 | 13731 | 0.213 | 0.812 | 0.812 | 0.605 |
| raw | Dense | 13731 | 0.200 | 0.740 | 0.812 | 0.502 |
| raw | Hybrid | 13731 | 0.200 | 0.750 | 0.750 | 0.579 |
| split256 | BM25 | 14630 | 0.200 | 0.750 | 0.750 | 0.610 |
| split256 | Dense | 14630 | 0.188 | 0.719 | 0.750 | 0.490 |
| split256 | Hybrid | 14630 | 0.213 | 0.771 | 0.812 | 0.637 |
| split256+merge30 | BM25 | 10009 | 0.188 | 0.719 | 0.750 | 0.575 |
| split256+merge30 | Dense | 10009 | 0.175 | 0.667 | 0.750 | 0.516 |
| split256+merge30 | Hybrid | 10009 | 0.200 | 0.729 | 0.812 | **0.648** |
| split512 | BM25 | 13953 | 0.213 | 0.812 | 0.812 | 0.595 |
| split512 | Dense | 13953 | 0.200 | 0.740 | 0.812 | 0.502 |
| split512 | Hybrid | 13953 | 0.200 | 0.750 | 0.750 | 0.593 |
| split512+merge30 | BM25 | 9293 | 0.200 | 0.781 | 0.812 | 0.583 |
| split512+merge30 | Dense | 9293 | 0.175 | 0.667 | 0.750 | 0.542 |
| split512+merge30 | Hybrid | 9293 | 0.200 | 0.740 | 0.812 | 0.637 |

Average per model across the 5 variants:

| Model | R@5 | P@5 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25 | 0.775 | 0.203 | 0.787 | 0.594 |
| Dense (MiniLM) | 0.707 | 0.188 | 0.775 | 0.510 |
| Hybrid (RRF) | 0.748 | 0.203 | 0.787 | 0.619 |

## Final verdict

- **Best overall: Hybrid + split256+merge30.** Best MRR (0.648), Hit@5 of 0.812, and about 30% fewer chunks than split256.
- **If recall is the priority: Hybrid + split256** (R@5 0.771, P@5 0.213, MRR 0.637).
- **Dense-only retrieval is the weakest** (MRR 0.49 to 0.54). It finds the right chunk but ranks it lower.
- **BM25 is strong on exact medical terms.** The best R@5 and P@5 come from raw + BM25 (0.812 / 0.213).
- **Merging short paragraphs improves ranking for Dense and Hybrid** but slightly lowers BM25 recall.

## Limitations

- Only 16 questions: one query equals 0.0625 in Hit@5, so differences of one or two queries are not statistically meaningful.
- Recall@5 is not perfectly comparable across variants, because the number of relevant chunks changes with chunk size. MRR and Hit@5 are the fairer metrics here.
- Precision@5 is capped near 0.2, because most questions have only about one relevant chunk.
- `diabetes.pdf` has a broken font encoding, so its text is garbled and all methods struggle on it.
- The corpus is imbalanced (`malaria.pdf` has 494 pages, `diabetes.pdf` has 12).
- After merging (min 30 tokens), 1248 of 10009 chunks in `split256+merge30` are still under 30 tokens. 1120 of them (about 90%) are the last chunk of a page (page numbers, running headers/footers) and cannot be merged forward; the rest are blocked by the max-token limit. This is expected, since the text is used without cleaning.

## Project structure

```
Paragraph_Chunking.ipynb   # full pipeline and evaluation
questions.json             # 16 evaluation questions with answer phrases
data/                      # the 10 WHO PDFs (not included in the repo)
raw_text/                  # extracted text per PDF (generated)
chunks_out/                # chunks per variant as .txt and .csv (generated)
emb_minilm_*.npy           # cached embeddings (generated)
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

# 🤖 HR Manual RAG Chatbot

## Overview
This project implements a Retrieval‑Augmented Generation (RAG) chatbot that answers questions about a (fictional) internal HR manual. The system combines semantic retrieval (FAISS + dense embeddings) with an LLM (via OpenRouter) constrained to the retrieved context. Goal: grounded answers, minimal hallucination, lightweight evaluation.

## Data
- Source: Single textual HR guide (plain .txt) with whimsical / pseudo‑procedural content.  
- Domain: Onboarding, schedules, leave requests, internal objects/processes (intentionally absurd for testing).  
- Scale: 1 document → cleaned → split into overlapping chunks.  
- Language: Spanish (queries + source).  

## Pipeline Summary
1. Load raw text.  
2. Clean & normalize (lowercase, strip URLs/emails/dates/symbols, accent removal).  
3. Chunk (character window: 500, overlap: 50) preserving partial continuity.  
4. Embed chunks with `all-MiniLM-L6-v2` (HuggingFace).  
5. Build FAISS index (L2 similarity).  
6. On query: retrieve top‑k (default k = 3).  
7. Construct context block.  
8. Generate answer using LLM (`deepseek/deepseek-r1:free` via OpenRouter) with strict system prompt (“answer ONLY from context”).  
9. Lightweight keyword coverage evaluation on a small set.  

## Preprocessing
Cleaning steps: lowercase, URL removal, email removal, date pattern removal, non‑alphanumeric filtering, accent normalization, whitespace trim.  
Rationale: Simplify noise; model (MiniLM) robust, so no aggressive stopword removal to avoid semantic loss.

## Chunking
- Method: Character sliding window (`chunk_size=500`, `overlap=50`).  
- Trade‑off: Enough span for semantic units; moderate redundancy improves recall at small k.  
- Result: Balanced number of vectors (manageable memory + fast retrieval).

## Embeddings & Index
- Model: `all-MiniLM-L6-v2` (384‑dimensional, fast inference).  
- Vector store: FAISS (in‑memory) – suitable for small corpus, no need for ANN service.  
- Index size: = number of chunks (≈ N after split; exact count depends on original length).

## Retrieval & Generation
- Similarity: Inner product (through FAISS L2 adaptation) over dense embeddings.  
- Context assembly: Simple concatenation with double newline separator (no reranking).  
- LLM Prompting:  
  - System: enforce grounding (“If answer not present, state it is absent”).  
  - User: Injects context block + user question.  
- Failure handling: If context is too generic → model may hedge; prompt guards reduce speculative drift.

## Evaluation
Light, heuristic approach (given fictional domain):

| Query (summary)                       | Keywords (count) | Coverage (sample run)* | Notes                                |
|--------------------------------------|------------------|------------------------|--------------------------------------|
| Vacation request procedure           | 3                | 0.67                   | 2/3 anchors surfaced                 |
| Official working hours               | 4                | 0.50                   | Partial temporal terms recovered     |
| First day process / onboarding       | 3                | 0.33                   | Missed 2 whimsical tokens            |

*Illustrative values; dependent on current manual text & chunk boundaries.

Metric:  
- Keyword Coverage = matched_keywords / total_keywords (case‑insensitive).  
Purpose: Sanity check retrieval grounding vs. hallucination—not a semantic correctness guarantee.

## Results & Insights
- Retrieval Precision: Adequate with k=3; raising k to 5 marginally improves recall but risks diluting prompt budget.  
- Grounding: System prompt + limited context keeps hallucinations low; failures usually manifest as “no aparece”.  
- Embedding Choice: MiniLM sufficient; larger (e.g. `bge-large`) would be overkill for single‑document scale.  
- Bottleneck: Not retrieval—latency dominated by external LLM call.

## Conclusion
This project provides a concise demonstration of grounded RAG over a single (deliberately absurd) Spanish HR manual.
It illustrates cleaning, chunking, MiniLM embeddings, FAISS retrieval, and simple coverage-based evaluation in a compact pipeline.
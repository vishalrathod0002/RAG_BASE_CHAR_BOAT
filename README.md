# Payment Docs Assistant — How Answering Works

## LLM

Only **one** LLM is used (Groq):
- `llama-3.3-70b-versatile` (default), or
- `llama-3.1-8b-instant`

Selectable from the sidebar dropdown. The same model handles both answer modes below — there is no second model.

## Fixed Pipeline (always runs, regardless of mode)

```
query
  │
  ▼
retrieve top_k chunks from FAISS   ← always runs, even for unrelated questions
  │
  ▼
build prompt: instructions + context + question
  │
  ▼
send to LLM (single call)
  │
  ▼
answer
```

Retrieval is **never skipped**. Even if the question has nothing to do with the
indexed documents, the app still fetches the top-k nearest chunks and inserts
them into the prompt.

## Answer Modes (sidebar toggle)

The mode does **not** change the pipeline above — it only changes the prompt
wording sent to the LLM. The retrieved context is always included; what
differs is what the LLM is told to *do* with it.

| Mode | Prompt instruction | Behavior |
|---|---|---|
| 📄 Docs only | "Use only the context. If it's not there, say so." | Refuses to answer if context doesn't cover the question. |
| 🌐 Docs + general knowledge | "Use context if relevant; otherwise answer from your own knowledge and prefix with `[General knowledge]`." | Falls back to the model's training knowledge when context is weak/irrelevant. |

## Who decides if context is "relevant"?

Currently: **the LLM decides**, based on the prompt wording. There is no code
that checks similarity scores or routes the query before calling the LLM.
This is simple, but not deterministic — a borderline match is judged by the
model each time, not enforced by your code.

## Optional Upgrades (not yet implemented)

### 1. Similarity-threshold check (cheap, deterministic)
FAISS returns a distance/score per retrieved chunk. If the best score is
worse than a cutoff, skip inserting context entirely and call the LLM
directly (general-knowledge-only prompt).

```
query → retrieve + scores → best score good enough?
                                 ├─ yes → RAG prompt (context + question)
                                 └─ no  → direct LLM prompt (question only)
```
Pros: no extra LLM call, fully deterministic.
Cons: needs a tuned threshold; FAISS score semantics vary by distance metric.

### 2. Router LLM call (more accurate, costs one extra call)
A small first LLM call classifies the query as "needs docs" vs "general
knowledge" before deciding whether to retrieve at all.

```
query → router LLM: "doc-question" or "general" ?
           ├─ doc-question → retrieve → RAG prompt
           └─ general       → skip retrieval → direct LLM prompt
```
Pros: most accurate routing, can skip retrieval cost for clearly unrelated
questions.
Cons: doubles LLM calls per question (extra latency + cost).

## Current File Reference

- `ask_llm(model_name, results, query, allow_general_knowledge)` — builds the
  mode-specific prompt and calls the single LLM.
- `retrieve(vectorstore, query, k)` — always runs before `ask_llm`, regardless
  of mode.
- Sidebar radio `answer_mode` sets `allow_general_knowledge` (`True`/`False`),
  which is passed into `ask_llm`.

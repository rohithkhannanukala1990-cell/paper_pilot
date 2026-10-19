# PaperPilot

A RAG research-paper assistant that summarizes papers, answers questions with numbered citations, and compares papers.

## Security notes

- **Prompt injection via PDF text.** Extracted paper text is treated as untrusted data. System prompts instruct the model to ignore instructions found inside passages, but a determined adversary can still try to steer answers through malicious PDF content. Prefer papers from sources you trust, and treat model output as advisory—not authoritative.
- **Secrets handling.** API keys and other secrets must live in environment variables (see `.env.example`). Never hardcode keys in source, notebooks, or commits. `.env` is gitignored; keep it out of screenshots and shared logs. Logs intentionally omit prompts and document text.
- **Cost abuse.** Each question, summary, and comparison calls a paid LLM (and summaries run several facet calls in parallel). Upload limits (`MAX_UPLOAD_MB`, `MAX_PAGES`) reduce runaway ingest cost; still avoid exposing an unauthenticated public endpoint without rate limits, auth, and spend caps on the provider account.

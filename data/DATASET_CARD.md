# Dataset card

- Name: Stream-RAG synthetic workshop and travel corpus, version 1.
- Source: authored for this implementation, not supplied by organizers or scraped.
- Contents: 12 venue documents (three cities × four topics), one accessibility policy, two travel reimbursement policies and two booking approval policies.
- Cedar Hall, Maple Hall and Willow Hall are fictional. Prices, capacities and policies are invented. No personal records or real customer conversations are included.
- Purpose: demonstrate multi-intent retrieval, scopes and evidence reuse. This is not a representative training dataset or model benchmark.
- Evaluation: nine separate cases in `scripts/evaluate.py`. Queries/expected source IDs are not indexed.
- Limits: English only, small vocabulary, few entities and no realistic ASR noise. The cases are known to the author, not a blind held-out set.
- Uploads replace the corpus and reset conversation state. Main-workspace uploads are not written to disk or sent to external APIs.

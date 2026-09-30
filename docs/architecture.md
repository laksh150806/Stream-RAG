# Architecture

## Separate modes

`app.py` is the corpus-isolated workspace. `pages/2_Live_News.py` exposes the imported BBC experiment. The main workspace never ingests radio output automatically.

## Event path

1. Validate IDs and scalar metadata, then split into exact source chunks.
2. Accept cumulative text, turn ID, monotonic timestamp and a final flag.
3. Hold incomplete clauses, suppress recognized presentation-only requests, and reuse unchanged final retrieval.
4. Split English conjunction/question boundaries into sub-queries and run independent jobs in a bounded thread pool.
5. Score eligible chunks using BM25. Topic hints and recognized metadata values restrict candidates. No dense embeddings are used in this mode.
6. Record each intent's evidence and metadata dependencies. Supported late corrections update dependent intents; unaffected claims remain. Revised transcripts retract obsolete current-turn intents.
7. Return exact excerpts. Look up every source ID and check every quote against its chunk. Missing evidence produces uncertainty.
8. Emit input, decision, retrieval and completion events. Changed evidence creates an answer version.

## State

One engine lives in each Streamlit session; it is not globally cached. The CLI creates a fresh engine. Corpus/conversation state remains in memory unless the user downloads an export.

## Scope semantics and limits

Metadata keys other than topic/source/date/author define named scopes. Explicit `key: value` recognizes unknown values. Natural-language scope values must already exist in the corpus. Missing metadata keys mean a chunk is global for that scope.

Numeric capacity comparisons, implicit unseen locations, complex negation and semantic entailment are not implemented. Exact excerpts establish source provenance, not relevance or completeness. The three failed development paraphrases document these limits.

## News path

FFmpeg → Groq Whisper → MiniLM → Chroma → Groq Llama. A worker owns its stop event and bounded queue and never accesses Streamlit. The UI drains messages and maintains a heartbeat. Persistent news data/external APIs are outside the Theme 4 evaluation path.

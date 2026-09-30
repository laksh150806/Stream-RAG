# Development evaluation — not official gate results

Run `python scripts/evaluate.py`. Expected and actual IDs are in `development-results.json`. These ten authored synthetic cases cannot establish general accuracy.

| Configuration | Exact final source-set checks |
|---|---:|
| Streaming controller | 10 / 10 |
| End-of-turn baseline | 10 / 10 |
| No decomposition | 7 / 10 |
| No selective refinement | 9 / 10 |

Each case includes a nonfinal and a final fragment. The end-of-turn baseline suppresses provisional searches, while the streaming controller starts provisional retrieval in all ten authored cases. This is a controlled replay, not an ASR latency benchmark. Both modes now reach the same 10/10 final source-set result, so the measured advantage here is earlier retrieval rather than higher final accuracy.

The earlier paraphrase failures are now handled by generic concept normalization, metadata-value inference, and confidence filtering rather than exact query-string rules. For example, "delegates" and "fit" map toward capacity, "called off" toward cancellation, and "overseas" toward the corpus value "international". These are still hand-authored development cases, not evidence of general semantic understanding. The tenth case verifies three independent sentences with three different city scopes in one utterance. The full controller returns Pune capacity, Delhi cancellation, and Bengaluru catering without collapsing them into one global city. Representative unseen evaluation remains required.

## Regression checks

The suite covers early retrieval, unchanged-final reuse, decomposition, selective refinement, formatting suppression, unknown scopes, citation tampering, revised transcripts, session isolation and trace coverage. It also checks repaired radio indexing, time windows and worker lifecycle. With Streamlit installed, it tests the keyless UI and follow-up flow.

Functional tests do not demonstrate the guide's percentage thresholds. Mechanical quote/ID integrity is distinct from relevance and completeness.

## Still required

Full BBC/Groq inference with credentials; clean-machine Docker execution; representative unseen partial transcripts, implicit constraints and false triggers; concurrency/latency measurements; official replay tests and a demo video.

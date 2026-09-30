# Development evaluation — not official gate results

Run `python scripts/evaluate.py`. Expected and actual IDs are in `development-results.json`. These nine authored synthetic cases cannot establish general accuracy.

| Configuration | Exact final source-set checks |
|---|---:|
| Streaming controller | 6 / 9 |
| End-of-turn baseline | 6 / 9 |
| No decomposition | 5 / 9 |
| No selective refinement | 5 / 9 |

Each case includes a nonfinal and a final fragment. The baseline suppresses provisional searches. The controller can retrieve before final input. This is a controlled replay, not an ASR latency benchmark. Source accuracy is unchanged from the baseline on this set.

## Three retained failures

1. “Could our thirty delegates fit in Cedar Hall?” returns capacity plus irrelevant parking evidence.
2. “What would we forfeit if we called off the gathering in Delhi?” returns no evidence because vocabulary does not match cancellation.
3. “What paperwork is needed to claim an overseas journey?” returns both domestic and international policies: the scope parser does not infer international from overseas.

The evaluation phrases were not added to production rules. A semantic retriever/controller needs separate evaluation on unseen queries.

## Regression checks

The suite covers early retrieval, unchanged-final reuse, decomposition, selective refinement, formatting suppression, unknown scopes, citation tampering, revised transcripts, session isolation and trace coverage. It also checks repaired radio indexing, time windows and worker lifecycle. With Streamlit installed, it tests the keyless UI and follow-up flow.

Functional tests do not demonstrate the guide's percentage thresholds. Mechanical quote/ID integrity is distinct from relevance and completeness.

## Still required

Full BBC/Groq inference with credentials; clean-machine Docker execution; representative unseen partial transcripts, implicit constraints and false triggers; concurrency/latency measurements; official replay tests and a demo video.

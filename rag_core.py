"""Session-scoped streaming retrieval. No network, persistent memory, or LLM facts."""
from __future__ import annotations

import copy
import json
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

STOP = set('a an the is are was were be to of for from in on at with and or i we you me my our your it this that what how where when which please need want know tell about can could would do does did have has had actually instead also there'.split())
ALIASES = {
    'attendees': 'people', 'attendee': 'people', 'persons': 'people',
    'delegates': 'people', 'delegate': 'people', 'participants': 'people',
    'participant': 'people', 'guests': 'people', 'guest': 'people',
    'seats': 'capacity', 'seating': 'capacity', 'fit': 'capacity',
    'fits': 'capacity', 'fitting': 'capacity', 'accommodate': 'capacity',
    'accommodates': 'capacity', 'accommodating': 'capacity',
    'food': 'catering', 'meals': 'catering', 'meal': 'catering',
    'cancel': 'cancellation', 'cancelled': 'cancellation',
    'canceled': 'cancellation', 'canceling': 'cancellation',
    'cancelling': 'cancellation', 'refund': 'cancellation',
    'refunds': 'cancellation', 'forfeit': 'cancellation',
    'forfeited': 'cancellation', 'penalty': 'cancellation',
    'penalties': 'cancellation', 'expenses': 'reimbursement',
    'expense': 'reimbursement', 'claim': 'reimbursement',
    'claims': 'reimbursement', 'claimed': 'reimbursement',
    'claiming': 'reimbursement', 'overseas': 'international',
    'abroad': 'international', 'foreign': 'international',
    'wheelchair': 'accessibility', 'accessible': 'accessibility',
    'cars': 'parking', 'car': 'parking', 'vehicles': 'parking',
    'vehicle': 'parking',
}
PHRASE_ALIASES = (
    (re.compile(r'\b(?:call|called|calling)\s+off\b', re.I), ' cancellation '),
    (re.compile(r'\bmoney\s+back\b', re.I), ' cancellation '),
    (re.compile(r'\boutside\s+(?:the\s+)?country\b', re.I), ' international '),
    (re.compile(r'\bstep[- ]?free\b', re.I), ' accessibility '),
)


def tokens(text):
    for pattern, replacement in PHRASE_ALIASES:
        text = pattern.sub(replacement, text)
    words = re.findall(r"[\w]+", text.casefold())
    return [ALIASES.get(w, w[:-1] if len(w) > 4 and w.endswith('s') else w) for w in words if w not in STOP]


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    metadata: dict


def load_corpus(payload):
    if isinstance(payload, str):
        payload = json.loads(payload)
    rows = payload.get('documents', []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1500:
        raise ValueError('Corpus must contain 1–1500 documents.')
    chunks, ids = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each document must be an object.')
        doc_id, text = row.get('id'), row.get('text')
        if not isinstance(doc_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,100}', doc_id):
            raise ValueError('Document IDs must use letters, numbers, dots, colons, underscores or hyphens.')
        if doc_id in ids or not isinstance(text, str) or not text.strip():
            raise ValueError('Document IDs must be unique and text must be nonempty.')
        ids.add(doc_id)
        if len(text) > 100000:
            raise ValueError('Each document must be smaller than 100,000 characters.')
        metadata = row.get('metadata', {})
        if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, (str, int, float, bool)) for k, v in metadata.items()):
            raise ValueError('Metadata must contain scalar values.')
        title = str(row.get('title', doc_id))[:200]
        # Split at paragraph/sentence boundaries, preserving exact source substrings.
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        parts = []
        for paragraph in paragraphs:
            while len(paragraph) > 1000:
                end = paragraph.rfind(' ', 400, 1000)
                end = end if end > 0 else 1000
                parts.append(paragraph[:end])
                paragraph = paragraph[end:].lstrip()
            if paragraph:
                parts.append(paragraph)
        for i, part in enumerate(parts, 1):
            chunks.append(Chunk(f'{doc_id}:s{i}', title, part, dict(metadata)))
    if len(chunks) > 4000:
        raise ValueError('Corpus exceeds 4000 chunks.')
    return chunks


class Retriever:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.by_id = {c.id: c for c in chunks}
        self.terms = [Counter(tokens(c.title + ' ' + c.text)) for c in chunks]
        self.df = Counter(t for terms in self.terms for t in terms)
        self.avg = sum(sum(t.values()) for t in self.terms) / max(1, len(chunks))
        self.slots = {}
        self.topics = set()
        for c in chunks:
            for k, v in c.metadata.items():
                if k == 'topic':
                    self.topics.add(str(v))
                elif isinstance(v, str) and k not in {'source', 'date', 'author'}:
                    self.slots.setdefault(k, set()).add(v)

    def constraints(self, text):
        found = {}
        canonical = set(tokens(text))
        for key, values in self.slots.items():
            # Explicit key:value also supports unknown values: those must not use stale evidence.
            explicit = re.search(r'\b' + re.escape(key) + r'\s*[:=]\s*([^,;\n]+)', text, re.I)
            if explicit:
                found[key] = explicit.group(1).strip().rstrip('.?!')
                continue
            matches = [(m.start(), v) for v in values for m in re.finditer(r'(?<!\w)' + re.escape(v) + r'(?!\w)', text, re.I)]
            if matches:
                found[key] = max(matches)[1]
                continue
            # Canonical aliases can imply a known metadata value without hard-coding
            # corpus-specific query strings (e.g. "overseas" -> trip_type=international).
            inferred = [v for v in values if set(tokens(str(v))) and set(tokens(str(v))) <= canonical]
            if len(inferred) == 1:
                found[key] = inferred[0]
        return found

    def strip_constraints(self, text):
        for key, values in self.slots.items():
            text = re.sub(r'\b' + re.escape(key) + r'\s*[:=]\s*[^,;\n]+', ' ', text, flags=re.I)
            for value in values:
                text = re.sub(r'(?<!\w)' + re.escape(value) + r'(?!\w)', ' ', text, flags=re.I)
        return text.strip(' ,;.!?')

    def search(self, query, constraints, limit=2):
        q = set(tokens(self.strip_constraints(query)))
        topic_matches = [t for t in self.topics if set(tokens(t)) & q]
        ranked = []
        for chunk, terms in zip(self.chunks, self.terms):
            if any(k in chunk.metadata and str(chunk.metadata[k]).casefold() != str(v).casefold() for k, v in constraints.items()):
                continue
            if topic_matches and chunk.metadata.get('topic') not in topic_matches:
                continue
            overlap = q & terms.keys()
            if not overlap:
                continue
            # Require meaningful query coverage so one stray word cannot become evidence.
            if len(q) >= 3 and len(overlap) / len(q) < 0.34:
                continue
            score = 0.0
            length = sum(terms.values())
            for term in overlap:
                freq = terms[term]
                idf = math.log(1 + (len(self.chunks) - self.df[term] + .5) / (self.df[term] + .5))
                score += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * length / max(self.avg, 1)))
            ranked.append((score, chunk.id, chunk))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        if ranked:
            # Keep near-ties for broad questions, but drop weak secondary matches when
            # a named entity or paraphrase makes one passage clearly more specific.
            best = ranked[0][0]
            ranked = [row for row in ranked if row[0] >= best * .82]
        return [c for _, _, c in ranked[:limit]]


class StreamingSession:
    def __init__(self, chunks, *, early=True, decompose=True, refine=True):
        self.retriever = Retriever(chunks)
        self.early, self.decompose, self.refine = early, decompose, refine
        self.constraints = {}
        self.intents = {}
        self.events = []
        self.versions = []
        self.version = 0
        self.last_timestamp = -1.0
        self.turn = None
        self.style = 'sections'
        self._closed_turns = set()
        self._turn_keys = set()
        self._turn_baseline = {}
        self._turn_start_constraints = {}

    def log(self, kind, **fields):
        self.events.append(dict(seq=len(self.events) + 1, event=kind, timestamp_s=self.last_timestamp,
                                wall_time_utc=datetime.now(timezone.utc).isoformat(), turn_id=self.turn,
                                answer_version=self.version, **fields))

    def _parts(self, text):
        if not self.decompose:
            return [text]
        # Split both conversational conjunctions and complete sentences. This lets
        # one utterance carry independent scopes, e.g. Pune capacity + Delhi policy.
        pattern = (
            r'\s+(?:and|also|plus|as well as|along with)\s+'
            r'|(?<=[.!?])\s+'
            r'|[;\n]+\s*'
            r'|,\s*(?=(?:what|how|where|when|which|tell|give|show)\b)'
        )
        return [
            p.strip(' ,;.!?')
            for p in re.split(pattern, text, flags=re.I)
            if p.strip(' ,;.!?')
        ]

    def _scoped_parts(self, parts):
        """Return per-part constraints plus only the constraints shared by the turn."""
        local = [self.retriever.constraints(part) for part in parts]
        shared = dict(self._turn_start_constraints)
        keys = set().union(*(scope.keys() for scope in local)) if local else set()
        for key in keys:
            values = {}
            for scope in local:
                if key in scope:
                    values[str(scope[key]).casefold()] = scope[key]
            if len(values) == 1:
                shared[key] = next(iter(values.values()))
            elif len(values) > 1:
                # Conflicting values belong to individual intents, not session scope.
                shared.pop(key, None)
        effective = [{**shared, **scope} for scope in local]
        return local, shared, effective

    @staticmethod
    def _format_only(text):
        return bool(re.fullmatch(r'(?:please\s+)?(?:(?:repeat|rewrite|format|summarize|shorten)\s+(?:your\s+|the\s+)?(?:(?:last|previous|same)\s+)?(?:answer|response|it)(?:\s+(?:in|as|into)\s+(?:two|2)\s+bullets)?|make\s+it\s+shorter|(?:two|2)\s+bullets)[.!?\s]*', text.strip(), re.I))

    def ingest(self, text, *, timestamp_s, final=False, turn_id='1'):
        started = time.perf_counter()
        if not isinstance(text, str) or len(text) > 12000:
            raise ValueError('Transcript must be text, at most 12,000 characters.')
        if not isinstance(final, bool) or not isinstance(timestamp_s, (int, float)) or isinstance(timestamp_s, bool) or not math.isfinite(timestamp_s):
            raise ValueError('Events need a finite numeric timestamp and boolean final flag.')
        if timestamp_s < 0 or timestamp_s < self.last_timestamp:
            raise ValueError('Timestamps must be nonnegative and monotonic across the session.')
        turn_id = str(turn_id)
        if turn_id in self._closed_turns:
            raise ValueError('This turn is finalized; use a new turn_id.')
        if self.turn != turn_id:
            if self.turn is not None:
                self._closed_turns.add(self.turn)
            self.turn = turn_id
            self._turn_keys = set()
            self._turn_baseline = copy.deepcopy(self.intents)
            self._turn_start_constraints = dict(self.constraints)
        self.last_timestamp = float(timestamp_s)
        self.log('transcript_received', text=text, final=final)
        previous = self.snapshot()['claims']
        if self.intents and self._format_only(text):
            self.style = 'two_bullets' if re.search(r'\b(?:two|2)\s+bullets\b', text, re.I) else 'sections'
            self.log('retrieval_suppressed', reason='presentation_only')
            return self._finish(previous, final, started, presentation=True)
        if not final and self.intents and re.match(r'^(?:please\s+)?(?:repeat|rewrite|format|shorten)\b', text, re.I):
            self.log('retrieval_wait', reason='presentation_request_incomplete')
            return self._finish(previous, final, started)
        parts = self._parts(text)
        local_scopes, target, effective_scopes = self._scoped_parts(parts)
        incoming = self.retriever.constraints(text)
        changed = {k for k in set(target) | set(self.constraints) if self.constraints.get(k) != target.get(k)}
        correction = bool(self._turn_baseline) and bool(incoming) and bool(re.search(r'\b(?:actually|instead|correction|meant|change|make that)\b', text, re.I))
        if not final and not self.early:
            self.log('retrieval_wait', reason='end_of_turn_baseline')
            return self._finish(previous, final, started)
        self.constraints = target
        jobs = []
        if correction and self.refine:
            for key, intent in self.intents.items():
                if changed & set(intent['dependencies']):
                    scope = dict(intent.get('constraints', {}))
                    for field in changed:
                        if field in target:
                            scope[field] = target[field]
                        else:
                            scope.pop(field, None)
                    jobs.append((key, self.retriever.strip_constraints(intent['query']), scope))
            affected = {job[0] for job in jobs}
            self.log(
                'refinement_planned',
                changed_constraints=sorted(changed),
                affected_intents=sorted(affected),
                preserved_intents=[k for k in self.intents if k not in affected],
            )
        else:
            if correction and not self.refine:
                self.intents = {}
            active = set()
            for idx, part in enumerate(parts):
                key = f'{turn_id}:{idx}'
                scope = effective_scopes[idx] if idx < len(effective_scopes) else dict(target)
                # Incomplete trailing clause waits; already stable earlier clauses can search.
                words = tokens(self.retriever.strip_constraints(part))
                stable = len(words) >= 2 and not re.search(r'\b(?:in|at|and|the|for|with|of|to)\s*$', part, re.I)
                if not final and not stable:
                    continue
                if not words:
                    continue
                active.add(key)
                old = self.intents.get(key)
                normalized = ' '.join(words)
                if old is None or old['normalized'] != normalized or old.get('constraints', {}) != scope:
                    jobs.append((key, part, scope))
            # Remove evidence retracted by a revised cumulative transcript in this same turn.
            for key in self._turn_keys - active:
                self.intents.pop(key, None)
            self._turn_keys = active
        if jobs:
            self.log(
                'retrieval_started',
                trigger='final' if final else 'provisional',
                sub_queries=[q for _, q, _ in jobs],
                intent_constraints=[dict(scope) for _, _, scope in jobs],
                constraints=dict(self.constraints),
            )

            def retrieve(job):
                key, query, scope = job
                before = time.perf_counter()
                docs = self.retriever.search(query, scope)
                return key, query, scope, docs, round((time.perf_counter() - before) * 1000, 3)

            with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
                results = list(pool.map(retrieve, jobs))

            for key, query, scope, docs, latency in results:
                dependencies = set(k for c in docs for k in c.metadata if k in self.retriever.slots)
                # For an unknown result, scoped changes must be able to retry the query.
                if not docs:
                    dependencies.update(scope)
                claims = [{'source_id': c.id, 'quote': c.text, 'title': c.title} for c in docs]
                self.intents[key] = dict(
                    query=query,
                    normalized=' '.join(tokens(self.retriever.strip_constraints(query))),
                    constraints=dict(scope),
                    dependencies=sorted(dependencies),
                    claims=claims,
                    uncertainty=None if docs else 'No supporting evidence found in this corpus.',
                )
                self.log(
                    'retrieval_completed',
                    intent_id=key,
                    query=query,
                    intent_constraints=dict(scope),
                    source_ids=[c.id for c in docs],
                    latency_ms=latency,
                )
        else:
            self.log('retrieval_wait' if not self.intents else 'retrieval_reused', reason='incomplete_or_unchanged')
        return self._finish(previous, final, started)

    def _finish(self, previous, final, started, presentation=False):
        claims = self.snapshot()['claims']
        if claims != previous or presentation:
            self.version += 1
            self.versions.append(dict(version=self.version, claims=copy.deepcopy(claims), constraints=dict(self.constraints), style=self.style))
            self.log('answer_updated', source_ids=sorted({c['source_id'] for item in claims for c in item['evidence']}), inference_tokens=0, generation='extractive')
        if final:
            self._closed_turns.add(self.turn)
        self.log('event_completed', final=final, latency_ms=round((time.perf_counter() - started) * 1000, 3))
        return self.snapshot()

    def snapshot(self):
        claims = [
            dict(
                intent_id=key,
                query=v['query'],
                scope=copy.deepcopy(v.get('constraints', {})),
                evidence=copy.deepcopy(v['claims']),
                uncertainty=v['uncertainty'],
            )
            for key, v in self.intents.items()
        ]
        # Mechanically verify every quote and ID before returning it.
        for item in claims:
            for claim in item['evidence']:
                source = self.retriever.by_id.get(claim['source_id'])
                if source is None or claim['quote'] not in source.text:
                    raise RuntimeError('Citation validation failed.')
        return dict(answer_version=self.version, constraints=dict(self.constraints), claims=claims, style=self.style, retrieval_count=sum(e['event'] == 'retrieval_completed' for e in self.events))

    def export(self):
        return dict(schema_version=1, mode='corpus_only_extractive', answer=self.snapshot(), versions=self.versions, events=self.events)


def simulate_stream(text, turn_id='1', start=0.0, group_size=4, interval=.4):
    words = text.split()
    if not words:
        return
    for n in range(group_size, len(words) + group_size, group_size):
        count = min(n, len(words))
        yield dict(text=' '.join(words[:count]), timestamp_s=round(start + ((n // group_size) - 1) * interval, 3), final=count == len(words), turn_id=str(turn_id))

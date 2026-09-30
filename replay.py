"""JSONL stdin/file replay runner with JSON output; standard library only."""
import argparse
import json
import sys
from pathlib import Path
from rag_core import StreamingSession, load_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default=str(Path(__file__).parent / 'data/demo_corpus.json'))
    parser.add_argument('--input', default='-')
    parser.add_argument('--end-of-turn', action='store_true')
    parser.add_argument('--no-decomposition', action='store_true')
    parser.add_argument('--no-refinement', action='store_true')
    args = parser.parse_args()
    engine = StreamingSession(load_corpus(Path(args.corpus).read_text()), early=not args.end_of_turn, decompose=not args.no_decomposition, refine=not args.no_refinement)
    source = sys.stdin if args.input == '-' else open(args.input, encoding='utf-8')
    try:
        for number, line in enumerate(source, 1):
            if line.strip():
                try:
                    engine.ingest(**json.loads(line))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'Line {number}: {exc}') from exc
    finally:
        if source is not sys.stdin:
            source.close()
    print(json.dumps(engine.export(), indent=2))


if __name__ == '__main__':
    main()

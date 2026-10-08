"""Standalone read-only report: python -m agent.incident_report."""
import argparse
import json
import sqlite3

from agent.incidents import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--examples', type=int, default=3)
    args = parser.parse_args()
    try:
        rows = report(limit=args.limit, examples=args.examples)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f'Incident report unavailable: {type(exc).__name__}\n')
    print(json.dumps(rows, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()

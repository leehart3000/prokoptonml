"""A tiny stand-in for VSEARCH, used by the leakage tests.

It understands only the options the leakage stage passes. An identical
sequence scores 100.0, and one differing only in its last base scores 99.5.
"""

import sys


def read_fasta(path: str) -> dict[str, str]:
    records: dict[str, str] = {}
    name = ""
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
                records[name] = ""
            elif line:
                records[name] += line
    return records


def main(args: list[str]) -> None:
    option = {
        args[i]: args[i + 1]
        for i in range(0, len(args) - 1)
        if args[i].startswith("--")
    }
    queries = read_fasta(option["--usearch_global"])
    database = read_fasta(option["--db"])
    with open(option["--userout"], "w", encoding="utf-8") as out:
        for query, query_sequence in queries.items():
            for target, target_sequence in database.items():
                if query_sequence == target_sequence:
                    out.write(f"{query}\t{target}\t100.0\n")
                elif query_sequence[:-1] == target_sequence[:-1]:
                    out.write(f"{query}\t{target}\t99.5\n")


if __name__ == "__main__":
    main(sys.argv[1:])

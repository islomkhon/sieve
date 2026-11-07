"""Build SIEVE's compact offline English lexicon from a WordNet archive."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import zipfile
from pathlib import Path


INDEX_FILES = {
    "noun": "wordnet/index.noun",
    "verb": "wordnet/index.verb",
    "adjective": "wordnet/index.adj",
    "adverb": "wordnet/index.adv",
}
DATA_FILES = {
    "noun": "wordnet/data.noun",
    "verb": "wordnet/data.verb",
    "adjective": "wordnet/data.adj",
    "adverb": "wordnet/data.adv",
}
EXCEPTION_FILES = {
    "noun": "wordnet/noun.exc",
    "verb": "wordnet/verb.exc",
    "adjective": "wordnet/adj.exc",
}
VALID_LEMMA = re.compile(r"[a-z][a-z0-9'_-]*$")


def read_glosses(archive: zipfile.ZipFile, name: str) -> dict[str, str]:
    glosses: dict[str, str] = {}
    with archive.open(name) as source:
        for raw_line in source:
            line = raw_line.decode("latin-1").rstrip()
            if not line or line.startswith(" "):
                continue
            synset, separator, gloss = line.partition("|")
            if not separator:
                continue
            offset = synset.split(" ", 1)[0]
            glosses[offset] = " ".join(gloss.strip().split())
    return glosses


def read_index(
    archive: zipfile.ZipFile,
    index_name: str,
    data_name: str,
    part_of_speech: str,
    entries: dict[str, dict[str, object]],
) -> None:
    glosses = read_glosses(archive, data_name)
    with archive.open(index_name) as source:
        for raw_line in source:
            line = raw_line.decode("latin-1").rstrip()
            if not line or line.startswith(" ") or line.startswith("Copyright"):
                continue
            fields = line.split()
            if len(fields) < 7:
                continue
            lemma = fields[0].lower()
            if not VALID_LEMMA.match(lemma):
                continue
            pointer_count = int(fields[3])
            offset_index = 4 + pointer_count + 2
            if offset_index >= len(fields):
                continue
            offset = fields[offset_index]
            definition = glosses.get(offset, "")
            description = entries.setdefault(
                lemma.replace("_", " "), {"p": [], "d": ""}
            )
            categories = description["p"]
            assert isinstance(categories, list)
            if part_of_speech not in categories:
                categories.append(part_of_speech)
            if definition and not description["d"]:
                description["d"] = definition


def read_exceptions(archive: zipfile.ZipFile, name: str) -> dict[str, str]:
    irregular: dict[str, str] = {}
    with archive.open(name) as source:
        for raw_line in source:
            fields = raw_line.decode("latin-1").lower().split()
            if len(fields) >= 2:
                irregular.setdefault(
                    fields[0].replace("_", " "), fields[1].replace("_", " ")
                )
    return irregular


def build(source: Path, destination: Path) -> None:
    entries: dict[str, dict[str, object]] = {}
    irregular: dict[str, str] = {}
    with zipfile.ZipFile(source) as archive:
        for part_of_speech, index_name in INDEX_FILES.items():
            read_index(
                archive,
                index_name,
                DATA_FILES[part_of_speech],
                part_of_speech,
                entries,
            )
        for name in EXCEPTION_FILES.values():
            irregular.update(read_exceptions(archive, name))

    payload = {
        "source": "WordNet 3.0",
        "entries": dict(sorted(entries.items())),
        "irregular": dict(sorted(irregular.items())),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8", compresslevel=9) as output:
        json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))

    phrase_count = sum(" " in word for word in entries)
    print(
        f"Wrote {len(entries):,} lexical entries "
        f"({phrase_count:,} multiword) to {destination}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wordnet_zip", type=Path)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("sieve/data/english_lexicon.json.gz"),
    )
    args = parser.parse_args()
    build(args.wordnet_zip, args.output)


if __name__ == "__main__":
    main()

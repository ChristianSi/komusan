# Notes for agents working on Komusan

Komusan is an auxiliary language (see README.md). This repo holds its
dictionary, the source data used to build it, and the Python scripts that do
the work. Many files in `data/` are logs, backups (`*.bak`, `*~`) or old study
output – ignore them unless the task needs them.

## Running scripts

- **Scripts must be run from `data/`** (they use relative file names), e.g.
  `cd data && ../scripts/buildvoc.py --polycheck ...`. Run elsewhere, they
  fail.
- Python 3, standard library plus `editdistance`.
- Checks: `make` in `scripts/` runs flake8 and mypy; pylint is also used. All
  should pass.

## Key data files (in `data/`)

- `termdict.txt` – English Wiktionary entries with their translations. Format:
  `key: value` lines, entries separated by empty lines. Important fields:
  `class` (POS: noun/adj/verb…), `sense` (the sense under which Wiktionary
  groups translations), `transcount` (number of languages with translations –
  an informal measure of how important and well-documented a concept is), `en`
  (the English word, often followed by `[pronunciation]`), plus one line per
  language code. It is large; avoid re-parsing it in ad-hoc scripts.
- `extradict.txt` – same format; corrections and additions merged into
  termdict entries with the same `en` word and `sense` (`tags: add` adds
  entries missing from termdict).
- `dict.txt` – the Komusan dictionary itself (`word:`, `class:`, `infl:` …
  entries).
- `sourcelangs.csv` – the source languages used for word building and the
  polycheck.
- `target_wordlist.txt` – words (from the Simple English Wikipedia list) that
  Komusan should have words for. Lines may contain variants ("big, large") and
  comments in parentheses ("May (month)").
- `target_wordlist.csv` – maps each target word to the Wiktionary concepts it
  should cover. Columns: `Word` (as in the .txt), `Concepts`
  (`en_word::sense`, several joined by ` | `), `Transcount` (the highest of
  the listed concepts). A word can have several rows if it has several
  important meanings that shouldn't be merged (e.g. "fine": good vs. penalty).
  Complete for the whole word list; edit it directly.

## Scripts

- `buildvoc.py` – the main tool: picks the next concept, proposes Komusan
  candidate words and adds the chosen one to the dictionary. See its
  docstring. `buildvoc.py --polycheck WORD1 SENSE1 WORD2 SENSE2` gives the
  share of languages that translate two concepts the same way.
- `findconcepts.py` – lists a word's termdict concepts sorted by transcount,
  with polycheck-style similarity hints and "act/result of" nouns for verbs.
  Supports `--range FIRST LAST` (lines of target_wordlist.txt) and `--sim
  WORD::SENSE WORD::SENSE...`. Caches the parsed termdict in the temp dir, so
  it's fast after the first run.
- `util.py` – shared helpers and file-name constants; reuse them in new
  scripts.

## Choosing concepts for a word

- Merge concepts when one Komusan word should cover them all: a polycheck
  similarity of 50%+ usually means merge; 40%+ is a judgment call (leave out
  the rarer or less fitting one). Use the polycheck when in doubt, own
  judgment when the case is clear.
- Concepts with transcount ≤ 20 can usually be ignored, unless the concept
  fits the word particularly well and is important anyway.
- For verbs, also add the "act or result of" noun(s) (agree → agreement,
  decide → decision); these don't need the polycheck.
- Target words are raw words, not concepts: merge list entries where
  reasonable ("appear, seem"), or split one entry into rows when its meanings
  are distinct.

## Code conventions

- Compact lowercase script names (`buildvoc.py`, `findconcepts.py`), made
  executable.
- Every script starts with a docstring explaining what it does, its input and
  output files, and its usage; use argparse for the command line.
- If necessary, silence pylint's size warnings with a `# pylint: disable=...`
  comment, as `buildvoc.py` does.
- Python line length: 100 characters
- Python quote style: single quotes

Markdown:

- Custom-written Markdown files in the repository should usually be wrapped at
  78 characters. Headings, code blocks, tables, URLs, literal paths, and
  generated output may be longer.
- Use sentence case for Markdown headings, retaining the standard
  capitalization of proper nouns and acronyms.

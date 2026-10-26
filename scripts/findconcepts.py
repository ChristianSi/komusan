#!/usr/bin/env python3
"""Find the Wiktionary concepts (termdict.txt entries) that may represent a target word.

This must be invoked in the data/ directory. It's a helper for choosing the concepts listed
in target_wordlist.csv.

For each target word, the termdict entries whose English word matches are listed, sorted by
transcount (number of translations), one per line in the form

    INDEX TRANSCOUNT CLASS SIMILARITY WORD::SENSE

SIMILARITY shows how closely the entry is related to the most similar entry listed above it,
e.g. "75%~0(8)" means that 75% of the 8 shared languages translate this entry and entry 0
in the same way. This is calculated like the --polycheck option of buildvoc.py, so 50%+
usually means both concepts should be merged.

For verbs, nouns denoting the act or result of the verb are listed afterwards (lines starting
with "N"). They are found by looking at their sense definitions ("act of deciding") and by
trying common derivational suffixes ("agreement", "decision").

Usage:

    findconcepts.py [--min N] [--max N] WORD...   # look up words
    findconcepts.py [--min N] [--max N] --range FIRST LAST
                                                  # look up lines of target_wordlist.txt
    findconcepts.py --sim WORD::SENSE WORD::SENSE...
                                                  # compare 1st concept with each other one

Target words may contain variants ("big, large") and comments in parentheses ("May (month)"),
just as in target_wordlist.txt.

The parsed termdict is cached in a pickle file in the system's temporary directory, so only
the first invocation (and the first one after termdict.txt has changed) is slow.
"""

import argparse
import csv
import os
import pickle
import re
import sys
import tempfile
from collections import defaultdict
from typing import Iterable, Optional, Sequence

from parsewikt import EXTRA_FALLBACKS
import util

# Type aliases: termdict entries are dicts of fields; the index maps English words to the
# positions of their entries
Entry = dict[str, str]
Index = dict[str, list[int]]

TARGET_WORDLIST = 'target_wordlist.txt'
CACHE_FILE = os.path.join(tempfile.gettempdir(), 'komusan-termdict.pkl')

# Fields of termdict entries that look like language codes but aren't
NON_LANG_FIELDS = frozenset(['cls'])

# Candidates with fewer translations are only shown if there are less than three candidates
DEFAULT_MIN_TRANSCOUNT = 21

# Maximum number of candidates and of action nouns shown per word
DEFAULT_MAX_CANDIDATES = 12
MAX_ACTION_NOUNS = 6

# Suffixes tried to find nouns derived from verbs
NOUN_SUFFIXES = ('ment', 'ion', 'ation', 'ition', 'al', 'ance', 'ence', 'ure', 'y', 'ing')


def read_source_langs() -> frozenset[str]:
    """Return the set of languages considered by the polysemy check.

    This mirrors buildvoc.py: our source languages (including fallback languages) and the
    auxlangs we use by default.
    """
    result = set(util.DEFAULT_AUXLANGS)
    with open(util.SOURCELANGS_FILE, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            result.update(row['ISO 639'].split('/'))
    for fallbacks in EXTRA_FALLBACKS.values():
        result.update(fallbacks.split('/'))
    return frozenset(result)


def english_word(entry: Entry) -> str:
    """Return the English word of an entry, without pronunciation."""
    return util.discard_text_in_brackets(entry.get('en', '')).strip()


def transcount(entry: Entry) -> int:
    """Return the number of translations of an entry."""
    return int(entry.get('transcount', 0))


def load_termdict() -> tuple[list[Entry], Index]:
    """Read termdict.txt (or its cached version) and return its entries and index."""
    if (os.path.exists(CACHE_FILE)
            and os.path.getmtime(CACHE_FILE) > os.path.getmtime(util.TERM_DICT)):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    entries: list[Entry] = []
    current: Entry = {}
    with open(util.TERM_DICT, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                if current:
                    entries.append(current)
                current = {}
            elif ': ' in line:
                key, value = line.split(': ', 1)
                current[key] = value
    if current:
        entries.append(current)

    index: Index = defaultdict(list)
    for pos, entry in enumerate(entries):
        index[english_word(entry)].append(pos)
    result = (entries, dict(index))

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(result, f)
    return result


def find_entries(entries: Sequence[Entry], index: Index, word: str,
                 sense: str) -> list[Entry]:
    """Return all entries with the specified English word and sense."""
    return [entries[pos] for pos in index.get(word, []) if entries[pos].get('sense') == sense]


def translation_sets(entry: Entry, langs: Optional[frozenset[str]],
                     is_first: bool) -> dict[str, set[str]]:
    """Map language codes to the set of lower-cased translations of an entry.

    If 'langs' is given, only these languages are considered. Following buildvoc.py,
    explanations in parentheses are removed from the English words of the first entry.
    """
    result = {}
    for key, value in entry.items():
        if len(key) > 3 or key in util.COMMON_AUXLANGS or key in NON_LANG_FIELDS:
            continue
        if langs is not None and key not in langs:
            continue
        translations = set()
        for trans in util.split_on_semicolons(util.discard_text_in_brackets(value)):
            if key == 'en' and is_first:
                trans = util.eliminate_parens(trans)
            translations.add(trans.lower())
        result[key] = translations
    return result


def similarity(first: Entry, second: Entry,
               source_langs: frozenset[str]) -> tuple[float, int]:
    """Calculate how many languages translate both entries in the same way.

    Returns the percentage and the number of shared languages, as buildvoc.py --polycheck
    would.
    """
    first_trans = translation_sets(first, None, True)
    second_trans = translation_sets(second, source_langs, False)
    shared = set(first_trans) & set(second_trans)
    if not shared:
        return 0.0, 0
    same = sum(1 for lang in shared if first_trans[lang] & second_trans[lang])
    return same / len(shared) * 100, len(shared)


def gerunds(verb: str) -> set[str]:
    """Return the possible -ing forms of a verb."""
    result = {verb + 'ing'}
    if verb.endswith('ie'):
        result.add(verb[:-2] + 'ying')
    elif verb.endswith('e') and not verb.endswith('ee'):
        result.add(verb[:-1] + 'ing')
    if (len(verb) >= 3 and verb[-1] not in 'aeiouwxy' and verb[-2] in 'aeiou'
            and verb[-3] not in 'aeiou'):
        result.add(verb + verb[-1] + 'ing')
    return result


def derived_nouns(verb: str) -> set[str]:
    """Return candidate nouns derived from a verb using common suffixes."""
    stems = {verb, verb[:-1]} if verb.endswith('e') else {verb}
    result = {stem + suffix for stem in stems for suffix in NOUN_SUFFIXES}
    if verb.endswith('de'):
        result.add(verb[:-2] + 'sion')  # decide → decision
    if verb.endswith('t'):
        result.add(verb[:-1] + 'ssion')  # admit → admission
    return result


def find_action_nouns(entries: Sequence[Entry], index: Index, verbs: Iterable[str],
                      exclude: set[int]) -> list[Entry]:
    """Find nouns denoting the act or result of one of the specified verbs.

    Entries whose positions are in 'exclude' are skipped. The result is sorted by transcount.
    """
    patterns = []
    candidate_words = set()
    for verb in verbs:
        alternatives = '|'.join(re.escape(form) for form in gerunds(verb) | {verb})
        patterns.append(re.compile(
            rf'\b(act|action|process|result|instance|state|fact)\b[^;]*?\b'
            rf'(of|in|or)\s+(\w+\s+)?({alternatives})\b', re.IGNORECASE))
        candidate_words |= derived_nouns(verb)

    found = set()
    for pos, entry in enumerate(entries):
        if entry.get('class') == 'noun' and pos not in exclude:
            if any(pattern.search(entry.get('sense', '')) for pattern in patterns):
                found.add(pos)
    for word in candidate_words:
        found.update(pos for pos in index.get(word, [])
                     if entries[pos].get('class') == 'noun' and pos not in exclude)
    return sorted((entries[pos] for pos in found), key=lambda entry: -transcount(entry))


def show_candidates(entries: Sequence[Entry], index: Index, target: str,
                    source_langs: frozenset[str], min_transcount: int,
                    max_candidates: int) -> None:
    """Print the candidate concepts for a target word."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    base = re.sub(r'\s*\(.*?\)', '', target).strip()
    variants = [variant.strip() for variant in base.split(',') if variant.strip()]
    positions = set()
    for variant in variants:
        positions.update(index.get(variant, []))
        if variant.islower():
            # Also find names such as "Earth" or "God"
            positions.update(index.get(variant.capitalize(), []))
    candidates = sorted(positions, key=lambda pos: -transcount(entries[pos]))

    shown = [pos for pos in candidates if transcount(entries[pos]) >= min_transcount]
    if len(shown) < 3:
        shown = candidates[:3]
    shown = shown[:max_candidates]
    hidden = len(candidates) - len(shown)
    print(f'=== {target}' + (f'  (+{hidden} more)' if hidden else ''))

    for num, pos in enumerate(shown):
        entry = entries[pos]
        best_info, best_value = '', -1.0
        for prev_num in range(num):
            value, shared = similarity(entries[shown[prev_num]], entry, source_langs)
            if value > best_value:
                best_info, best_value = f'{value:.0f}%~{prev_num}({shared})', value
        print(f'{num:2} {transcount(entry):>4} {entry.get("class", "?"):5} {best_info:13} '
              f'{english_word(entry)}::{entry.get("sense", "")}')

    verbs = {english_word(entries[pos]) for pos in candidates
             if entries[pos].get('class') == 'verb'} & set(variants)
    if verbs:
        for entry in find_action_nouns(entries, index, verbs, set(candidates))[
                :MAX_ACTION_NOUNS]:
            print(f'   N{transcount(entry):>4}  {english_word(entry)}::{entry.get("sense")}')


def compare_concepts(entries: Sequence[Entry], index: Index, specs: Sequence[str],
                     source_langs: frozenset[str]) -> None:
    """Print the similarity of the first concept to each of the other ones."""
    found = []
    for spec in specs:
        word, sense = spec.split('::', 1)
        matches = find_entries(entries, index, word, sense)
        if not matches:
            sys.exit(f'error: no matching entry found for "{spec}"')
        found.append(matches[0])
    for spec, entry in zip(specs[1:], found[1:]):
        value, shared = similarity(found[0], entry, source_langs)
        print(f'{value:5.1f}% ({shared:2}) {transcount(entry):>4}  {spec}')


def main() -> None:
    """Parse the command line and do the requested lookups."""
    parser = argparse.ArgumentParser(
        description='Find the Wiktionary concepts that may represent target words.')
    parser.add_argument('--min', type=int, default=DEFAULT_MIN_TRANSCOUNT,
                        help='minimum transcount of candidates to show (default: %(default)s)')
    parser.add_argument('--max', type=int, default=DEFAULT_MAX_CANDIDATES,
                        help='maximum number of candidates to show per word '
                        '(default: %(default)s)')
    parser.add_argument('--range', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help=f'also look up these lines (1-based) of {TARGET_WORDLIST}')
    parser.add_argument('--sim', action='store_true',
                        help='compare the first WORD::SENSE concept with each of the others')
    parser.add_argument('words', nargs='*', metavar='WORD',
                        help='words to look up (or WORD::SENSE concepts if --sim is used)')
    args = parser.parse_args()

    entries, index = load_termdict()
    source_langs = read_source_langs()

    if args.sim:
        if len(args.words) < 2:
            parser.error('--sim needs at least two WORD::SENSE concepts')
        compare_concepts(entries, index, args.words, source_langs)
        return

    targets = list(args.words)
    if args.range:
        first, last = args.range
        with open(TARGET_WORDLIST, encoding='utf-8') as f:
            lines = [line.strip() for line in f]
        targets.extend(line for line in lines[first - 1:last] if line)
    for target in targets:
        show_candidates(entries, index, target, source_langs, args.min, args.max)


if __name__ == '__main__':
    main()

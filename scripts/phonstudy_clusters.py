#!/usr/bin/env python3
"""Do a phonology study of the word-inintial consonant pairs occurring in our source languages."""

import argparse
from collections import defaultdict
import re
from types import MethodType
from typing import Counter, Dict, FrozenSet, Sequence, Set

import buildutil as bu
from buildutil import LOG, export_word
import buildvoc
from linedict import LineDict
import phonstudy as ps
import util


##### Constants and precompiled regexes #####

VOWEL_RE = re.compile(rf'[{bu.SIMPLE_VOWELS}]')


# Monkeypatch logger to suppress warnings we don't need
LOG.warn = MethodType(ps.selective_warn, LOG)  # type: ignore


class PhonStudy(buildvoc.VocBuilder):
    """Adapts VocBuilder to do the study."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Create a new instance."""
        super().__init__(args)
        # Mapping from language codes to the words in this language
        self.word_map: Dict[str, Set[str]] = defaultdict(set)

    def determine_kinds_to_add(self, entries: Sequence[LineDict]) -> FrozenSet[buildvoc.Kind]:
        """We overwrite this since we don't need it."""
        return frozenset()

    def process_entry(self, entry: LineDict) -> None:
        """Process an entry, generating candidates and updating statistics."""
        try:
            cls = entry['class']
        except KeyError:
            LOG.warn(f'Incomplete entry without class: {entry}')

        if cls.endswith('fix') or cls.endswith('phrase') or cls == 'proverb':
            # Skip affixes, phrases and proverbs since they aren't actual words and don't add
            # anything new
            return

        cand_map = self.build_candidates(entry)
        for lang, cands in cand_map.items():
            for cand in cands:
                # We only accept valid candidates (note that the validate() method doesn't
                # work like you would expect, hence the "not"!)
                if not cand.validate():
                    for word in cand.word.split():
                        self.word_map[lang].add(word)

    def analyze_initial_clusters(self, lang: str, words: Set[str]) -> Set[str]:
        """Analyze initial consonant pairs found in a language.

        Return the set of those pairs that start at least 0.1% of all words.
        """
        word_count = len(words)
        LOG.info(
            f'*** Consonant pairs ending in semivowel or glide in {lang} ({word_count} words):')
        pair_counter: Counter[str] = Counter()

        for word in words:
            if len(word) > 2 and word[0] in bu.ALL_CONSONANTS and word[1] in bu.SECOND_CONSONANTS:
                pair_counter[word[:2]] += 1

        result = set()

        for pair, count in sorted(pair_counter.items(), key=lambda pair: (-pair[1], pair[0])):
            percentage = count * 100.0 / word_count
            if percentage >= 0.1:
                result.add(pair)
                LOG.info(f'{export_word(pair)}: {percentage:.1f}% ({count})')
            else:
                break
        return result

    def show_results(self, accepted_results: Dict[str, Set[str]]) -> None:
        """Print results accepted by various languages in descending order of acceptance."""
        # Reorganize results, asking for each value: which languages did accept it?
        results_by_value: Dict[str, Set[str]] = defaultdict(set)
        for lang, result_set in accepted_results.items():
            for value in result_set:
                results_by_value[value].add(lang)

        LOG.info('*** Patterns by cross-lingual acceptance:')
        for pattern, lang_set in sorted(results_by_value.items(),
                                        key=lambda pair: (-len(pair[1]), pair[0])):
            lang_set_str = ', '.join(sorted(lang_set))
            LOG.info(f'{export_word(pattern)}: {len(lang_set)} ({lang_set_str})')

    def run(self) -> None:
        """Run the study."""
        # Reset existing entries
        self.existing_entries = []
        counter = 1

        count_map = self.sort_entries_by_transcount()
        LOG.info('Processing all entries to collect words...')

        # Iterate all entries
        for entries in count_map.values():
            for entry in entries:
                self.process_entry(entry)
                counter += 1
                if counter % 10000 == 0:
                    LOG.info(f'Processed {counter} entries')

        accepted_results: Dict[str, Set[str]] = {}
        # Do requested study for each language
        for lang, words in sorted(self.word_map.items()):
            # We only consider main languages, not fallback ones
            if lang not in self._sourcelang_list:
                continue

            # Eliminate 'ə' from all words
            pure_word = {word.replace('ə', '') for word in words}
            accepted_results[lang] = self.analyze_initial_clusters(lang, pure_word)

        self.show_results(accepted_results)
        LOG.info(f'* Study completed on {util.current_datetime()}.')


##### main entry point #####

if __name__ == '__main__':
    # pylint: disable=invalid-name
    study = PhonStudy(argparse.ArgumentParser().parse_args(namespace=ps.DefaultFalseNamespace()))
    study.run()

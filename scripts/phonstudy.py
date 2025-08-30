#!/usr/bin/env python3
"""Do a phonology study of the consonants occurring at the end of words in our source languages.

Note that this study only works with an early version of buildvoc.py where postprocessing and
validation are are disabled or minimized. Results with later versions will be misleading!
"""

import argparse
from collections import defaultdict
import re
from types import MethodType
from typing import Counter, Dict, FrozenSet, Sequence, Set

import buildutil as bu
from buildutil import LOG, export_word
import buildvoc
from linedict import LineDict
import util


##### Constants and precompiled regexes #####

VOWEL_RE = re.compile(rf'[{bu.SIMPLE_VOWELS}]')


# Monkeypatch logger to suppress warnings we don't need
def selective_warn(self, msg: str) -> None:
    """Print a warning method, except in cases where we don't need it.

    This will be monkeypatched into LOG to suppress unnecessary warnings.
    """
    if (msg.startswith('Invalid candidate for') or msg.startswith('Unexpected character')
            or msg.startswith('No raw candidate') or 'word found for' in msg):
        return
    self.info('WARNING: ' + msg)


LOG.warn = MethodType(selective_warn, LOG)  # type: ignore


##### Helper classes #####

class DefaultFalseNamespace(argparse.Namespace):
    """A namespace that returns False when an attribute isn't found."""

    def __getattr__(self, name):
        """Just returns False."""
        if name == '_unrecognized_args':
            # Let argparse handle this attribute normally
            raise AttributeError(name)
        return False


##### Main class #####

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
        cls = entry['class']
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
                        # Voiced plosives become devoiced at the end of words in German and Russian
                        if lang in ('de', 'ru') and word[-1] in 'bdg':
                            match word[-1]:
                                case 'b':
                                    replacement = 'p'
                                case 'd':
                                    replacement = 't'
                                case 'g':
                                    replacement = 'k'
                            word = word[-1] + replacement

                        self.word_map[lang].add(word)

    @staticmethod
    def is_consonant(letter: str) -> bool:
        """Return True if a letter is a consonant, False otherwise.

        This assumes the internal representation and is therefore case-sensitive (e.g.
        'C' is recognized as a consonant, while 'c' is not).
        """
        return letter in bu.ALL_CONSONANTS

    def analyze_final_consonants(self, lang: str, words: Set[str]) -> Set[str]:
        """Analyze final consonants found in a language.

        Return the set of those consonants that end at least 1% of all words.
        """
        # 1st limit: consonants that will be accepted, 2nd (lower) one: does that will be shown
        word_count = len(words)
        LOG.info(f'*** Final consonants in {lang} ({word_count} words):')
        final_cons_counter: Counter[str] = Counter()

        for word in words:
            last_letter = word[-1]
            if self.is_consonant(last_letter):
                final_cons_counter[last_letter] += 1

        LOG.info('* Consonants above limit:')
        above = True
        result = set()

        for consonant, count in sorted(final_cons_counter.items(),
                                       key=lambda pair: (-pair[1], pair[0])):
            percentage = count * 100.0 / word_count

            if percentage >= 0.5:
                result.add(consonant)
            else:
                if percentage <= 0.1:
                    break  # Stop showing results at this point
                if above:
                    # First value below the limit
                    LOG.info('* Consonants below limit:')
                    above = False

            LOG.info(f'{export_word(consonant)}: {percentage:.1f}% ({count})')
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
            accepted_results[lang] = self.analyze_final_consonants(lang, pure_word)

        self.show_results(accepted_results)
        LOG.info(f'* Study completed on {util.current_datetime()}.')


##### main entry point #####

if __name__ == '__main__':
    # pylint: disable=invalid-name
    study = PhonStudy(argparse.ArgumentParser().parse_args(namespace=DefaultFalseNamespace()))
    study.run()

#!/usr/bin/env python3
"""Determine the relative frequencies of the various WALS features among our source languages."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import logging
import re
from typing import Dict, List, Tuple, Set

import util


##### Constants #####

OUTFILE = 'walsfeatures.csv'

# We use Egyptian Arabic (the best documented variant) as a fallback for Modern Standard Arabic
# and Sango (the best documented creole) for Nigerian Pidgin
EXTRA_FALLBACKS = {'ar': 'arz', 'pcm': 'sg'}

EXTRA_NAMES = {'arz': 'Egyptian Arabic', 'sg': 'Sango'}


##### Dataclass #####

@dataclass(frozen=True, eq=True)
class FeatureValue:
    """Stores which feature value a language represents in one feature map."""

    iso_code: str
    """The ISO code of the language."""

    map_name: str
    """The name of the map."""

    value: int
    """The value the language represents in this map."""


##### Main class and entry point #####

class FeatureFreqFinder:
    """Determines the frequencies of the various WALS features among our source languages."""

    def __init__(self):
        """Create a new instance."""
        # Create instance attributes
        self._source_dict, self._fallback_map, self._lang_names = self._read_sourcelangs_file()
        self._iso3_to_iso1_map = self._fill_iso3_to_iso1_map()
        self._wals_to_iso_map = self._map_wals_codes()
        self._feature_mapping = self._fill_feature_mapping()
        self._value_names = self._name_feature_values()
        self._lang_counter = Counter()

    @property
    def lang_names(self) -> Dict[str, str]:
        """Returns the mapping from ISO codes to language names, e.g. 'en' to 'English'.

        This includes fallback languages.
        """
        return self._lang_names

    @property
    def iso3_to_iso1_map(self) -> Dict[str, str]:
        """Return a mapping from ISO 639-3 code (3 letters) to ISO 639-1 (2 letters) code.

        The result only lists source and fallback languages that have both these codes.
        """
        return self._iso3_to_iso1_map

    def source_set(self, incl_fallback_langs: bool = True) -> Set[str]:
        """Return the set of source languages (ISO codes).

        Set `incl_fallback_langs` to False to exclude fallback languages."""
        if incl_fallback_langs:
            return set(self._source_dict)
        return set(self._source_dict) - set(self._fallback_map)

    @staticmethod
    def _read_sourcelangs_file() -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str]]:
        """Read the "sourcelangs.csv" file and return info based on it.

        Specifically, it returns three values:

        1. A mapping of the ISO 639 codes of the source languages used to their position in the list
           (1 for the first entry, i.e. the most widely spoken language; 2 for the second entry
           etc.).
        2. A mapping from fallback to main languages (identified by ISO code), e.g. 'ur' is the
           fallback for 'hi' and 'ms' the fallback for 'id'.
        3. A mapping from ISO codes to language names, e.g. 'en' to 'English'
        """
        source_dict = {}
        fallback_map = {}
        lang_names = EXTRA_NAMES.copy()  # These extra fallback names will be needed in any case

        with util.open_csv_reader(util.SOURCELANGS_FILE) as reader:
            for pos, row in enumerate(reader, start=1):
                # 2nd is the ISO code, or it maybe by a main/fallback combination such as 'hi/ur'
                name, iso_code = row[:2]

                if '/' in iso_code:
                    main, fallback = iso_code.split('/', 1)
                    source_dict[main] = pos
                    source_dict[fallback] = pos
                    fallback_map[fallback] = main

                    # Split name field in the same way
                    main_name, fallback_name = name.split('/', 1)
                    lang_names[main] = main_name
                    lang_names[fallback] = fallback_name
                else:
                    source_dict[iso_code] = pos
                    lang_names[iso_code] = name

                    # Integrate extra fallback, if defined
                    extra_fallback = EXTRA_FALLBACKS.get(iso_code)
                    if extra_fallback is not None:
                        source_dict[extra_fallback] = pos
                        fallback_map[extra_fallback] = iso_code

        # Rename for clarity, since we use another variety of Arabic as fallback
        lang_names['ar'] = 'Standard Arabic'
        return source_dict, fallback_map, lang_names

    @staticmethod
    def _fill_iso3_to_iso1_map() -> Dict[str, str]:
        """Return a mapping from ISO 639-3 code (3 letters) to ISO 639-1 (2 letters) code.

        This information is read from the "codescripts.csv" file and is relevant for languages that
        have codes, since we the prefer the shorter one, while WALS prefers the longer one.
        """
        iso3_to_iso1_map = {}
        with util.open_csv_reader(util.CODESCRIPTS_FILE) as reader:
            for row in reader:
                # Map rows to local variables
                iso1, iso3 = row[:2]
                # Add code mapping, if both codes exist
                if iso1 != '–' and iso3 != '–':
                    iso3_to_iso1_map[iso3] = iso1

        return iso3_to_iso1_map

    def _map_wals_codes(self) -> Dict[str, str]:
        """Map the ISO codes of all source languages to their WALS codes.

        For example, 'cmn' (Mandarin Chinese) becomes 'mnd', 'es' (Spanish) becomes 'spa'.

        Returns a mapping from found WALS codes to the corresponding ISO codes listed in
        `self._source_dict`.

        If there are source languages whose WALS codes can't be found, a warning will be logged.
        """
        result: Dict[str, str] = {}
        iso_codes_identified = set()

        with util.open_csv_reader('cldf/languages.csv') as reader:
            for row in reader:
                wals = row[0]
                name = row[1]
                iso3 = row[6]
                # WALS always uses the ISO 639-3 code, while we prefer the shorter ISO 639-1 code
                # for languages that have both
                actual_iso = self._iso3_to_iso1_map.get(iso3, iso3)

                # Sometimes WALS has additional codes for language variants with an additional
                # identifier in parentheses, e.g. "Indonesian (Jakarta)" or "Tamil (Spoken)".
                # Except for variants of Arabic, we skip those, as they tend to have fewer
                # entries than the main language.
                # There's also a spurious entry, which we skip
                if ('(' in name and 'Arabic' not in name) or name == 'Kunming':
                    continue

                if actual_iso in self._source_dict:
                    # Warn if a language is listed repeatedly, since we have to make sure we get
                    # the right one
                    if actual_iso in iso_codes_identified:
                        logging.warning(f'Found several WALS codes for "{actual_iso}": '
                                        f'{result[actual_iso]}, {wals}')

                    result[wals] = actual_iso
                    iso_codes_identified.add(actual_iso)

        # Check that we found all languages
        if self._source_dict.keys() - iso_codes_identified:
            logging.warning('No WALS codes found for '
                            + ', '.join(sorted(self._source_dict.keys() - iso_codes_identified)))

        return result

    def _fill_feature_mapping(self) -> Dict[str, List[FeatureValue]]:
        """Create a mapping from the WALS maps to the source language representations in that map.
        """
        result = defaultdict(list)
        with util.open_csv_reader('cldf/values.csv') as reader:
            for row in reader:
                wals_code, map_name, value = row[1:4]

                # If the language is one of source sources, we create an entry for this mapping
                iso_code = self._wals_to_iso_map.get(wals_code)
                if iso_code is not None:
                    result[map_name].append(FeatureValue(iso_code, map_name, int(value)))

        return result

    def _name_feature_values(self) -> Dict[Tuple[str, int], str]:
        """Return a mapping to the feature names.

        Keys are (map name, value) tuples, values are the corresponding names describing the
        value. For example, ('1A', 3)' will be mapped to 'Average'.
        """
        result = {}
        with util.open_csv_reader('cldf/codes.csv') as reader:
            for row in reader:
                map_name = row[1]
                value_num = int(row[4])
                value_name = row[2]
                result[(map_name, value_num)] = value_name

        return result

    def _group_feature_values(self, feature_values: List[FeatureValue]) -> Dict[int, Set[str]]:
        """Group the reported values of a feature by he specific values.

        Returns a mapping of each specific value to the set of source languages having this value.

        All FeatureValue's are supposed to belong to the same feature map, i.e. to have the same
        `map_name`.

        The function also filters out fallback languages – if both the main (e.g. 'hi') and the
        fallback language (e.g. 'ur') are present, the latter will be dropped, regardless of
        whether both represent the same value.

        This method also updates `self._lang_counter`, incrementing the count of each listed
        language. Counters for filtered-out fallbacks will not be incremented.
        """
        result = defaultdict(set)
        # Create set of all represented languages for fallback filtering
        langset = {feature_value.iso_code for feature_value in feature_values}

        for feature_value in feature_values:
            main_lang = self._fallback_map.get(feature_value.iso_code)
            if main_lang is not None and main_lang in langset:
                # Drop fallback since main language is present too
                continue

            result[feature_value.value].add(feature_value.iso_code)
            self._lang_counter[feature_value.iso_code] += 1
        return result

    def _find_smallest_language_position(self, langset: Set[str]) -> int:
        """Return the smallest (best) position of any of the languages in the set.

        `langset` must be a set of languages in `self._source_dict`. If some elements aren't
        listed there, an exception will be raised.
        """
        return min(self._source_dict[lang] for lang in langset)

    def _write_feature_mapping(self) -> None:
        """Write the WALS feature values of our source languages to a CSV file."""
        with util.open_csv_writer(OUTFILE) as writer:
            # Write header row
            writer.writerow(['Feature', 'Value', 'Name', 'Languages', 'Language count',
                             'Relative frequency'])

            # Iterate features in order (sorted first by the number in their name, then the full
            # name)
            for map_name, feature_list in sorted(
                    self._feature_mapping.items(),
                    # We sort the feature maps first by their numeric part, then by their full name
                    key=lambda pair: (int(re.search(r'\d+', pair[0]).group()),  # type: ignore
                                      pair[0])):
                grouped_values = self._group_feature_values(feature_list)
                max_langcount = None

                # Print feature values in order of frequency (from most to least frequent).
                # Ties are resolved by the relative position the most widely spoken language
                # (a value represented by English will always win as long as that language is
                # listed first in our source languages list, Mandarin beats Hindi und Spanish, etc.)
                for value, langset in sorted(
                        grouped_values.items(),
                        key=lambda pair: (-len(pair[1]),
                                          self._find_smallest_language_position(pair[1]))):
                    langcount = len(langset)
                    if max_langcount is None:
                        max_langcount = langcount
                    rel_frequency = round(langcount * 100 / max_langcount)

                    writer.writerow([map_name, value, self._value_names[(map_name, value)],
                                     ', '.join(sorted(langset)), langcount,
                                     str(rel_frequency) + '%'])

    def _print_stats(self) -> None:
        """Print frequency statistics to a separate file."""
        statsfilename = 'walsfeatures-stats.txt'
        util.rename_to_backup(statsfilename)

        # Also count each main language together with its fallback, and remember to skip them later
        main_and_fallback_langs: Set[str] = set()
        for fallback, main in self._fallback_map.items():
            self._lang_counter[f'{main}/{fallback}'] = (self._lang_counter[main]
                                                        + self._lang_counter[fallback])
            main_and_fallback_langs.update((main, fallback))

        with open(statsfilename, 'w') as statsfile:
            for lang, count in sorted(self._lang_counter.items(),
                                      key=lambda pair: (-pair[1], pair[0])):
                if lang in main_and_fallback_langs:
                    continue  # These will be printed together

                if '/' in lang:
                    # It's a fallback pair
                    main, fallback = lang.split('/', 1)
                    main_name = self._lang_names[main]
                    fallback_name = self._lang_names[fallback]
                    main_count = self._lang_counter[main]
                    fallback_count = self._lang_counter[fallback]
                    statsfile.write(f'{count} values known for {main_name} ({main_count}) and '
                                    f'{fallback_name} ({fallback_count})\n')
                else:
                    statsfile.write(f'{count} values known for {self._lang_names[lang]}\n')

    def find_feature_freq(self) -> None:
        """Determines the frequencies of the various WALS features among our source languages."""
        self._write_feature_mapping()
        self._print_stats()


if __name__ == '__main__':
    freq_finder = FeatureFreqFinder()
    freq_finder.find_feature_freq()

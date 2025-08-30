#!/usr/bin/env python3
"""Check some WALS features for consistency and completeness."""

from collections import Counter, defaultdict
import logging
from typing import DefaultDict, Dict
from typing import Counter as CounterType

from printwalsarea import FeatureValue
import util
import walsfeaturefreq


##### Constants #####

INOUTFILE = walsfeaturefreq.OUTFILE


##### Main class and entry point #####

class WalsChecker:
    """Check some WALS features for consistency and completeness"""

    def __init__(self):
        """Create a new instance."""
        # Create instance attributes
        freq_finder = walsfeaturefreq.FeatureFreqFinder()
        self._source_set = freq_finder.source_set(False)
        self._fallback_map = freq_finder._fallback_map
        self._feature_maps = self._read_feature_maps()
        self._all_well = True
        self._data_has_changed = False

    def _read_feature_maps(self) -> DefaultDict[str, list[FeatureValue]]:
        """Read and return all features with their values.

        Also stores the header row to allow later reserialization
        """
        result = defaultdict(list)
        with util.open_csv_reader(INOUTFILE, False) as reader:
            self._header = next(reader)
            for row in reader:
                feature_value = FeatureValue.from_row(row)
                feature_id = feature_value.feature
                result[feature_id].append(feature_value)
        return result

    def _check_feature_completeness_and_consistency(self, feature_map: str) -> None:
        """Check that a feature map is complete and consistent.

        It's complete if all our principle source languages have a value listed, and no other
        language has.

        It's consistent with no language occurs more than once.

        Note that while all features should be consistent, only manually completed ones can be
        expected to be complete.

        Prints a warning and sets self._all_well to False if problems are detected.
        """
        features = self._feature_maps[feature_map]

        if not features:
            logging.warning(f'Feature map {feature_map} not found')
            self._all_well = False
            return

        # Count how often each language occurs
        lang_counter: CounterType[str] = Counter()
        for feature in features:
            lang_counter.update(feature.languages)

        # Count fallbacks for main languages instead (e.g. 'ur' is counted as 'hi')
        for fallback, main in self._fallback_map.items():
            if fallback in lang_counter:
                lang_counter[main] += lang_counter[fallback]
                del lang_counter[fallback]

        # Check that no language is listed repeatedly
        repeated_langs = {lang: count for lang, count in lang_counter.items() if count >= 2}
        if repeated_langs:
            logging.warning(f'Some languages are listed repeatedly for feature map {feature_map}: '
                            f'{", ".join(sorted(repeated_langs))}')
            self._all_well = False

        # Check that all source languages are listed and that no other languages are
        langs_listed = set(lang_counter.keys())
        missing_languages = self._source_set - langs_listed
        if missing_languages:
            logging.warning(f'Some languages are missing for feature map {feature_map}: '
                            f'{", ".join(sorted(missing_languages))}')
            self._all_well = False

        extra_languages = langs_listed - self._source_set
        if extra_languages:
            logging.warning(f'Spurious languages listed for feature map {feature_map}: '
                            f'{", ".join(sorted(extra_languages))}')
            self._all_well = False

    def _correct_feature_counts(self, feature_map: str) -> None:
        """Check and if needed correct language counts and relative frequencies in a feature map.

        This makes sure that afterwards features are again sorted by their frequencies and that
        the summary data is accurate.

        If any change in the feature data is made, self._feature_maps is updated accordingly and
        self._data_has_changed is set to True.
        """
        data_has_changed = False
        features = self._feature_maps[feature_map]
        prev_lang_count = 1000000  # starting with absurdly high initial value
        features_are_out_of_order = False

        for feature in features:
            actual_lang_count = len(feature.languages)
            if actual_lang_count != feature.language_count:
                feature.language_count = actual_lang_count
                data_has_changed = True
            if actual_lang_count > prev_lang_count:
                features_are_out_of_order = True
            prev_lang_count = actual_lang_count

        if features_are_out_of_order:
            # Features need to be resorted, from highest to lowest language count
            features = sorted(features, key=lambda f: f.language_count, reverse=True)

        if data_has_changed:
            # Relative frequencies need to be recalculated
            top_lang_count = features[0].language_count
            for feature in features:
                feature.relative_frequency = round(100 * feature.language_count / top_lang_count)

            # Store modified data and set the flag signaling that it has changed
            self._feature_maps[feature_map] = features
            self._data_has_changed = True

    def _check_consistency_between_article_features(self) -> None:
        """Check that the features maps referring to articles are consistent with each other.

        Prints a warning and sets self._all_well to False if problems are detected.
        """
        has_neither: CounterType[str] = Counter()
        has_definite = set()
        has_definite_but_no_indefinite = set()
        has_indefinite = set()
        has_indefinite_but_no_definite = set()

        for feature in self._feature_maps['37A']:
            match feature.value:
                case 5:  # Neither definite nor indefinite article
                    has_neither.update(feature.languages)
                case 4:  # No definite article but indefinite article
                    has_indefinite_but_no_definite.update(feature.languages)
                case _:  # Some kind of definite article
                    has_definite.update(feature.languages)

        for feature in self._feature_maps['38A']:
            match feature.value:
                case 5:  # Neither definite nor indefinite article
                    has_neither.update(feature.languages)
                case 4:  # No indefinite article but definite article
                    has_definite_but_no_indefinite.update(feature.languages)
                case _:  # Some kind of indefinite article
                    has_indefinite.update(feature.languages)

        # "Neither" features should be identical, hence each language should occur twice
        singleton_languages = {lang: count for lang, count in has_neither.items() if count == 1}
        if singleton_languages:
            logging.warning('Language lists for 37A.5 and 38A.5 should be identical, but some '
                            f'languages occur only once: {", ".join(singleton_languages)}')
            self._all_well = False

        # These sets should be empty, else there is a contradiction
        spurious_definite = has_definite_but_no_indefinite - has_definite
        if spurious_definite:
            logging.warning('Some languages are listed in 38A as having a definite article, '
                            f'but not in 37A: {", ".join(spurious_definite)}')
            self._all_well = False

        spurious_indefinite = has_indefinite_but_no_definite - has_indefinite
        if spurious_indefinite:
            logging.warning('Some languages are listed in 37A as having an indefinite article, '
                            f'but not in 38A: {", ".join(spurious_indefinite)}')
            self._all_well = False

    def _cross_combine(self, fmap1: str, fmap2: str, outfmap: str) -> None:
        """Stores a cross-combination of two feature maps as an additional feature map.

        The results feature map lists which languages have which value of the first map
        in combination with which value of the second one. Both parts are separated by a slash.

        When language isn't mentioned in one of the source features, the corresponding part will
        be set to '???'. If a language isn't mentioned at all, it won't be added to the combined
        feature either.

        Sets self._data_has_changed to True if `outfmap` (the combined feature map) didn't exist
        before or if some of its values have changed.
        """
        features1 = self._feature_maps[fmap1]
        features2 = self._feature_maps[fmap2]
        old_extra_features = self._feature_maps.get(outfmap)

        # For each mentioned source language, calculate its cross-combined feature name (value)
        combined_names_per_language = {}

        for feature in features1:
            for lang in feature.languages:
                combined_names_per_language[lang] = feature.name

        for feature in features2:
            for lang in feature.languages:
                combined_name = combined_names_per_language.get(lang, '???')
                combined_names_per_language[lang] = combined_name + '/' + feature.name

        # Add '/???' for languages mentioned in the first, but not the second feature map
        for lang, name in combined_names_per_language.items():
            if '/' not in name:
                combined_names_per_language[lang] += '/???'

        # Sort set of combined names and assign them numeric values based on their sort order
        sorted_names = sorted(set(combined_names_per_language.values()))
        name_value_map = {name: idx for idx, name in enumerate(sorted_names, start=1)}

        # Create resulting features
        name_feature_map: Dict[str, FeatureValue] = {}
        for lang, name in sorted(combined_names_per_language.items()):
            feature = name_feature_map.get(name)
            if feature:
                # Just add the language
                feature.languages.append(lang)
            else:
                # Create new feature (count and relative frequency will be set later)
                feature = FeatureValue(outfmap, name_value_map[name], name, [lang], 0, 0)
                name_feature_map[name] = feature

        extra_features = name_feature_map.values()

        # Add language counts
        for feature in extra_features:
            feature.language_count = len(feature.languages)

        # Sort based on language count (highest to lowest)
        sorted_extra_features = sorted(extra_features, key=lambda f: f.language_count, reverse=True)

        # Add relative frequencies
        top_lang_count = sorted_extra_features[0].language_count
        for feature in sorted_extra_features:
            feature.relative_frequency = round(100 * feature.language_count / top_lang_count)

        if old_extra_features is None or old_extra_features != sorted_extra_features:
            self._feature_maps[outfmap] = sorted_extra_features
            self._data_has_changed = True

    def check(self) -> None:
        """Run all relevant checks."""
        for feature_id in ('37A', '38A', '117A', '121A'):
            self._check_feature_completeness_and_consistency(feature_id)
            self._correct_feature_counts(feature_id)
        self._check_consistency_between_article_features()
        self._cross_combine('46A', '56A', '56E')
        self._cross_combine('85A', '86A', '86X')
        self._cross_combine('86A', '87A', '87X')
        self._cross_combine('87A', '90A', '90X')

        if self._data_has_changed:
            # Some data has changed, so we replace the file with an updated copy
            util.rename_to_backup(INOUTFILE)
            with util.open_csv_writer(INOUTFILE) as writer:
                writer.writerow(self._header)
                # We sort the feature maps first by their numeric part, then by their full name
                for mapname in sorted(self._feature_maps, key=lambda x:
                                      (int(''.join(filter(str.isdigit, x))), x)):
                    feature_map = self._feature_maps[mapname]
                    for feature in feature_map:
                        writer.writerow(feature.to_row())

        # Write status message
        msg = 'All is well' if self._all_well else 'Problems detected'
        yes_or_no = '' if self._data_has_changed else "n't"
        msg += f'; {INOUTFILE} has{yes_or_no} changed'
        print(msg)


if __name__ == '__main__':
    checker = WalsChecker()
    checker.check()

#!/usr/bin/env python3
"""Print a human-readable summary of the feature frequencies for any given WAlS area.

As of 2024, WALS is grouped into 11 areas, numbered from 1 to 11."""

from collections import defaultdict
from dataclasses import dataclass
import sys
from typing import Dict, IO, List, Set, Tuple

import util
import walsfeaturefreq


##### Dataclass #####

@dataclass(eq=True)
class FeatureValue:
    """Represents a feature values with its occurrences in our source languages."""

    feature: str
    """The feature ID."""

    value: int
    """The number of the value."""

    name: str
    """The name of the value."""

    languages: List[str]
    """The list of source languages (ISO codes) having this feature value."""

    language_count: int
    """The number of languages having this value."""

    relative_frequency: int
    """The relative frequency of this value compared to the most frequent one.

    This is a rounded percentage converted to an int: 100 for most frequent, 50 for half as
    frequent, etc.
    """

    def to_row(self) -> List[str]:
        """Export the fields of this instance as a list suitable for writing into a CSV file."""
        return [self.feature, str(self.value), self.name, ', '.join(sorted(self.languages)),
                str(self.language_count), f'{self.relative_frequency}%']

    @classmethod
    def from_row(cls, data: List[str]) -> 'FeatureValue':
        """Create an instance object from a CSV row."""
        if len(data) != 6:
            raise ValueError("Input row must contain exactly 6 elements")

        feature, value, name, languages, lang_count, frequency = data

        return cls(
            feature=feature,
            value=int(value),
            name=name,
            languages=[lang.strip() for lang in languages.split(',')],
            language_count=int(lang_count),
            relative_frequency=int(frequency.rstrip('%'))
        )


##### Main class and entry point #####

class AreaPrinter:
    """Print a human-readable summary of WALS feature frequencies for any given WAlS area."""

    def __init__(self):
        """Create a new instance."""
        # Create instance attributes
        freq_finder = walsfeaturefreq.FeatureFreqFinder()
        self._lang_names = freq_finder.lang_names
        self._langs_seen = set()
        self._quorum = self._calc_quorum()

    @staticmethod
    def _calc_quorum() -> int:
        """Calculate the quorum below which features will be skipped.

        That's the case if less than 40% (rounded) of all source languages have values for them.
        """
        with util.open_csv_reader(util.SOURCELANGS_FILE) as reader:
            return round(len(list(reader)) * 0.4)

    def print_area(self, area_num: int) -> None:
        """Print a human-readable summary of the feature frequencies in the specified WAlS area.

        The output will be written to a Markdown (md) file.

        If no features can be found for that area, an error will be raised instead.
        """
        outfilename = f'wals-area-{area_num}.md'
        # Retrieve area name as well as the features in it
        area_name = self._find_area_name(area_num)
        feature_map = self._find_features_in_area(area_num)
        feature_value_map = self._collect_feature_values(set(feature_map.keys()))

        # Make sure we got some features
        if not feature_value_map:
            raise ValueError(f'No features found for WALS area {area_num}')

        # Filter by quorum
        feature_value_map, features_below_quorum = self._filter_features_by_quorum(
            feature_value_map)

        util.rename_to_backup(outfilename)
        with open(outfilename, 'w') as outfile:
            outfile.write(f'# WALS Features for the "{area_name}" section\n\n')
            outfile.write(f'"{area_name}" is section {area_num} in WALS.\n')

            # Iterate over features
            for feature_id, feature_values in feature_value_map.items():
                self._print_feature_values(
                    outfile, feature_id, feature_map[feature_id], feature_values)

            if features_below_quorum:
                self._print_features_below_quorum(outfile, features_below_quorum, feature_map)

        print(f'Output written to {outfilename}')

    def _find_area_name(self, area_num: int) -> str:
        """Look up the name corresponding to an area number."""
        with util.open_csv_reader('cldf/areas.csv') as reader:
            for row in reader:
                if int(row[0]) == area_num:
                    return row[1]  # Got it!

        raise ValueError(f'WALS area {area_num} not found')

    def _find_features_in_area(self, area_num: int) -> Dict[str, str]:
        """Return a mapping from the features in an area to their descriptive names.

        Keys are the feature IDs tying them to a chapter, such as '2A' or '137B'.
        The dictionary is created in the natural order in which feature maps are listed in WALS,
        i.e sorted first by chapter number and then feature ID within that chapter (A, B, C etc.).
        """
        # Get set of chapters belonging to the area
        chapter_set = set()
        with util.open_csv_reader('cldf/chapters.csv') as reader:
            for row in reader:
                raw_area_num = row[7]  # Either empty or a number
                if raw_area_num and int(raw_area_num) == area_num:
                    chapter_set.add(int(row[0]))

        # Find features belonging to the found chapters
        result = {}
        with util.open_csv_reader('cldf/parameters.csv') as reader:
            for row in reader:
                if int(row[4]) in chapter_set:
                    feature_id = row[0]
                    feature_name = row[1]
                    result[feature_id] = feature_name

        # We also add "Extra" feature (map name ending in E or X) since some such are added by own
        # own scripts
        chapter_ids = [feature_id[:-1] for feature_id in result.keys()]
        for chapter_id in chapter_ids:
            for ending in ('E', 'X'):
                extra_feature_id = chapter_id + ending
                if extra_feature_id not in result:
                    result[extra_feature_id] = 'Cross-combination'

        return result

    @staticmethod
    def _collect_feature_values(feature_ids: Set[str]) -> Dict[str, List[FeatureValue]]:
        """Returns an ordered listing if all found feature values in our area.

        The values of a feature will be returned as listed by walsfeaturefreq.py, ordered from
        most to least frequent among our source languages.
        """
        result = defaultdict(list)
        with util.open_csv_reader(walsfeaturefreq.OUTFILE) as reader:
            for row in reader:
                feature_value = FeatureValue.from_row(row)
                feature_id = feature_value.feature
                if feature_id in feature_ids:
                    result[feature_id].append(feature_value)

        return result

    def _filter_features_by_quorum(self, value_map: Dict[str, List[FeatureValue]]) -> Tuple[
            Dict[str, List[FeatureValue]], Dict[str, int]]:
        """Feature a feature value mapping by whether the language quorum is reached.

        Return a tuple of 2 values:

        1. A mapping of feature IDs to feature values -- as passed on input, but restricted to
           those features for which at least the quorum of source languages have known values
        2. A mapping those feature IDs for which this is not the case to the number of languages
           for which their value is known (all these numbers will be below the quorum)
        """
        filtered_value_map = {}
        features_below_quorum = {}

        for feature_id, feature_values in value_map.items():
            total_language_count = sum(fv.language_count for fv in feature_values)
            if total_language_count >= self._quorum:
                filtered_value_map[feature_id] = feature_values
            else:
                features_below_quorum[feature_id] = total_language_count

        return filtered_value_map, features_below_quorum

    def _print_feature_values(self, outfile: IO[str], feature_id: str, feature_name: str,
                              feature_values: List[FeatureValue]) -> None:
        """Print the ordered values of a feature in human-readable form, in a Markdown section.

        Values with a relative frequency of at least 50% will be explicitly printed.
        Rarer values will merely be mentioned as existing.
        """
        outfile.write(f'\n## {feature_name} (WALS feature {feature_id})\n\n')

        # Group by relative frequency: 100%, >= 50%, < 50%
        top_values = []
        frequent_values = []
        rare_values = []

        for value in feature_values:
            if value.relative_frequency == 100:
                top_values.append(value)
            elif value.relative_frequency >= 50:
                frequent_values.append(value)
            else:
                rare_values.append(value)

        top_header = 'Most frequent values' if len(top_values) > 1 else 'Most frequent value'
        outfile.write(f'{top_header} ({top_values[0].language_count} languages):\n\n')

        for value in top_values:
            outfile.write(f'* **{value.name}** (#{value.value} – '
                          f'{self._format_language_list(value.languages)})\n')

        if frequent_values:
            if len(frequent_values) == 1:
                outfile.write('\nAnother frequent value:\n\n')
            else:
                outfile.write('\nOther frequent values:\n\n')

            for value in frequent_values:
                outfile.write(f'* **{value.name}** (#{value.value}) – {value.language_count} '
                              f'languages ({self._format_language_list(value.languages)} – '
                              f'{value.relative_frequency}% relative frequency)\n')

        if rare_values:
            rare_values_formatted = [
                f'"{value.name}" (#{value.value}, '
                f'{self._lang_count_formatted(value.language_count)})'
                for value in rare_values
            ]

            # Print as formatted list using commas and 'and' as needed
            if len(rare_values_formatted) <= 2:
                rare_values_joined = ' and '.join(rare_values_formatted)
            else:
                rare_values_formatted[-1] = 'and ' + rare_values_formatted[-1]
                rare_values_joined = ', '.join(rare_values_formatted)

            rare_intro = 'A rarer value is' if len(rare_values) == 1 else 'Rarer values are'
            outfile.write(f'\n{rare_intro} {rare_values_joined}.\n')

    def _format_language_list(self, iso_codes: List[str]) -> str:
        """Return a list of languages as a formatted string.

        Open first mention of each language, its full name is printed followed by its ISO code.
        On repeated mentions, only the ISO code is printed.
        """
        lang_names = []
        for iso_code in iso_codes:
            if iso_code in self._langs_seen:
                lang_names.append(iso_code)
            else:
                lang_names.append(f'{self._lang_names[iso_code]}/{iso_code}')
                self._langs_seen.add(iso_code)

        return ', '.join(lang_names)

    @staticmethod
    def _lang_count_formatted(language_count: int) -> str:
        """Format a language count by adding "language" or "languages" after it."""
        return '1 language' if language_count == 1 else f'{language_count} languages'

    def _print_features_below_quorum(self, outfile, features_below_quorum: Dict[str, int],
                                     feature_map: Dict[str, str]) -> None:
        """Print a short summary of the feature that stayed below the language quorum."""
        outfile.write('\n## Features below the language quorum\n\n')
        num_features = len(features_below_quorum)
        intro_text = '1 feature was' if num_features == 1 else f'{num_features} features were'
        outfile.write(f"{intro_text} skipped because they didn't reach the quorum of at least "
                      f'{self._quorum} source languages:\n\n')

        for feature_id, lang_count in features_below_quorum.items():
            outfile.write(f'* {feature_id} ({feature_map[feature_id]}; '
                          f'{self._lang_count_formatted(lang_count)})\n')


if __name__ == '__main__':
    if len(sys.argv) == 2:
        area_printer = AreaPrinter()
        area_printer.print_area(int(sys.argv[1]))
    else:
        # Print an error/help message if no arguments are provided
        print('Error: Specify exactly one WALS area number to print')
        sys.exit(1)

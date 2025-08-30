#!/usr/bin/env python3
"""Collects the phoneme sets given in PHOIBLE for each source language.

Requires the file phoible.csv from the data directory of the PHOIBLE repository at
https://github.com/phoible/dev in the current directory.

Writes three output files:

* phonemes_by_language_raw.csv: lists the phoneme sets of each source language in the form as
  PHOIBLE has them
* phonemes_by_language.csv: likewise, but with variants of the same phoneme combined (e.g. if the
  difference is only aspiration or length), as that more closely corresponds to the methodology used
  by WALS
* frequent_consonants.csv: lists the consonants that occur in at least 20% of the source languages;
  the variants of combined related sounds are listed too, with their frequencies shown
* frequent_vowels.csv: lists the vowels that occur in at least 20% of the source languages, in the
  same ways as for the consonants
"""

from collections import Counter, defaultdict
import dataclasses
from dataclasses import dataclass, field
import logging
import statistics
from typing import FrozenSet, List, Optional, Set

import util
import walsfeaturefreq


##### Constants #####

PHOIBLE_FILE = 'phoible.csv'

# Inventories we don't want to use because of errors
BLOCKED_INVENTORIES: FrozenSet[int] = frozenset({
    286,  # Mandarin, but wrongly lacks /x/
    304,  # German, but excludes both /ʒ/ and /d̠ʒ/, which German has in some words (e.g. 'Genie'
          # 'Dschungel')
    423,  # Korean, but wrongly includes the vowel /æ/
    530,  # Russian, but wrongly includes the vowel /ɛ/
    553,  # Spanish, but wrongly lacks the rhotic /ɾ/
    574,  # Thai, but wrongly includes the vowel /æ/
    597   # Turkish, but wrongly includes the vowel /ɛ/
})


##### Dataclass #####

@dataclass
class LanguageDetails:
    """Wraps information on a language and its sounds."""

    name: str
    """Its English name."""

    iso_code: str
    """Its ISO 639-1 or 639-3 code (2 or 3 letters)."""

    iso3_code: Optional[str] = None
    """Its ISO 639-3 code (3 letters).

    Use this field only for languages that also have a shorter ISO 639-1 code.
    """

    inventory_id: Optional[int] = None
    """The ID of the used inventory stored in PHOIBLE for this language."""

    consonant_list: List[str] = field(default_factory=list)
    """The list of consonant sounds in this language."""

    vowel_list: List[str] = field(default_factory=list)
    """The list of vowel sounds in this language."""

    @property
    def consonant_count(self) -> int:
        """The number of consonant sound."""
        return len(self.consonant_list)

    @property
    def vowel_count(self) -> int:
        """The number of vowel sound."""
        return len(self.vowel_list)


##### Main class and entry point #####

class PhoibleLister:
    """Collects the phoneme sets given in PHOIBLE for each source language."""

    def __init__(self):
        """Create a new instance."""
        freq_finder = walsfeaturefreq.FeatureFreqFinder()
        # Create auxiliary mapping from ISO 639-1 to ISO 639-3 code
        iso1_to_iso3_map = {value: key for key, value in freq_finder.iso3_to_iso1_map.items()}
        lang_names = freq_finder.lang_names

        # Set of inventories with marginality information --they are the ones we prefer
        self._useful_inventories = self._find_useful_inventories()

        # Create mapping from ISO 639-3 codes to language details
        self._lang_map = {}
        for iso_code in freq_finder.source_set(False):
            iso3_code = iso1_to_iso3_map.get(iso_code, iso_code)
            self._lang_map[iso3_code] = LanguageDetails(
                lang_names[iso_code], iso_code, None if iso_code == iso3_code else iso3_code)

        # ISO codes of the 5 most widely spoken source languages
        self._top_5_langs = self._top_5_langs()

        # Translation table for phoneme simplication: Lists the IPA symbols and modifiers characters
        # to remove from phonemes to arrive at the basic phoneme: long (ː), aspirated (ʰ),
        # palatalized (ʲ), labialized (ʷ), pharyngealized (ˤ), ejective (ʼ), velarization (ˠ),
        # breathy voice (double dot below, e.g. b̤), retracted coronal articulation (right tack
        # below, e.g. d̙), dental (bridge below), nasalization (tilde), non-syllabic/semivowel
        # (inverted breve below), rhotic (rhotic hook), glottal stop (modifier), relative
        # articulation (down tack, plus sign below, or minus sign below), syllabic (vertical line
        # below), laminal (square below), voiceless (ring below),  centralization (diaeresis),
        # apical (inverted bridge below)
        self._phoneme_trans_table = str.maketrans(
            '', '',
            'ːʰʲʷˤʼˠ\u0324\u0319\u032a\u0303\u032f\u02de\u02c0\u031e\u031f\u0320\u0329\u033b'
            '\u0325\u0308\u033a')

        # Store variants of each basic sound (that only differ in aspiration, length etc.)
        self._variant_map = defaultdict(set)

    @staticmethod
    def _find_useful_inventories() -> Set[int]:
        """Return a list of those inventories that store marginality information.

        We prefer those inventories of such that just collect phonemes without distinguishing
        whether or not they are marginal. Inventories listed in BLOCKED_INVENTORIES are likewise
        skipped because they contain errors.
        """
        result = set()

        with util.open_csv_reader(PHOIBLE_FILE) as reader:
            for row in reader:
                inventory_id = int(row[0])
                if inventory_id in result or inventory_id in BLOCKED_INVENTORIES:
                    continue  # inventory already handled or we don't want it.
                marginal = row[8]
                if marginal in ('TRUE', 'FALSE'):
                    # Inventory has the marginality information we prefer
                    result.add(inventory_id)

        return result

    @staticmethod
    def _top_5_langs(number: int = 5) -> Set[str]:
        """Returns the top 5 most widely spoken source languages.

        `number` can optionally be set to another number of top languages to return.

        A set of ISO codes will be returned. Fallback languages are not included.
        """
        result = set()
        with util.open_csv_reader(util.SOURCELANGS_FILE) as reader:
            for pos, row in enumerate(reader, start=1):
                if pos > number:
                    break

                # 2nd is the ISO code, or it maybe by a main/fallback combination such as 'hi/ur'
                iso_code = row[1]

                if '/' in iso_code:
                    iso_code = iso_code.split('/', 1)[0]
                result.add(iso_code)

        return result

    def list_sounds(self):
        """Collect and print the phonemes listed in PHOIBLE for the source languages."""
        with util.open_csv_reader(PHOIBLE_FILE) as reader:
            for row in reader:
                iso3_code = row[2]

                # Check if this entry belong to a language we're interested in
                lang_details = self._lang_map.get(iso3_code)
                if lang_details is None:
                    continue

                # Check if the inventory ID matches -- we set it when first encountering a language
                # and skip entries belonging to other inventories since PHOIBLE often has several
                # of them for the same language
                inventory_id = int(row[0])
                if lang_details.inventory_id is None and inventory_id in self._useful_inventories:
                    lang_details.inventory_id = inventory_id
                elif lang_details.inventory_id != inventory_id:
                    continue

                phoneme = row[6]
                if '|' in phoneme:
                    # Some inventories list phonemes in a double form, e.g. 'l̪|l'. In such cases
                    # we keep just the (more general) part after the pipe character.
                    phoneme = phoneme.split('|', 1)[-1]

                marginal = row[8]
                segment_class = row[9]

                if marginal == 'TRUE':
                    continue  # Skip marginal sound
                elif marginal not in ('NA', 'FALSE'):
                    logging.warning(f'Unexpected marginal value: {marginal}')

                if segment_class == 'consonant':
                    lang_details.consonant_list.append(phoneme)
                elif segment_class == 'vowel':
                    lang_details.vowel_list.append(phoneme)
                elif segment_class != 'tone':
                    logging.warning(f'Unexpected segment class: {segment_class}')

        # Sort languages by ISO code and write raw output
        sorted_lang_details = sorted(self._lang_map.values(), key=lambda ld: ld.iso_code)
        self._write_phonemes_by_language(sorted_lang_details, 'phonemes_by_language_raw.csv')

        # Combine related sounds and write additional output files
        filtered_lang_details = [self._combine_related_sounds(ld) for ld in sorted_lang_details]
        self._write_phonemes_by_language(filtered_lang_details, 'phonemes_by_language.csv')
        self._write_frequent_phonemes(filtered_lang_details, sorted_lang_details, True)
        self._write_frequent_phonemes(filtered_lang_details, sorted_lang_details, False)

    def _combine_related_sounds(self, lang_detail: LanguageDetails) -> LanguageDetails:
        """Combine related sounds in the phoneme mappings of a language.

        Returns a new `lang_detail` object with related consonant and vowel sounds combined,
        leading to a counting of phonemes more similar to the methodology used by WALS.
        The original `lang_detail` object is left as is.

        All variants found are also added to `self._variant_map` to allow keeping track of them.
        """
        return dataclasses.replace(
            lang_detail,
            consonant_list=self._simplify_phoneme_set(lang_detail.consonant_list, True),
            vowel_list=self._simplify_phoneme_set(lang_detail.vowel_list, False))

    def _simplify_phoneme_set(self, phoneme_list: List[str], consonants: bool) -> List[str]:
        """Combine related sounds in a phoneme list and return the resulting shorter list.

        The returned list is sorted alphabetically and will contain no duplicates.

        Set `consonants` to True for a list of consonants, False for one of vowels.
        """
        result_set = set()
        for phoneme in sorted(phoneme_list, key=len):
            if len(phoneme) > 1:
                # Strip any modifier symbols and characters
                base_phoneme = phoneme.translate(self._phoneme_trans_table)
            else:
                # Nothing to simplify here
                base_phoneme = phoneme

            if not consonants and len(base_phoneme) > 1:
                # Check for diphthongs (such as 'ai' or 'ɔo') whether all their parts are
                # already listed
                if all(letter in result_set for letter in base_phoneme):
                    # All parts are known, so we don't add this diphthong, but just remember it as a
                    # variant of the first letter (e.g. 'ai' as variant of 'a')
                    self._variant_map[base_phoneme[0]].add(phoneme)
                    continue

            if base_phoneme != phoneme:
                # Add base phoneme and remember actual phoneme as variant
                result_set.add(base_phoneme)
                self._variant_map[base_phoneme].add(phoneme)
            else:
                # Add the phoneme as it is
                result_set.add(phoneme)

        return sorted(result_set)

    def _write_phonemes_by_language(self, lang_details: List[LanguageDetails],
                                    filename: str) -> None:
        """Write the phoneme set of each source language to a CSV file.

        `lang_details` should already be sorted in the order in which they are to be written.

        The average number of consonants and vowels will be stored in the outfile file as well.
        """
        with util.open_csv_writer(filename) as writer:
            # Write header row and init variables
            writer.writerow(['Language', 'ISO 639', 'Consonants', 'Consonant count', 'Vowels',
                             'Vowel count', 'Inventory ID'])
            consonant_count = []
            vowel_counts = []

            # Write individual entries and calculate the sums
            for lang_detail in lang_details:
                writer.writerow([lang_detail.name, lang_detail.iso_code,
                                 ', '.join(lang_detail.consonant_list), lang_detail.consonant_count,
                                 ', '.join(lang_detail.vowel_list), lang_detail.vowel_count,
                                 str(lang_detail.inventory_id)])
                consonant_count.append(lang_detail.consonant_count)
                vowel_counts.append(lang_detail.vowel_count)

            # Add average and median
            writer.writerow(['AVERAGE', '', '',
                             round(statistics.mean(consonant_count), 2), '',
                             round(statistics.mean(vowel_counts), 2), ''])
            writer.writerow(['MEDIAN', '', '',
                             round(statistics.median(consonant_count), 2), '',
                             round(statistics.median(vowel_counts), 2), ''])
            writer.writerow(['MOST FREQUENT', '', '',
                             ', '.join(str(num) for num in
                                       self._most_frequent_values(consonant_count)),
                             '',
                             ', '.join(str(num) for num in
                                       self._most_frequent_values(vowel_counts)),
                             ''])

    @staticmethod
    def _most_frequent_values(values: List[int]) -> List[int]:
        """Find the most frequently repeated value(s) in a list of values.

        In case of ties, more than one value will be returned, sorted in increasing oder.
        """
        counter = Counter(values)
        max_count = max(counter.values())
        return sorted(value for value, count in counter.items() if count == max_count)

    def _write_frequent_phonemes(self, lang_details: List[LanguageDetails],
                                 orig_lang_details: List[LanguageDetails],
                                 consonants: bool) -> None:
        """Collect phonemes by frequency and write the more frequent ones to a file.

        Parameters:

        * lang_details: the list of source languages with variants of the same phoneme simplified
          and reduced to their base forms
        * orig_lang_details: the list of source languages with their original full phoneme mappings
          -- will be used to determine which phoneme variants are sufficiently frequent to print
        * consonants: if True, the consonant phonemes will be processed, otherwise the vowels

        Phonemes will be sorted by how often they occur among the source languages and
        alphabetically; those that occur in at least 12% of the source languages (rounded)
        will be output.

        If there are several variants of a phoneme (e.g. regular and aspirated variants of a
        consonant), those that occur in at least 20% (rounded) of the source languages will all
        printed as variants. If there is only one such variant and it's identical to the base
        form, it will be omitted.
        """
        threshold = round(len(lang_details) / 8)
        variant_threshold = round(len(lang_details) / 5)
        phoneme_type = 'consonant' if consonants else 'vowel'
        phoneme_counter = self._count_phonemes(lang_details, consonants, 'Merged ' + phoneme_type)
        orig_phoneme_counter = self._count_phonemes(orig_lang_details, consonants,
                                                    'Original ' + phoneme_type)
        # Filter for top 5 languages
        top_5_lang_details = [lang_detail for lang_detail in lang_details
                              if lang_detail.iso_code in self._top_5_langs]  # type: ignore
        top_5_phoneme_counter = self._count_phonemes(top_5_lang_details, consonants,
                                                     'Top 5 ' + phoneme_type)

        # Write output file
        with util.open_csv_writer(f'frequent_{phoneme_type}s.csv') as writer:
            writer.writerow(['Phoneme', 'Count', 'Top-5 count', 'Variants'])

            for phoneme, count in sorted(phoneme_counter.items(),
                                         key=lambda pair: (-pair[1], pair[0])):
                if count < threshold:
                    break

                top_5_count = top_5_phoneme_counter[phoneme]

                # Print any variants sufficiently frequent to reach the threshold
                variants = ''
                variant_phonemes = self._variant_map.get(phoneme)

                if variant_phonemes:
                    variant_frequencies: Counter[str] = Counter()
                    for variant in variant_phonemes:
                        variant_freq = orig_phoneme_counter[variant]
                        if variant_freq >= variant_threshold:
                            variant_frequencies[variant] = variant_freq

                    # Print variants if any reached the threshold
                    if variant_frequencies:
                        base_freq = orig_phoneme_counter[phoneme]
                        if base_freq:
                            # Also add the original frequency of the base phoneme if its above 0
                            # (it may be 0 if its a simplified spelling, e.g. 't̠ʃ' becoming 'tʃ')
                            variant_frequencies[phoneme] = base_freq
                        else:
                            # Base phoneme doesn't actually occur, so it's a simplified spelling
                            # that we replace with the most common variant (which should be the
                            # original spelling), hence e.g. restoring 'tʃ' to the phonetically more
                            # correct 't̠ʃ'
                            phoneme = variant_frequencies.most_common(1)[0][0]

                        # If more than one variant was found, we printed them, sorted first by
                        # frequency and then alphabetically
                        if len(variant_frequencies) > 1:
                            variants = ', '.join(f'{count}x {variant}' for variant, count in sorted(
                                variant_frequencies.items(), key=lambda pair: (-pair[1], pair[0])))

                writer.writerow([phoneme, count, top_5_count, variants])

    @staticmethod
    def _count_phonemes(lang_details: List[LanguageDetails], consonants: bool,
                        phoneme_type: str) -> Counter[str]:
        """Count the phonemes that occur in a list of languages.

        Set `consonants` to True to count consonants, to False to count vowels.
        Set `phoneme_type` to a textual description of the investigated phoneme type
        (will be used to log a warning if a list contains duplicates).
        """
        phoneme_counter: Counter[str] = Counter()

        for lang_detail in lang_details:
            phoneme_list = lang_detail.consonant_list if consonants else lang_detail.vowel_list
            # Make sure there are no duplicates
            phoneme_set = set(phoneme_list)
            if len(phoneme_set) != len(phoneme_list):
                logging.warning(f'{phoneme_type} list of {lang_detail.name} contains duplicates!')
            phoneme_counter.update(phoneme_set)

        return phoneme_counter


if __name__ == '__main__':
    phoible_lister = PhoibleLister()
    phoible_lister.list_sounds()

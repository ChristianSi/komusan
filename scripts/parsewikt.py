#!/usr/bin/env python3
"""Parse the Wiktionary data to create a dictionary of terms (termdict.txt).

Should be run in the data/ directory.
"""

from __future__ import annotations
import gzip
from collections import defaultdict
from dataclasses import dataclass, field
import json
from typing import Counter, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from warnings import warn

import linedict
import util
import walsfeaturefreq


##### Constants #####

# Name of the output file
TERMDICT_FILE = util.TERM_DICT

# We ignore words and word senses that have less than this number of translations
TRANS_MINIMUM = 10

# Adapt the mapping of fallback languages to fit Wiktionary.
# For Nigerian Pidgin, we use Tok Pisin as fallback (likewise English-based, second best
# represented creole).
EXTRA_FALLBACKS = walsfeaturefreq.EXTRA_FALLBACKS | {'pcm': 'tpi'}

EXTRA_NAMES = walsfeaturefreq.EXTRA_NAMES | {'tpi': 'Tok Pisin'}

# Other languages well represented in Wiktionary -- we preserve them to make our own
# multilingual dictionary even more useful
OTHER_MAJOR_LANGUAGES = frozenset('bg ca cs da el fi hu it nl pl pt ro sv uk'.split())

# Translations marked with one of these tags are skipped as unsuitable or irrelevant
TAGS_TO_SKIP = frozenset('archaic colloquial dated derogatory dialectal historical obsolete '
                         'pejorative rare regional slang uncommon vulgar'.split())

# Likewise, but listing regional variants for specific languages
DE_TAGS_TO_SKIP = frozenset('Alemannic-German Austria Austrian Bavaria Liechtenstein Palatine '
                            'Swabian Swiss Swiss-German Switzerland'.split() + ['Swabian German'])


ES_TAGS_TO_SKIP = frozenset(
    'Aragon Argentina Belize Bolivia Canary-Islands Caribbean Central-America Chile Colombia '
    'Costa-Rica Cuba Dominican-Republic Ecuador El-Salvador Guatemala Honduras Latin-America '
    'Mexico Nicaragua Panama Paraguay Peru Philippines Puerto-Rico South-America Uruguay US '
    'Venezuela'.split())

FR_TAGS_TO_SKIP = frozenset(
    'Belgium Canada Canadian-French Louisiana Luxembourg North-America Quebec Switzerland'.split())

# Set of tags indicating a US pronunciation
US_TAGS = frozenset(['US', 'General-American', 'General American'])


##### Helper functions #####

def extract_codes_and_scripts(row: Sequence[str], filename: str) -> Tuple[str, str]:
    """Extract language codes (field 2) and scripts (field 6) from our sourcelangs file."""
    if len(row) < 6:
        raise ValueError(f'CSV row too short: {len(row)} columns instead of 6 – {", ".join(row)}')
    code = row[1]
    script = row[5]
    return (code, script)


##### Types #####

@dataclass(order=True)
class Translation():
    """How a term is expressed in one specific language."""
    code: str
    word: list[str] = field(default_factory=list)
    word_set: set[str] = field(default_factory=set)  # Same as set, to avoid adding duplicates
    latin: list[str] = field(default_factory=list)  # the romanization, if applicable
    ipa: list[str] = field(default_factory=list)

    # Static translation tables
    _WORD_TRANS = str.maketrans(';[]', ',()')
    _IPA_TRANS = str.maketrans('', '', ';[]/')

    @staticmethod
    def create(code: str, word: str, latin: Optional[str] = None, ipa: Optional[str] = None):
        """Create a new instance for a single translation.

        For robustness, this factory method should be used instead of invoking the constructor
        directly.
        """
        # Replace semicolon by comma and (square) brackets by parentheses, so we can use semicolons
        # to separate multiple translations and brackets to enclose the romanization or IPA
        if word:
            word = word.translate(Translation._WORD_TRANS)
        if latin:
            latin = latin.translate(Translation._WORD_TRANS)
        # IPA is usually surrounded by [...] or /.../, which we can simply remove;
        # semicolons can also be removed should they occur (normally they shouldn't)
        if ipa:
            ipa = ipa.translate(Translation._IPA_TRANS)

        # To allow proper serialization, use empty entries for romanization and IPA to indicate
        # thy are missing
        return Translation(code, [word], set([word]), [''] if latin is None else [latin],
                           [''] if ipa is None else [ipa])

    def merge(self, other: Translation) -> None:
        """Mere another translation with this one.

        Duplicate translations (both listing the same word) are skipped
        """
        if self.code != other.code:
            raise ValueError(f'Cannot merge {self.code} translation with one for {other.code}')

        if self.word_set.issuperset(other.word_set):
            return  # Nothing to do (duplicate)

        self.word += other.word
        self.word_set |= other.word_set
        self.latin += other.latin
        self.ipa += other.ipa

    def serialize(self) -> str:
        """Convert this translation to a machine-reading string representation.

        Each word is followed by its romanization or IPA in square brackets, if given.
        Multiple translations are semicolon-separated.

        The language code is not included.
        """
        entries = []
        for idx, word in enumerate(self.word):
            extra = self.latin[idx]
            if not extra:
                extra = self.ipa[idx]
            entry = word
            if extra:
                entry += f' [{extra}]'
            entries.append(entry)
        return '; '.join(entries)


@dataclass(order=True)
class Term(linedict.ToStringDict):
    """Wraps info on a term (a specific word sense)."""
    # Lowercase form of the English translation -- automatically copied here once its added or
    # modified, in order to ensure proper ordering
    en_word_lower: str = ''
    cls: str = ''  # word class (or POS tag)
    sense: str = ''
    # Number of translations added to this Term (deliberately not reduced if 'filter_trans'
    # is invoked
    transcount: int = 0
    # Mapping from language code to translation -- we exclude it from comparisons since
    # dicts cannot be sorted automatically
    transdict: Dict[str, Translation] = field(default_factory=dict, compare=False)

    def add_trans(self, trans: Translation) -> None:
        """Add a translation to the 'transdict'.

        If another translation with the same language code exists already, both are combined
        (semicolon-separated).
        """
        if trans.code in self.transdict:
            self.transdict[trans.code].merge(trans)
        else:
            self.transdict[trans.code] = trans
            self.transcount += 1
        if trans.code == 'en':
            self.en_word_lower = ';'.join(self.transdict[trans.code].word).lower()

    def filter_trans(self, code_set: Set[str]) -> None:
        """Filter the translations, keeping only those whose codes are listed in 'code_set'."""
        self.transdict = dict((code, trans) for code, trans in self.transdict.items()
                              if code in code_set)

    def to_dict(self) -> Mapping[str, str]:
        """Convert this object into a dictionary of string/string pairs"""
        result = {}
        result['class'] = self.cls
        # We use a default sense if the entry lacks one (a few translation listings do)
        result['sense'] = self.sense if self.sense else 'Translations'
        result['transcount'] = str(self.transcount)

        for code, trans in sorted(self.transdict.items()):
            if not code:
                warn(f'{trans} lacks a language code!')
            result[code] = trans.serialize()

        return result


##### Main class #####

class WiktParser:
    """Parse the Wiktionary data to create a vocabulary."""

    def __init__(self) -> None:
        """Create a new instance."""
        # List of terms to store
        self._termlist: List[Term] = []
        # Mapping from languages to codes, used in cases where the language code is missing
        self.lang2code: Dict[str, str] = util.read_dict_from_csv_file('langcodes.csv')
        # Count for each language specified without a corresponding ISO code how often it occurred
        self.unknown_lang_counter: Counter[str] = Counter()

        # Dict of languages we're interested in (source and fallback as well as other major
        # languages)
        sourcelang_dict: Dict[str, str] = util.read_dict_from_csv_file(
            util.SOURCELANGS_FILE, converter=extract_codes_and_scripts)
        sourcelangs = set()
        # The subset of source and fallback languages that need romanization
        # (since they don't use the Latin alphabet by default)
        needs_romanization = set()

        # Some language codes still need to be split
        for code, script in sourcelang_dict.items():
            codes = code.split('/')
            sourcelangs.update(codes)
            if script != 'Latin':
                needs_romanization.update(codes)

        # Likewise for the extra fallback languages
        for origcode, code in EXTRA_FALLBACKS.items():
            codes = code.split('/')
            sourcelangs.update(codes)
            # We assume that if the main languages needs romanization, its fallback does too
            if origcode in needs_romanization:
                needs_romanization.update(codes)

        # Add the other languages and auxlangs most frequently represented in Wiktionary
        sourcelangs |= OTHER_MAJOR_LANGUAGES | util.COMMON_AUXLANGS
        self.sourcelangs = sourcelangs
        self.needs_romanization = needs_romanization

    def get_code(self, transinfo: dict) -> str:
        """Retrieve the language of the translation.

        In a few cases, the "code" field is null, but the "lang" is set. In such cases we
        convert the language into a code. If this isn't possible, "mis" for "missing" is
        returned.
        """
        code = transinfo.get('code')
        lang = transinfo.get('lang')

        if lang:
            lang = lang.strip()

        if (not code) or (code == 'en' and lang != 'English'):
            # Sometimes the code is missing or translations are mislabeled as 'en'
            code = self.lang2code.get(lang)
            if not code:
                # Set code to mis(sing) and remember that this language didn't have a code
                code = 'mis'
                self.unknown_lang_counter[lang] += 1

        # zh with 'Mandarin' in tag or lang name is equivalent to 'cmn' (for Yue/Cantonese,
        # no tags are used)
        if (code == 'zh'
                and (('tags' in transinfo and 'Mandarin' in transinfo['tags'])
                     or 'Mandarin' in lang)):
            code = 'cmn'

        return code

    @staticmethod
    def skip_trans(trans: Translation, trans_entry: dict,
                   existing_translations: Sequence[Translation]) -> bool:
        """Check if a translation should be skipped.

        This is the case

        * if any of its tags belong to the TAGS_TO_SKIP (marking words that are archaic, rare,
          slang etc.) – except when it's the first translation from a language
        * if an Arabic translation represents a local variant such as Moroccan Arabic –
          except when it's the first Arabic translation, since we want to keep at least one.
        * if a Spanish translation belongs to one of the regional variants listed in
          ES_TAGS_TO_SKIP (such as "Latin-America"), without "Spain" being listed as another
          region – except when it's the first Spanish translation; also if a Spanish word is labeled
          as "disused", as is sometimes the case
        * if a French translation belongs to one of the regional variants listed in
          FR_TAGS_TO_SKIP (such as "Canada"), without "France" being listed as another region
          – except when it's the first French translation
        * if a German translation uses a language name other than "German" (sometimes used for
          Bavarian German or other nonstandard variants) or if it belongs to one of the regional
          variants listed in DE_TAGS_TO_SKIP – except when it's the first German translation
        * if a Javanese translation has a latinization (we only keep those entries using
          the form common Roman alphabet, skipping those that use the traditional alphabet)
        * if a Korean translation is tagged "North Korea"
        * if a Malay translation is tagged "Jawi" (alternative alphabet)

        'existing_translations' are the translations with the same word sense already added.
        """
        # pylint: disable=too-many-return-statements
        tags = trans_entry.get('tags', []) + trans_entry.get('raw_tags', [])
        lang = trans_entry.get('lang')
        code = trans.code
        if (any(tag in TAGS_TO_SKIP for tag in tags)
                and any(trans.code == code for trans in existing_translations)):
            return True
        if (trans.code == 'ar'
                and (any(' Arabic' in tag for tag in tags) or any('-Arabic' in tag for tag in tags))
                and any(trans.code == 'ar' for trans in existing_translations)):
            return True
        if (trans.code == 'de' and (lang != 'German'
                                    or (any(tag in DE_TAGS_TO_SKIP for tag in tags)
                                        and any(trans.code == 'de' for trans
                                                in existing_translations)))):
            return True
        if (trans.code == 'es' and any(tag in ES_TAGS_TO_SKIP for tag in tags)
                and 'Spain' not in tags
                and any(trans.code == 'es' for trans in existing_translations)):
            return True
        if (trans.code == 'es' and (any(word.endswith('disused)') for word in trans.word))
                or any(word.startswith('disused:') for word in trans.word)):
            return True
        if (trans.code == 'fr' and any(tag in FR_TAGS_TO_SKIP for tag in tags)
                and 'France' not in tags
                and any(trans.code == 'fr' for trans in existing_translations)):
            return True
        if (trans.code == 'jv'
                and 'roman' in trans_entry
                and (trans_entry['roman'] != trans_entry.get('word', ''))):
            # Occasionally the Javanese latinization it just a copy of the word itself,
            # then we keep it (but not if it's a true latinization of the Javanese script)
            return True
        if trans.code == 'ko' and 'North Korea' in tags:
            return True
        if trans.code == 'ms' and 'Jawi' in tags:
            return True
        return False

    def add_trans_to_dict(self, trans: dict, transdict: Dict[str, list]) -> None:
        """Add a translation to a translation dictionary."""
        word = trans.get('word')
        if not word:
            # Read "note" instead, but put it in parentheses (and skip if it's not helpful)
            note = trans.get('note', '')
            if note != 'please add this translation if you can':
                word = '(' + note + ')'
        latin = trans.get('roman')
        code = self.get_code(trans)

        # Discard romanizations for non-source languages and those that use the Latin alphabet
        # anyway (the latter very occasionally have spurious romanizations nevertheless)
        if latin and code not in self.needs_romanization:
            latin = None

        if not word:
            return  # Nothing to add

        translation = Translation.create(code, word, latin)
        translist_by_sense = transdict[trans.get('sense', '')]

        if not self.skip_trans(translation, trans, translist_by_sense):
            translist_by_sense.append(translation)

    @staticmethod
    def only_in_second(val: str, first: str, second: str) -> bool:
        """Return True if 'val' is a substring of 'second', but not of 'first'."""
        return (val in second) and (val not in first)

    def find_en_ipa(self, sounds: dict, word: str) -> Optional[str]:
        """Find the IPA pronunciation of an English word.

        If the first US pronunciation seems to preserve rhotic consonants better, it is
        chosen; otherwise the very first pronunciation is returned.
        """
        very_first = None
        first_us = None

        for sound in sounds:
            if 'ipa' in sound:
                ipa = sound['ipa']
                if very_first is None:
                    very_first = ipa
                if first_us is None:
                    tags = sound.get('tags', {})
                    if set(tags) & US_TAGS:
                        first_us = ipa
                        break
        if very_first == first_us or first_us is None:
            return very_first

        if 'r' in word and (self.only_in_second('ɹ', very_first, first_us)       # type: ignore
                            or self.only_in_second('ɚ', very_first, first_us)):  # type: ignore
            # Prefer the US pronunciation if it contains one of the rhotic sounds 'ɹ' or 'ɚ'
            return first_us

        return very_first

    def process_entry(self, entry: dict) -> None:
        """Process an entry and add its word senses to our internal list, if they qualify."""
        cls = entry['pos']

        # Create English translation
        en_word = entry['word']
        en_ipa = self.find_en_ipa(entry.get('sounds', {}), en_word)

        if en_ipa:
            # In a few cases the IPA contains a spurious space after the stress marker
            en_ipa = en_ipa.replace('ˈ ', 'ˈ')

        en_trans = Translation.create(code='en', word=en_word, ipa=en_ipa)

        # Create mapping from sense to list of translations
        transdict: Dict[str, list] = defaultdict(list)
        for trans in entry.get('translations', []):
            self.add_trans_to_dict(trans, transdict)
        if "senses" in entry:
            # Translations are often listed under items in the "senses" object, so we need to
            # parse those as well
            for sense in entry["senses"]:
                for trans in sense.get('translations', []):
                    self.add_trans_to_dict(trans, transdict)

        # We only keep senses with the required number of translations
        for sense, translations in transdict.items():
            if not sense and len(transdict) > 1:
                # Skip entries without a sense, unless there are no other senses
                continue

            if len(translations) >= TRANS_MINIMUM - 1:  # English counts as well
                translations.append(en_trans)
                term = Term(cls=cls, sense=sense)
                for trans in translations:
                    term.add_trans(trans)

                # 2nd check, since translations may have been merged and hence their number reduced
                if len(term.transdict) >= TRANS_MINIMUM:
                    self._termlist.append(term)
                    if len(self._termlist) % 1000 == 0:
                        print(f'Added {len(self._termlist)} terms to term list')

    def warn_about_unknown_languages(self) -> None:
        """Warn about languages without an accompanying language code.

        Warnings are printed only for languages specified in 10+ entries. All others are
        automatically set to mis(sing) without a warning. There are many malformed entries
        containing an invalid language field, so this is a robust solution.
        """
        for lang, count in sorted(self.unknown_lang_counter.items(),
                                  key=lambda pair: (-pair[1], pair[0])):
            if count < 10:
                break
            print(f'WARNING: Language "{lang}" is mentioned {count} times and lacks a language '
                  'code – please add it to langcodes.csv')

    def build_termlist(self) -> None:
        """Build list of terms based on the kaikki.org dump."""
        print('Building list of terms – this may take some time...')
        with gzip.open(util.KAIKKI_EN_FILE + '.gz', 'rt') as infile:
            for line in infile:
                # Each line is a JSON object
                entry = json.loads(line)
                self.process_entry(entry)
                ## if len(self._termlist) > 1000: return
        print(f'Total number of terms: {len(self._termlist)}')
        self.warn_about_unknown_languages()

    def count_translations(self) -> None:
        """Optional step: Count translations into various languages and print some statistics."""
        transcounter: Counter[str] = Counter()
        ## auxlang_codes = frozenset('avk eo ia ie io jbo lfn nov vo'.split())

        for term in self._termlist:
            for langcode in term.transdict.keys():
                ## if langcode in auxlang_codes:
                transcounter[langcode] += 1

        for lang, count in sorted(transcounter.items(), key=lambda pair: (-pair[1], pair[0])):
            # Skip the languages we keep
            if lang in self.sourcelangs:
                continue
            if count < 1000:
                break
            print(f'{lang} has {count} translations')

    def discard_extra_translations(self) -> None:
        """Filter translations of each term, keeping only those that belong to source languages."""
        for term in self._termlist:
            term.filter_trans(self.sourcelangs)

    def store_termlist(self) -> None:
        """Store the term dictionary in a file."""
        util.rename_to_backup(TERMDICT_FILE)
        linedict.dump_dicts(sorted(self._termlist), TERMDICT_FILE)

    def run(self) -> None:
        """Main function: build the term dictionary."""
        self.build_termlist()
        ##self.count_translations()
        self.discard_extra_translations()
        self.store_termlist()


##### Main entry point #####

if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = WiktParser()
    parser.run()

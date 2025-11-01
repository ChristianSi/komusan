#!/usr/bin/env python3
"""Build our vocabulary, generating and selecting suitable candidate words.

This must be invoked repeatedly in the data/ directory, to add one word after the other.

Words in Wiktionary are placed in three groups: (1) nouns, (2) adjectives and adverbs, (3)
verbs and other words. By default, the word with the highest number of translations in
Wiktionary not yet in the dictionary from the currently least represented group will be added
next. (Nouns are particularly less representing in Wiktionary and many of them have a high
number of translations, so we use this grouping to avoid creating a dictionary that's made up
of nouns.) The --add command-line (CLI) option can be used to select another word for addition
instead.

Each run proposes a candidate for selection, but a suitable CLI option must be specified to
confirm or revise this choice.

The algorithm is as follows:

* Read termdict.txt.
* Read extradict.txt and merge the entries into the termdict entries that have the same
  English (en) word and sense. This can be used to supply missing translations and to
  correct translations. If entries in this file have field "tags" with the value "add",
  the entry will be added even if there is no corresponding entry in the main termdict.txt
  file – this allows supplying entries missing altogether.
* If no other word was specified, the not-yet-handled word with the highest number of
  translations becomes to next word to be added (according to the grouping explained above).
  In case of a tie, candidate proposals for generated for each word and the one with the
  alphabetically first candidate is selected.
* Candidate proposals are generated for the selected word and the combined penalty for each
  candidate is calculated, following the algorithm described in
  https://www.reddit.com/r/auxlangs/comments/mlf8h8/vocabulary_selection_for_a_worldlang/.
* Candidates are sorted by their penalty (the lower, the better). By default, the first one
  will be added, but the user has to confirm this and they also also select another one to
  be added instead -- in that case, a reason for the choice must be given.

Requires Python 3.7+ and the editdistance library.
"""
# pylint: disable=too-many-lines

import argparse
from collections import defaultdict
import csv
from enum import Enum
from datetime import datetime
from functools import lru_cache, total_ordering
import glob
from itertools import cycle
from operator import itemgetter
from os import path
import re
import sys
import textwrap
from typing import Counter, FrozenSet, List, Optional, Sequence, Set, Tuple

import editdistance

import linedict
from linedict import LineDict
from parsewikt import EXTRA_FALLBACKS
import buildutil as bu
from buildutil import Candidate
import util


##### Constants and precompiled regexes #####

# Logger
LOG = bu.LOG

# Names of input and output files
DICT_FILE = util.DICT_FILE
DICT_CSV = 'dict.csv'
SELECTION_LOG = 'selectionlog.txt'

# Dummy word used for the --polycheck argument
POLY_DUMMY = 'dummy'

# The number of entries to keep. The actual number may be somewhat higher, as we keep
# the entries that have more than a certain number of translations
TARGET_ENTRY_COUNT = 1000

# Word classes of content words (adverbs are deliberately no longer included here, since
# pure adverbs are rare and may well be short)
CONTENT_CLASSES = frozenset('adj name noun verb'.split())


# Enum for grouping all words into three kinds
@total_ordering
class Kind(Enum):
    NOUN = 1
    ADJ = 2
    VERB = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


KIND_DESC = {
    Kind.NOUN: 'a noun',
    Kind.ADJ: 'an adjective or adverb',
    Kind.VERB: 'a verb or other word',
}


##### Main class #####

class VocBuilder:
    """Build our vocabulary, generating and selecting suitable candidate words."""
    # pylint: disable=too-many-instance-attributes, too-many-public-methods

    def __init__(self, args: argparse.Namespace) -> None:
        """Create a new instance."""
        self.args = args

        # Read existing entries, if any
        self.existing_entries: Sequence[LineDict] = []
        if path.exists(DICT_FILE):
            self.existing_entries = linedict.read_dicts_from_file(DICT_FILE)

        if not (self.args.add
                or self.args.addenglish
                or self.args.addkomusan
                or self.args.delete
                or self.args.delete_only
                or self.args.polycheck
                or self.args.auxfile
                or self.args.findconcepts):
            # Which kind(s) of word should be added next
            self.kinds_to_add: FrozenSet[Kind] = self.determine_kinds_to_add(self.existing_entries)

        # Sets used for the polysemy check
        self.polyseme_dict: dict[str, Set[str]] = self.fill_polyseme_dict(self.existing_entries)
        self.langcode_sets: dict[str, Set[str]] = self.fill_langcode_sets(self.existing_entries)

        # Mapping from existing words (converted to lower-case) to their classes
        existing_words_dict: dict[str, Set[str]] = defaultdict(set)
        # Just the words, normalized to avoid minimal pairs and with affix hyphens removed
        existing_norm_words: Set[str] = set()

        for entry in self.existing_entries:
            wordlist = entry.get('word', '')
            classes = set(util.split_on_commas(entry.get('class', '')))
            for word in util.split_on_semicolons(wordlist):
                existing_words_dict[word.lower()] |= classes
                existing_norm_words.add(bu.normalize_word(word))

        self.existing_words_dict = existing_words_dict
        self.existing_norm_words = existing_norm_words

        # Candidates selected to be committed (in --auxfile mode)
        self.chosen_candidates: list[LineDict] = []

        # Komusan words to which a new meaning (concept) will be added
        self.merged_words: set[str] = set()

        # A cache of candidates to avoid having to build them twice. Keys have the form
        # "langcode:word". Note that the class is NOT part of the key. Usually this works best
        # when merging words of different classes, where the class might change from e.g.
        # "verb" to "verb, noun".
        self.candi_cache: dict[str, Optional[Candidate]] = {}
        self._init_source_languages()

        # Init mappings to convert the different orthographies and romanizations into our
        # phonology. We read all files corresponding to the "phon_*.csv" pattern since we
        # don't know which auxlangs will be requested later for conversion.
        mapping_files = glob.glob('phon_*.csv')
        convdicts = {}

        for mapping_file in mapping_files:
            lang = mapping_file.split('_')[1].split('.')[0]
            convdicts[lang] = self.read_conversion_dict(mapping_file)

        # What's the length of the longest each in each of these dictionaries?
        maxkeylengths = {}
        for name, dct in convdicts.items():
            maxkeylengths[name] = len(max(dct.keys(), key=len))

        self.convdicts = convdicts
        self.maxkeylengths = maxkeylengths

        # Used by the --addkomusan option
        self.old_word = ''
        # Used by the --auxfile option
        self.auxlangs: List[str] = []
        self.preferred_cand = -1

        # Set of currently active fallback languages
        self._active_fallbacks: set[str] = set()

        # For English (IPA): Combining tilde marks nasalization, so we change it to 'n');
        # remove other combining marks commonly used in IPA (inverted breve, bridge below,
        # bridge above, voiceless, syllabic)
        self.ipa_trans_table = str.maketrans('\u0303', 'n', '\u032f‿\u0361\u0325\u0329')
        # For Pinyin, also used for various other languages that use many diacritics
        self.tone_trans_table = str.maketrans('āēīōūǖáéíóúǘǎěǐǒǔǚàèìòùǜăĕĭŏŭâêîôû',
                                              'aeiouüaeiouüaeiouüaeiouüaeiouaeiou')
        # For Spanish and other languages
        self.acute_accents_trans_table = str.maketrans(bu.ACUTE_ACCENTS, bu.VOWELS_WO_ACCENT)
        # For Thai (combining tone marks)
        self.thai_tone_marks_trans_table = str.maketrans('', '',
                                                         '\u0302\u0300\u0301\u030C\u0304\u0312')
        # For Vietnamese
        self.vietnamese_vowel_trans_table = str.maketrans(
            'ắằẵẳấầẫẩãảạặậçếềễểëẽẻẹệĩỉịńốồỗổõỏọộqüũủứừữửựụýỳỹỷỵờớởỡợ',
            'aaaaaaaaaaaaaceeeeeeeeeiiinooooooooquuuuuuuuuyyyyyơơơơơ')

    def _init_source_languages(self) -> None:
        """Initialize the attributes keeping track of source languages."""
        sourcelang_list: List[str] = []
        main_to_fallbacks: dict[str, List[str]] = {}
        fallback_to_main: dict[str, str] = {}
        latin_main_set: set[str] = set()
        latin_fallback_set: set[str] = set()
        # Mapping from language codes to script
        sourcelang_dict: dict[str, str] = util.read_dict_from_csv_file(
            util.SOURCELANGS_FILE, converter=bu.extract_code_and_script)

        # Some language codes still need to be split into main and fallback language (e.g. hi/ur)
        for code, script in sourcelang_dict.items():
            if '/' in code:
                main, fallback = code.split('/')
                sourcelang_list.append(main)
                main_to_fallbacks[main] = [fallback]
                fallback_to_main[fallback] = main
                if script == 'Latin':
                    latin_main_set.add(main)
                    latin_fallback_set.add(fallback)
            else:
                sourcelang_list.append(code)
                if script == 'Latin':
                    latin_main_set.add(code)

        for main, fallbacks in EXTRA_FALLBACKS.items():
            # Add extra fallback languages (some of which may be themselves slash-separated)
            fallbacks_split = fallbacks.split('/')
            main_to_fallbacks[main] = fallbacks_split
            for fallback in fallbacks_split:
                fallback_to_main[fallback] = main
                # If the main language uses the Latin script, we assume the fallback does too
                if main in latin_main_set:
                    latin_fallback_set.add(fallback)

        # Init attributes
        self._sourcelang_list = sourcelang_list
        self._main_to_fallbacks = main_to_fallbacks
        self._fallback_to_main = fallback_to_main
        self._latin_main_set = latin_main_set
        self._latin_fallback_set = latin_fallback_set
        ##print('Main source languages using the Latin alphabet: '
        ##      + ', '.join(sorted(latin_main_set)))
        ##print('Fallback source languages using the Latin alphabet: '
        ##      + ', '.join(sorted(latin_fallback_set)))

        # Set of all typical source languages, including fallback languages and those
        # auxlangs used by default
        self._full_sourcelang_set = (set(sourcelang_list) | set(fallback_to_main)
                                     | util.DEFAULT_AUXLANGS)

    @staticmethod
    def get_kind(entry: LineDict) -> Kind:
        """Read the class of an entry and convert it into one of three kinds.

        * Nouns and names (proper nouns) are treated as Kind.NOUN
        * Adjectives and adverbs (quantifiers and selectors are also included here) are
          treated as Kind.ADJ
        * Verbs and words of all other classes are treated as Kind.VERB

        If a word belongs to several classes, only its first kind will be returned. So,
        "verb, noun" will be treated as Kind.VERB.
        """
        classes = util.split_on_commas(entry.get('class', ''))
        if not classes:
            word = entry.get('word', '')
            LOG.warn(f'Entry {word} lacks a class (treating as noun)')
            return Kind.NOUN

        first_class = classes[0]
        if first_class in ('noun', 'name'):
            return Kind.NOUN
        if first_class in ('adj', 'adv'):
            return Kind.ADJ
        return Kind.VERB

    def count_existing_kinds(self, entries: Sequence[LineDict]) -> dict[Kind, int]:
        """Count the existing distribution of kinds of words in our dictionary."""
        # We cannot use Python's counter class here, since 0 counts matter as well
        counter: dict[Kind, int] = {}
        for kind in KIND_DESC:
            counter[kind] = 0
        for entry in entries:
            counter[self.get_kind(entry)] += 1
        return counter

    def determine_kinds_to_add(self, entries: Sequence[LineDict]) -> FrozenSet[Kind]:
        """Determine the kinds of word to be added next.

        Words are grouped into three kinds as per the 'get_kind' method (nouns, adjectives,
        verbs). We try to ensure that all kinds are equally common in the dictionary, so this
        will return the rarest kind. If two kinds are equally rare, both will be returned.
        If all kinds are equally common, all three will be returned.
        """
        counter: dict[Kind, float] = {}
        counter.update(self.count_existing_kinds(entries))

        # Nouns only count 50% since they are much for common in the termdict, hence we allow
        # twice as many of them to be added
        counter[Kind.NOUN] *= 0.5

        result = set()
        rarest_count = -1.0

        for kind, count in sorted(counter.items(), key=lambda pair: pair[1]):
            if rarest_count < 0:
                result.add(kind)
                rarest_count = count
            else:
                if count == rarest_count:
                    result.add(kind)
                else:
                    break

        # Log choice, unless an option is specified that makes this unnecessary
        if len(result) == 3:
            msg = 'All kinds of words are equally common.'
        else:
            kinds = ''
            for kind in result:
                if kinds:
                    kinds += ' or '
                kinds += KIND_DESC[kind]
            # Add compact statistical info
            stats = '/'.join(f'{kind.name[0]}:{round(count, 1)}' for kind, count in sorted(
                counter.items(), key=lambda pair: pair[1]))
            msg = f'Should add {kinds} ({stats}).'
        LOG.info(msg)

        return frozenset(result)

    @staticmethod
    def fill_polyseme_dict(entries: Sequence[LineDict]) -> dict[str, Set[str]]:
        """Initialize and fill the dictionary used in order to check for possible polysemes.

        Polysemes are words that have several related meanings.

        Keys have the form "langcode:word" (e.g. "en:water"), values are sets of words in our
        language (e.g. "pani"). All keys are converted to lower-case (values are left as is).
        """
        result: dict[str, Set[str]] = defaultdict(set)
        for entry in entries:
            for key, translations in entry.items():
                word = entry.get('word', '')
                if len(key) > 3 or key in util.COMMON_AUXLANGS:
                    # We're only interested in field names that have at most 3 letters
                    # (language codes); auxlangs are ignored
                    continue

                # Remove IPA or romanizations in brackets
                translations = util.discard_text_in_brackets(translations)

                for trans in util.split_on_semicolons(translations):
                    if key == 'en':
                        # English words may be followed or preceded by an explanation in
                        # parentheses which we discard
                        trans = util.eliminate_parens(trans)
                    full_trans = f'{key}:{trans}'
                    result[full_trans.lower()].add(word)
        return result

    @staticmethod
    def fill_langcode_sets(entries: Sequence[LineDict]) -> dict[str, Set[str]]:
        """Initialize a set of language codes giving the translations of each word.

        Maps each Komusan word to a set of language code into which the word has been translated
        (e.g. ar, cmn, en, fr, es etc.)
        """
        result: dict[str, Set[str]] = defaultdict(set)
        for entry in entries:
            for key in entry.keys():
                word = entry.get('word', '')
                if not (len(key) > 3 or key in util.COMMON_AUXLANGS):
                    result[word].add(key)
        return result

    @staticmethod
    def read_conversion_dict(filename: str) -> dict[str, bu.Conversion]:
        """Read a dictionary of phonetic rules from a CSV file.

        See 'extract_phonetic_conversion_rule' for a description of the expected format.
        """
        result = util.read_dict_from_csv_file(filename,
                                              converter=bu.extract_phonetic_conversion_rule)
        # Add whitespace and standard punctuation so there's no need to list them explicitly
        # (unless they are already listed, e.g. '.' in IPA should be deleted instead of becoming
        # a space)
        for char in list(' -.'):
            if char not in result:
                result[char] = bu.Conversion(' ', False)
        return result

    @staticmethod
    def format_entry_key(engl: str, sense: str) -> str:
        """Construct a formatted key string from the English and sense fields of an entry."""
        return f'{engl} ({sense})'

    def mk_entry_key(self, entry: LineDict) -> str:
        """Construct a key string from an entry.

        The returned string has the form "ENGLISH (SENSE)", where "ENGLISH" is the English word
        ("en" field) and "SENSE" is the "sense" field.

        Pronunciations in brackets are stripped from the English word. A final note in
        parentheses is also stripped, unless the word consists in nothing but that note.
        """
        engl = util.discard_text_in_brackets(entry.get('en', ''))
        main_text, explanation = util.split_text_and_explanation(engl)
        if main_text:
            engl = main_text
        sense = entry.get('sense', '')
        return self.format_entry_key(engl, sense)

    @staticmethod
    def combine_entries(entry_1: LineDict, entry_2: LineDict) -> None:
        """Combine 'entry_2' into 'entry_1'.

        All fields from 'entry_2' are added to 'entry_1'. If the latter already had such a field,
        it is overwritten. Fields only in 'entry_1' are left as is.
        """
        for key, value in entry_2.items():
            entry_1.add(key, value, -1, True)

    def build_existing_entry_dict(self) -> dict[str, LineDict]:
        """Parse the existing entries into a dictionary mapping from keys to entries.

        Keys are formed as per the 'mk_entry_key' method.

        If an entry has been merged from several separate original entries (e.g. `sense:
        personal pronoun | direct object of a verb`; `en: I, me`), two separate dictionary
        entry will be created, one for each original entry. Both will point to the same
        merged entry.
        """
        result = {}

        for entry in self.existing_entries:
            engl = entry.get('en', '')
            sense = entry.get('sense', '')
            parse_word_from_sense = False

            if ' | ' in sense:
                sense_list = sense.split(' | ')
                engl_list = util.split_on_semicolons(engl)
                engl_word = None

                # When there is just one English word, it's used for all senses; otherwise
                # there should be one word per sense – if that's not the case, the intended
                # word must be added in parentheses at the end of each sense, e.g.
                # "to make a declaration (declare) | act or process of declaring (declaration)
                # | written or oral indication of a fact, opinion, or belief (declaration)"
                if len(engl_list) == 1:
                    engl_word = engl_list[0]
                elif len(sense_list) != len(engl_list):
                    parse_word_from_sense = True

                for idx, subsense in enumerate(sense_list):
                    if parse_word_from_sense:
                        # Intended word must be given at the end of each sense
                        orig_subsense = subsense
                        subsense, engl = util.split_text_and_explanation(subsense)  # type: ignore
                        if engl is None:
                            LOG.warn(f'Problem splitting merged entry: sense "{orig_subsense}" '
                                     "doesn't give the intended word in parentheses, and the "
                                     'numbers of senses and of English words differ')
                    else:
                        engl = engl_word if engl_word else engl_list[idx]
                    if subsense is not None:
                        # English word may end in an explanation in parentheses which we discard
                        engl = util.split_text_and_explanation(engl)[0]
                        result[self.format_entry_key(engl, subsense)] = entry
            else:
                # English word may end in an explanation in parentheses which we discard
                engl = util.split_text_and_explanation(engl)[0]
                result[self.format_entry_key(engl, sense)] = entry
        return result

    def sort_entries_by_transcount(self) -> dict[int, List[LineDict]]:
        """Sort entries in termdict by number of translations.

        All entries that already exist in our own dictionary are skipped. No skipping is
        performed, however, if the --copy or --polycheck option has been specified.

        Also reads extradict.txt and uses this to complete the entries before their
        translations are counted. If an extradict entry is not used, a warning is printed.
        """
        count_map: dict[int, List[LineDict]] = defaultdict(list)
        entries = linedict.read_dicts_from_file(util.TERM_DICT)
        existing_entry_dict = self.build_existing_entry_dict()
        extra_entries = linedict.read_dicts_from_file(util.EXTRA_DICT)
        extra_entry_dict = {}

        for extra_entry in extra_entries:
            extra_entry_dict[self.mk_entry_key(extra_entry)] = extra_entry

        extra_keys_seen = set()

        for entry in entries:
            entry_key = self.mk_entry_key(entry)
            filter_out_existing = not (self.args.polycheck or self.args.copy)

            # Skip if the entry already exists in our dictionary
            # (but remember the key to avoid spurious warnings about unused extradict entries)
            if filter_out_existing and (entry_key in existing_entry_dict):
                extra_keys_seen.add(entry_key)
                continue

            # Merge with extra entry, if any
            extra_entry = extra_entry_dict.get(entry_key, None)  # type: ignore
            if extra_entry:
                extra_keys_seen.add(entry_key)
                self.combine_entries(entry, extra_entry)

            transcount = int(entry['transcount'])
            count_map[transcount].append(entry)

        # Add entries with "tags: add" and warn if any unused extra entries remain
        for entry in extra_entry_dict.values():
            entry_key = self.mk_entry_key(entry)
            if entry_key not in extra_keys_seen:
                tags = entry.get('tags', '')
                if tags == 'add':
                    del entry['tags']
                    transcount = int(entry.get('transcount', '1'))
                    count_map[transcount].append(entry)
                else:
                    LOG.warn(f'Unused entry "{entry_key}" in {util.EXTRA_DICT}, line '
                             f'{entry.first_lineno()}.')

        return count_map

    def preprocess_candidate_word(self, word: str, conv_dict_name: str, cls: str) -> str:
        """Do some language-specific preprocessing on a candidate word."""
        # pylint: disable=too-many-branches, too-many-statements
        original = word
        # Convert to lower case
        word = word.lower()

        # Strip accents marking tones / diacritics
        if conv_dict_name in ('cmn', 'de', 'fr', 'ja', 'ha', 'th'):
            word = word.translate(self.tone_trans_table)

        if conv_dict_name == 'ar':
            # Strip initial 'al-' from Arabic words (provided its not the whole word)
            if word.startswith('al-') and len(word) > 3:
                word = word[3:]

        elif conv_dict_name == 'cmn':
            # Insert a '+' sign (will be deleted during conversion) between vowels and 'n' or 'r',
            # to distinguish initials such as 'fùnǚ' (fù+nǚ) from finals such as 'ānchún'
            word = bu.NR_BETWEEN_VOWELS.sub(r'\1+\2', word)

        elif conv_dict_name == 'en':
            word = word.translate(self.ipa_trans_table)

        elif conv_dict_name in ('fr', 'ja'):
            if conv_dict_name == 'fr':
                # We need to preserve the diacritic in these two cases where combinations
                # are typically pronounced differently, hence we upper-case them
                # (the conversion dictionary will handle the rest)
                word = word.replace('éa', 'ÉA')
                word = word.replace('ée', 'ÉE')
                word = word.replace('oê', 'OÊ')

            if conv_dict_name == 'fr':
                # Handle soft c and soft g and some related cases
                word = re.sub(r's?c(?=[eÉiy])', 's', word)
                word = re.sub(r'g(?=[eÉiy])', 'j', word)
                word = re.sub(r'gu(?=[eÉiy])', 'g', word)
                word = re.sub(r'[cx]c(?=[eÉiy])', 'x', word)
                word = re.sub(r'ge(?=[aoOu])', 'j', word)

                # 'ouill' becomes ultimately 'uy' (we prepare this here)
                word = word.replace('ouill', 'ouY')

                # Simplify final -ie to -i
                if word.endswith('ie'):
                    word = word[:-1]

                # Pronunciation of i/y and ou before vowels (except after another vowel)
                word = re.sub(rf'(?<![{bu.SIMPLE_VOWELS}AEÊ])[iy](?=[{bu.SIMPLE_VOWELS}ÉO])', 'Y',
                              word)
                word = re.sub(rf'(?<![{bu.SIMPLE_VOWELS}AEÊ])ou(?=[{bu.SIMPLE_VOWELS}ÉO])', 'w',
                              word)
                # But also after 'qu'
                word = re.sub(rf'(?<=qu)i(?=[{bu.SIMPLE_VOWELS}ÉO])', 'Y', word)

                # Delete final letters that are usually silent
                if (len(word) > 3
                        and word.endswith(('er', 'il')) and not word.endswith('eil')
                        and re.search(rf'[{bu.SIMPLE_VOWELS}AEOÉÊYw]', word[:-2])):
                    word = word[:-1]
                elif word.endswith('ngt'):
                    word = word[:-1]
                elif (len(word) > 2
                        and word.endswith('s')
                        and word[-2] in 'bdefgpt'
                        and not original.endswith('ès')):
                    word = word[:-2]
                elif len(word) > 1 and word[-1] in 'bdegpstxz' and not (
                        re.search('(ng|[ps]t|^.ix|^-.)$', word) or original.endswith('é')):
                    word = word[:-1]

        elif conv_dict_name == 'ha':
            # Remove combining grave accent
            word = word.replace('\u0300', '')

        elif conv_dict_name == 'hi':
            # Remove combining double tilde
            word = word.replace('\u0360', '')

        elif conv_dict_name == 'id' and cls == 'verb':
            # Strip meng- prefix (and its variants) from Indonesian verbs, see
            # https://en.wiktionary.org/wiki/meng-#Indonesian
            orig_verb = word
            if word.startswith('menge'):
                # menge -> ke, e.g. mengepung -> kepung
                word = 'ke' + word[5:]
            else:
                word = re.sub(r'^meng(?=[ghkaeiou])', '', word)
            if orig_verb == word:
                word = re.sub(r'^mem(?=[bfp])', '', word)
            if orig_verb == word:
                word = re.sub(r'^men(?=[cdjstz])', '', word)
            if orig_verb == word:
                word = re.sub(r'^me(?=[lmnrwy])', '', word)
            if orig_verb == word:
                word = re.sub(r'^meny', 's', word)
            ##if orig_verb != word:
            ##    print(f 'Note: Indonesian verb "{orig_verb}" changed to "{word}".')

        elif conv_dict_name == 'ru':
            # 'lj' is simplified to 'l', since the /j/ is very reduced
            word = word.replace('lj', 'l')
            # Russian word-final 'd' is always pronounced /t/ – we convert it accordingly,
            # since that sound is allowed to end a syllable in our phonology
            if word.endswith('d'):
                word = word[:-1] + 't'
            # Stress first vowel, if none is stressed (stressed vowels are often
            # pronounced differently, hence this matters)
            if not bu.ACUTE_ACCENT_RE.search(word):
                word = bu.VOWEL_WO_ACCENT_RE.sub(
                    lambda matchobj: bu.ACUTE_ACCENTS[bu.VOWELS_WO_ACCENT.find(matchobj.group(0))],
                    word, count=1)

        elif conv_dict_name == 'th':
            word = word.translate(self.thai_tone_marks_trans_table)

        elif conv_dict_name in ('tl', 'tr'):
            word = word.translate(self.acute_accents_trans_table)

            if conv_dict_name == 'tr':
                # Strip dot above
                word = word.replace('\u0307', '')

        elif conv_dict_name == 'vi':
            word = word.translate(self.tone_trans_table)
            word = word.translate(self.vietnamese_vowel_trans_table)

        # Remove any zero-width non-joiners
        word = word.replace('\u200C', '')
        return word

    def postprocess_candidate(self, candidate: str, original: str, conv_dict_name: str,
                              cls: str) -> str:
        """Perform language-specific postprocessing on a candidate entry, where needed.

        Returns the candidate, either as is or with the necessary changes performed.
        """
        # pylint: disable=too-many-branches, too-many-statements
        words = candidate.split()
        new_words = []

        for word in words:
            # /n/ before /g/ or /k/ is pronounced /N/ instead
            word = word.replace('ng', 'Ng')
            word = word.replace('nk', 'Nk')
            # /tS/ equals /C/ and /dj/ can be simplified to /j/
            word = word.replace('tS', 'C')
            word = word.replace('dj', 'j')

            # /N/ followed by a vowel (after optionally a second consonant) becomes /Ng/,
            # since that's how it is pronounced
            word = re.sub(rf'N(?=[{bu.SECOND_CONSONANTS}]?[{bu.INTERNAL_VOWELS}])', 'Ng', word)

            # Final /Ng/ becomes just /N/
            if word.endswith('Ng'):
                word = word[:-1]

            if conv_dict_name == 'lidepla':
                # If a word has at least three vowel letters (excluding words like "zoo"),
                # doubled vowels represent a single stressed vowel sound
                if bu.count_vowels_internal(word) >= 3:
                    word = re.sub(rf'([{bu.SIMPLE_VOWELS}])\1', r'\1', word)

            if conv_dict_name == 'de':
                # 'h' between a vowel and another consonants tends to just make the vowel
                # longer, so we remove it
                word = re.sub(rf'([{bu.SIMPLE_VOWELS}])h([{bu.ALL_CONSONANTS}])', r'\1\2', word)

                # Word-final 'er' (e.g. 'Wasser') becomes 'a' as that better represents the
                # pronunciation
                if word.endswith('er') and re.search(r'er\b', original):
                    word = word[:-2] + 'a'

            if conv_dict_name == 'en':
                # Word-final R-colored schwa as in 'water' or 'user' becomes 'a' rather than 'er',
                # thus better reflecting that its unstressed (and following the example of
                # Swahili 'picha' from 'picture'
                original = original.strip('/')
                if word.endswith('er') and (bu.FINAL_R_COLORED_SCHWA.search(original)
                                            and not original.endswith('ɛəɹ')):
                    word = word[:-2] + 'a'
                elif word.endswith('e') and original.endswith('ə'):
                    # We also convert a final schwa to 'a' if it's written like that in the
                    # original word, as it nearly always is (e.g. 'comma')
                    word = word[:-1] + 'a'

            elif conv_dict_name == 'es':
                # Word-initial 'x' is 's' instead of 'ks'
                if word.startswith('ks'):
                    word = word[1:]

                # 'i' before a vowel becomes 'y', 'u' becomes 'w'
                word = bu.IU_BEFORE_VOWEL.sub(
                    lambda matchobj: ('y' if matchobj.group(1) == 'i' else 'w') + matchobj.group(2),
                    word)

                # 'H' was used as a marker of silent letters (h) -- now, after semivowels have
                # been converted, we can delete it
                word = word.replace('H', '')
                # Strip acute accents -- we do this only now since accented vowels never become
                # semivowels
                word = word.translate(self.acute_accents_trans_table)

            elif conv_dict_name == 'fr':
                if word.endswith('tyon') and original.endswith('tYon'):
                    # Final -tion is pronounced more like -syon rather than -tyon
                    word = word[:-4] + 'syon'

            elif conv_dict_name == 'id':
                # Final 'k' usually represents the glottal stop, so we delete it
                # (but final 'nk', e.g. in 'bank', becomes 'n' rather than 'N'
                if word.endswith('Nk'):
                    word = word[:-2] + 'n'
                elif word.endswith('k'):
                    word = word[:-1]

            elif conv_dict_name == 'ru':
                if word.endswith('i'):
                    # Final 'e' is pronounced 'e'
                    if original.endswith('e'):
                        word = word[:-1] + 'e'
                    # Final 'ja' is /jə/ – here we convert the final vowel to 'a' since
                    # that seems to fit better (cf. the spelling of the romanization)
                    elif original.endswith('ja') and word.endswith('i'):
                        word = word[:-1] + 'a'

            if conv_dict_name in ('es', 'fr'):
                # Double 'oo' is reduced to one, e.g. in 'alcohol/alcool'
                word = word.replace('oo', 'o')

            if cls == 'verb':
                # Strip typical infinitive/base form markers from verbs
                if conv_dict_name == 'bn':
                    if bu.A_AFTER_FINAL_CONSONANT.search(word):
                        # Strip the final -a from the dictionary form of Bengali verbs, if possible
                        word = word[:-1]
                elif conv_dict_name == 'de':
                    if word.endswith('en'):
                        # Strip the final -n of German verbs
                        word = word[:-1]
                elif conv_dict_name in ('es', 'fr'):
                    if conv_dict_name == 'fr' and word.endswith('var') and (
                            original.endswith('avoir')
                            or original.endswith('ouvoir')
                            or original.endswith('loir')):
                        # In a handful of French verbs (avoir, mouvoir, pouvoir; falloir,
                        # prévaloir, vouloir), replace the final -oir with -e, since this
                        # sequence doesn't appear in the conjugated forms, while -e appears
                        # in the related Spanish/Portuguese verbs
                        word = word[:-3] + 'e'
                    elif conv_dict_name == 'fr' and bu.E_AFTER_FINAL_CONSONANT.search(word):
                        # The final 'er' is stripped from French verbs if it's preceded by
                        # a single consonant allowed to end a word, since the 'e' is
                        # silent in most conjugated forms – e.g. 'arrêter' becomes 'aret'.
                        # Actually the -r was already stripped in preprocessing, so we just
                        # need to strip one letter.
                        word = word[:-1]
                    elif conv_dict_name == 'fr' and word.endswith('tr'):
                        # In French verbs ending in 'tr(e)' (e.g. 'croître'), both these final
                        # consonants are silent in the typical present-tense forms
                        word = word[:-2]
                    elif word.endswith('r'):
                        word = word[:-1]
                    elif conv_dict_name == 'es' and word.endswith('rse'):
                        word = word[:-3] + 'se'
                elif conv_dict_name == 'hi':
                    if word.endswith('na'):
                        word = word[:-2]
                elif conv_dict_name == 'ko':
                    if word.endswith('da'):
                        # Strip final -da from Korean verbs
                        word = word[:-2]
                elif conv_dict_name == 'fa':
                    # Strip final -dan or -n from Persian verbs
                    if word.endswith('dan'):
                        word = word[:-3]
                    elif word.endswith('n'):
                        word = word[:-1]
                elif conv_dict_name == 'ru':
                    if word.endswith('t'):
                        word = word[:-1]
                elif conv_dict_name == 'ar':
                    # Strip the final -a from the dictionary form of Arabic verbs with three
                    # (or more) vowels if the result is allowed by Komusan's phonology (i.e.
                    # the -a is kept after a consonant that's not allowed to end a word)
                    if (bu.count_vowels_internal(word) >= 3
                            and word[-1] == 'a'
                            and (word[-2] in bu.WORD_FINAL_CONSONANTS
                                 or word[-2].lower() in bu.SIMPLE_VOWELS)
                            # Simplification of 'iy' and 'uv' happens only later
                            or word[-3:].lower() in ('iya', 'uva')):
                        word = word[:-1]
                        ##print(f 'Arabic verb changed from "{word}a" to "{word}".')
                elif conv_dict_name == 'ja':
                    # Likewise strip the final -u from the dictionary form of Japanese verbs with
                    # three (or more) vowels if the result is allowed by Komusan's phonology
                    if (bu.count_vowels_internal(word) >= 3
                            and word[-1] == 'u'
                            and (word[-2] in bu.WORD_FINAL_CONSONANTS
                                 or word[-2].lower() in bu.SIMPLE_VOWELS)
                            # Simplification of 'iy' and 'uv' happens only later
                            or word[-3:].lower() in ('iyu', 'uvu')):
                        word = word[:-1]
                        ##print(f 'Japanese verb changed from "{word}u" to "{word}".')
                elif conv_dict_name == 'sw':
                    # Swahili: strip initial ku- unless it's a short verb with just two syllables,
                    # since these tend to preserve the ku- in many cases
                    if word.startswith('ku') and bu.count_vowels_internal(word) >= 3:
                        word = word[2:]
                elif conv_dict_name in ('ta', 'te'):
                    if bu.U_AFTER_FINAL_CONSONANT.search(word):
                        # Strip the final -u from the dictionary form of Tamil, and Telugu verbs,
                        # if possible
                        word = word[:-1]
                elif conv_dict_name == 'tr':
                    if word.endswith('mak') or word.endswith('mek'):
                        # Strip final -mak/-mek from Turkish verbs
                        word = word[:-3]

            new_words.append(word)

        result = ' '.join(new_words)
        return result

    def mk_candidate(self, word: str, langcode: str, conv_dict_name: str,
                     cls: str, true_original: str | None = None) -> Optional[Candidate]:
        """Convert a word or its phonetic representation into a candidate word.

        'conv_dict_name' is the name of the conversion dictionary to use, e.g. 'en' or 'cmn'.

        A word may be followed by an explanation or comment enclosed in parentheses, which
        will be stripped.

        If 'word' is a romanization or IPA, 'true_original' should be actual original word.

        If the word is empty or completely enclosed in parentheses or if doesn't contain any
        Latin letters, None will be returned.
        """
        # pylint: disable=too-many-locals
        cache_key = f'{langcode}:{word}'
        if cache_key in self.candi_cache:
            return self.candi_cache[cache_key]

        if word.endswith(')'):
            start_idx = word.find('(')
            if start_idx >= 0:
                # Strip comment in parentheses
                word = word[:start_idx]
                word = word.strip()
        if not (word and util.has_latin_letter(word)):
            result = None
            self.candi_cache[cache_key] = result
            return result

        original = word

        if conv_dict_name not in self.convdicts:
            # For languages requested via --consider that don't have a conversion dictionary,
            # we just return the word converted to lower-case
            result = Candidate(original.lower(), 0, langcode, original, true_original,
                               self.auxlangs)
            self.candi_cache[cache_key] = result
            return result

        word = self.preprocess_candidate_word(word, conv_dict_name, cls)
        convdict = self.convdicts[conv_dict_name]
        maxkeylen = self.maxkeylengths[conv_dict_name]
        out_word = ''
        penalty = 0
        rest_word = word

        while rest_word:
            found_match = False
            # We always try to find the longest match in the convdict
            for idx in range(min(maxkeylen, len(rest_word)), 0, -1):
                conv = convdict.get(rest_word[0:idx])

                if conv is not None:
                    out_word += conv.output
                    penalty += conv.penalty
                    rest_word = rest_word[idx:]
                    found_match = True
                    break

            if not found_match:
                # Just copy the first character and warn
                out_word += rest_word[0]
                LOG.warn(f'Unexpected character "{rest_word[0]}" encountered while converting '
                         f'{langcode} candidate "{word}".')
                rest_word = rest_word[1:]

        out_word = self.postprocess_candidate(out_word, word, conv_dict_name, cls)

        #if langcode == 'fr':
        #    LOG.info(f 'Input: {word}, output: {out_word}')
        result = Candidate(out_word, penalty, langcode, original, true_original, self.auxlangs)
        self.candi_cache[cache_key] = result
        return result

    def build_candidates_for_lang(self, langcode: str, conv_dict_name: str,
                                  entry: LineDict) -> Sequence[Candidate]:
        """Build and candidate words for one specific language."""
        # pylint: disable=too-many-branches, too-many-locals
        cands = []
        raw_cand = entry.get(langcode, '')
        cls = entry.get('class', '')

        if raw_cand:
            raw_cand = raw_cand
            raw_cand_words = util.split_on_semicolons(raw_cand)
            # We keep a set of words to avoid adding duplicates
            cand_word_set: Set[str] = set()

            for cand_word in raw_cand_words:
                # If there's text in square brackets in the candidate (e.g. "water [ˈwɔtəɹ]"),
                # we use just that part -- for English, we skip any candidates that don't have
                # such text (usually that's multi-word expressions which lack IPA information)
                raw_cand_word = util.extract_text_in_brackets(cand_word, langcode != 'en')
                if not raw_cand_word:
                    continue

                if raw_cand_word != cand_word and '[' in cand_word:
                    # Sometimes there are comma-separated in the romanization, e.g. "wui⁵, wui³"
                    # -- in that case we use just the first one
                    raw_cand_word = util.split_on_commas(raw_cand_word)[0]

                    # Remember the text BEFORE the brackets as the true original (text in a
                    # non-Latin script or the actual English word)
                    true_original = cand_word.split('[', 1)[0].strip()
                else:
                    true_original = None

                candidate = self.mk_candidate(raw_cand_word, langcode, conv_dict_name, cls,
                                              true_original)
                if not candidate:
                    continue

                candidate.insert_filler_vowels()
                val_err = candidate.validate()
                if val_err:
                    LOG.warn(
                        f'{langcode} candidate "{candidate.word}" failed validation: {val_err}')
                ##else:
                ##    LOG.info(f'{langcode} candidate accepted: {candidate.word}')

                if self.args.schwastrip:
                    self._schwastrip(candidate)

                # Add candidate, if it's valid and not yet known
                if not (val_err or candidate.word in cand_word_set):
                    cand_word_set.add(candidate.word)
                    cands.append(candidate)

            #LOG.info(
            #    f'{langcode}:{raw_cand} -> {"; ".join(cand.export_word() for cand in cands)}')

        return cands

    @staticmethod
    def _schwastrip(candidate: Candidate) -> None:
        """Strip a schwa (filler vowel) from the start and/or end of this candidate."""
        if candidate.word.startswith('ə'):
            candidate.word = candidate.word[1:]
        if candidate.word.endswith('ə'):
            candidate.word = candidate.word[:-1]

    @staticmethod
    def print_frequency_distribution(intro_msg: str, counter: Counter[str]) -> None:
        """Print 'intro_msg' followed by frequency statistics from a counter."""
        LOG.info(intro_msg)
        total = sum(counter.values())
        formatted_entries = []

        for item, count in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0])):
            percentage = count / total * 100.0
            formatted_entries.append(f'{item}: {count} ({percentage:.1f}%)')

        LOG.info(util.format_compact_string_list(formatted_entries))

    @staticmethod
    def update_infl_counts(infl_counts: dict[str, float], infl_field: str) -> None:
        """Increment the influence statistics based on the "infl" field of one entry."""
        taglist = util.split_on_commas(infl_field)
        tagset: Set[str] = set()

        for tag in taglist:
            if tag in tagset:
                LOG.warn(f'Duplicated language tag in "infl" field: {tag}.')
            else:
                tagset.add(tag)

        local_infl = 1.0 / len(tagset)
        for tag in tagset:
            infl_counts[tag] += local_infl

    @staticmethod
    def print_influences(infl_dict: dict[str, float], infl_penalties: dict[str, float],
                         derived_words: int, total_words: int) -> None:
        """Print influence percentages and penalties for each language.

        Sorted first by descending percentage and then alphabetically.

        'derived_words' is the number of words derived from source languages.

        'total_words' is the total number of words in the dictionary.
        """
        LOG.info(f'{derived_words} of {total_words} entries directly derived from source '
                 'languages.')
        LOG.info('Influence distribution and penalties:')
        formatted_entries = []

        for tag, infl in sorted(infl_dict.items(), key=lambda pair: (-pair[1], pair[0])):
            percentage = infl * 100.0
            penalty = infl_penalties[tag]
            lang = tag + ':'
            formatted_entries.append(f'{lang:4} {percentage:4.1f}% (PI: {penalty:5.3f})')

        LOG.info(util.format_compact_string_list(formatted_entries))

    @staticmethod
    @lru_cache(maxsize=None)
    def calc_distance(word: str, other_word: str) -> Tuple[int, bool]:
        """Return the edit distance between two candidate words.

        Also return True if the two words are similar enough to be considered related.
        This is the case, if the ED divided by (this time) the length of the longer word is 0.5
        or less OR if the longer word starts or ends with the shorter word, provided that the
        latter has at least 2 letters (discounting any outer filler 'ə'). Additionally, if both
        words include consonants, they must share at least one consonant – just sharing vowels
        is not sufficient.
        """
        # If both words start or end with 'ə', we do the comparison without that filler vowel.
        # Otherwise e.g. "kubə" and "sabə" would be considered related, through they share
        # only one actual (non-filler) letter.
        if word.startswith('ə') and other_word.startswith('ə'):
            return VocBuilder.calc_distance(word[1:], other_word[1:])
        if word.endswith('ə') and other_word.endswith('ə'):
            return VocBuilder.calc_distance(word[:-1], other_word[:-1])

        # Edit distance is symmetric, hence we always pass the arguments in alphabetic order
        # to avoid needless recalculations
        if word > other_word:
            # pylint: disable=arguments-out-of-order
            return VocBuilder.calc_distance(other_word, word)

        # Inner 'ə' is no longer converted to 'e' since that would lead to exaggerated
        # similarities between words that have 'e' and those that actually have no vowel
        # in that position
        edist = editdistance.eval(word, other_word)
        related = edist / max(len(word), len(other_word)) <= 0.5

        if not related:
            if len(word) < len(other_word):
                shorter = word
                longer = other_word
            else:
                shorter = other_word
                longer = word
            related = len(shorter) >= 2 and (longer.startswith(shorter)
                                             or longer.endswith(shorter))

        if related:
            # Make sure that both words share a consonant, if they both include any.
            # 'N' (ng) and 'n' are considered equivalent for this comparison, since they are
            # fairly close to each other.
            word_consonants = set(word.replace('N', 'n')) & bu.ALL_CONSONANTS_SET
            other_word_consonants = set(other_word.replace('N', 'n')) & bu.ALL_CONSONANTS_SET
            if word_consonants and other_word_consonants:
                related = bool(word_consonants & other_word_consonants)

        return edist, related

    def calc_sim_penalties(self, lang: str, candidates: dict[str, Sequence[Candidate]]) -> None:
        """Calculate raw similarity penalties (PS) for candidates of one language.

        The lower the PS, the more similar a candidate is to the candidate words in the other
        languages. The result will be stored in the candidate itself; any related candidates
        are also stored in the candidate.

        Pseudo-candidates (empty language code as used by the --word option) are NOT considered
        as related.
        """
        my_cands = candidates[lang]
        for cand in my_cands:
            pen = 0
            for other_lang in candidates.keys():
                if other_lang == lang or not other_lang or not candidates[other_lang]:
                    continue
                min_dist = 1000

                for other_cand in candidates[other_lang]:
                    dist, related = self.calc_distance(cand.word, other_cand.word)
                    if dist < min_dist:
                        min_dist = dist
                    if related:
                        cand.related_cands[other_lang].append(other_cand)

                # We add the lowest distance to the penalty
                pen += min_dist

            # Store penalty in the candidate itself
            cand.raw_psim = pen

    def store_normalized_sim_penalties(self, candidates: dict[str, Sequence[Candidate]]
                                       ) -> Sequence[Candidate]:
        """Store the normalized simscore in each candidate.

        The normalized simscore (similarity score) is the raw PS (similarity penalty) inverted and
        normalized to that it will be 1.0 for the lowest (best) raw PS and 0.0 for the highest
        (worst) raw PS.

        The candidate are modified in-place; a unified list of all candidates is returned for
        convenience.

        Note that pseudo-candidates (empty language code as used by the --word option) are
        omitted from the returned list. However, those added via Add: constraint are shown as
        usual, despite not belonging to a source language.
        """
        result: List[Candidate] = []
        for lang_cands in candidates.values():
            # Hide language-less candidates added via --word option
            if lang_cands and (lang_cands[0].lang or not self.args.word):
                result += lang_cands

        if result:
            max_psim = max(result, key=lambda cand: cand.raw_psim).raw_psim
            min_psim = min(result, key=lambda cand: cand.raw_psim).raw_psim
            diff = max_psim - min_psim
        else:
            max_psim = min_psim = diff = 0

        LOG.info(f'Raw PS range: {min_psim}..{max_psim}.')
        #max_simscore = -10
        #min_simscore = 10

        for cand in result:
            if diff:
                cand.simscore = (min_psim - cand.raw_psim) / diff + 1
                #if cand.simscore > max_simscore:
                #    max_simscore = cand.simscore
                #if cand.simscore < min_simscore:
                #    min_simscore = cand.simscore
            else:
                # Special case: there is only one candidate (or all have exactly the same PS)
                cand.simscore = 1

        #LOG.info(f'Simscore range: {min_simscore:.2f}..{max_simscore:.2f}.')
        return result

    def min_length(self, entry: LineDict, constraints: bu.Constraints | None = None) -> int:
        """Determine the minimum length (in sounds) candidates should fulfill.

        For words that belong to the 'CONTENT_CLASSES', the minimum length is 3, unless the
        --allowshort option has been specified or the "Allow short" constraint is set.

        Otherwise there is no specific minimum length (1 will be returned).
        """
        if self.args.allowshort or (constraints is not None and constraints.allowshort):
            if self.args.allowshort:
                # The constraint is logged elsewhere if set, so no need to repeat that here
                LOG.info(self.format_msg('Allowing candidates that would otherwise be considered '
                                         f'too short, rationale: {self.args.allowshort}'))
            return 1
        # If the entry has multiple classes (e.g. "verb, noun"), we consider the first one
        first_class = util.split_on_commas(entry.get('class', ''))[0]
        return 3 if first_class in CONTENT_CLASSES else 1

    @staticmethod
    def cleanup_translation_value(value: str) -> str:
        """Cleanup the translations of a concept in some language for export.

        Multiple translations are supposed to be semicolon-separated.

        * Text in brackets (IPA or romanization) is removed.
        * If some of the semicolon-separated translations are identical, only the first one is kept.
        * If one of the translations is just a comment (enclosed in parentheses), it is
          removed, unless it's the only translation.
        """
        value = util.discard_text_in_brackets(value)
        translations = util.split_on_semicolons(value)
        result_list = []
        result_set = set()

        for trans in translations:
            if len(translations) > 1 and trans.startswith('(') and trans.endswith(')'):
                continue
            if trans in result_set:
                continue
            result_list.append(trans)
            result_set.add(trans)

        # Add the first translation if ALL of them were in parentheses
        if not result_list and translations:
            result_list.append(translations[0])

        return '; '.join(result_list)

    def export_entry(self, entry: LineDict) -> LineDict:
        """Prepare an entry for export into the dictionary.

        This method ensure that the fields of the returned entry will be serialized in the
        proper order:

        Order of serialization:
        1. the word itself
        2. fields that aren't translations in alphabetic order (class, sense etc.)
        3. translations in alphabetic order

        A few keys from the original entry are skipped altogether (esp. 'transcount').
        """
        # pylint: disable=too-many-branches
        if self.args.core:
            # Add "core" tag
            self.add_or_append_field('tags', 'core', entry)
        if self.args.tags:
            # Add specified tag(s)
            self.add_or_append_field('tags', self.args.tags, entry)
        if self.args.field:
            # Add custom fields
            for name, value in self.args.field:
                self.add_or_append_field(name, value, entry)

        other_keys = []
        trans_keys = []

        for key in entry:
            if key in ('transcount', 'word') or key.endswith('-ipa'):
                continue
            if len(key) <= 3 or key in self.auxlangs or '-' in key:
                trans_keys.append(key)
            else:
                other_keys.append(key)

        all_keys = sorted(other_keys) + sorted(trans_keys)
        trans_key_set = set(trans_keys)
        word = entry['word']
        cls = entry['class']

        if cls == 'name' and entry['en'][0].isupper():
            # Names (proper nouns) are capitalized if the English version is
            word = util.capitalize(word)
        elif cls == 'prefix' and not word.endswith('-'):
            # Add a hyphen after prefixes and before suffixes, if not yet present
            word = word + '-'
        elif cls == 'suffix' and not word.startswith('-'):
            word = '-' + word

        # Now create the result itself
        result = LineDict()
        result.add('word', word)
        for key in all_keys:
            value = entry[key]
            if key in trans_key_set:
                value = self.cleanup_translation_value(value)
            if value:
                # Skip pair if no value remains after processing (e.g. if the complete raw
                # value is in angle brackets)
                result.add(key, value)
        return result

    @staticmethod
    def add_or_append_field(key: str, value: str, entry: LineDict) -> None:
        """Add a key/value pair to `entry`.

        If the key already exists in the entry, the new value will be appended after the
        existing one, separated by a comma (and a space).
        """
        if key in entry:
            entry.append_to_val(key, ', ' + value)
        else:
            entry.add(key, value)

    def add_entries_to_dict(self, entries: List[LineDict], export_needed: bool = True) -> None:
        """Add the selected candidate entries to the output dictionary and write them to disk.

        Set 'entries' to an empty list to indicate that one or more entries were deleted
        (--delete option).

        If 'export_needed' is True (default), newly added entries will be exported field to
        ensure that fields are serialized in the correct order and that inappropriate fields
        are skipped. Set this to False if existing entries were modified in-place, so no
        re-export is needed.
        """
        # pylint: disable=too-many-branches, too-many-locals
        if export_needed:
            out_entries = [self.export_entry(entry) for entry in entries]
        else:
            out_entries = entries

        if self.merged_words:
            # Some of the existing entry must be skipped since they have been revised
            existing_entries = []
            for existing_entry in self.existing_entries:
                if existing_entry.get('word', '') not in self.merged_words:
                    existing_entries.append(existing_entry)
        else:
            existing_entries = list(self.existing_entries)

        out_entries += existing_entries

        # We sort first by the word itself, then by the word class, finally by the English
        # translation. If two words differ only by case, the capitalized one will be put first
        # ('Jungvo' precedes 'jungvo').
        sorted_entries = sorted(out_entries,
                                key=lambda entry: (entry.get('word', '').lower(),
                                                   entry.get('word'),
                                                   entry.get('class', '').lower(),
                                                   entry.get('en', '').lower()))

        # Add the words to the dictionary
        util.rename_to_backup(DICT_FILE)
        linedict.dump_dicts(sorted_entries, DICT_FILE)

        # Also append new rows to the CSV file – unless an option was used which means that
        # we have to (potentially) modify or delete some existing entries instead of adding a
        # one
        util.copy_to_backup(DICT_CSV)
        first_entry = out_entries[0] if entries else None

        if (self.args.addmeaning or self.args.addenglish or self.args.addkomusan) and first_entry:
            with open(DICT_CSV, newline='', encoding='utf8') as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)

                if self.args.addkomusan:
                    # The Komusan word has changed
                    old_word = self.old_word
                    new_word = first_entry.get('word', '')
                    en_word = first_entry.get('en', '')

                    for idx, line in enumerate(lines):
                        if line[1] == old_word:
                            lines[idx] = [en_word, new_word]
                            break
                    else:
                        LOG.warn(f'Entry "{old_word}" not found in {DICT_CSV}')
                else:
                    en_word = first_entry.get('en', '')
                    our_word = first_entry.get('word', '')

                    for idx, line in enumerate(lines):
                        if line[1] == our_word:
                            lines[idx] = [en_word, our_word]
                            break
                    else:
                        LOG.warn(f'Entry "{en_word}" – "{our_word}" not found in {DICT_CSV}')

            # Reopen file for writing
            with open(DICT_CSV, 'w', newline='', encoding='utf8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(lines)

        elif self.args.delete or self.args.delete_only:
            words_to_keep = set()
            for out_entry in out_entries:
                words_to_keep.add(out_entry.get('word', ''))
            with open(DICT_CSV, newline='', encoding='utf8') as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                lines_to_keep = []

                for line in lines:
                    if line[1] in words_to_keep:
                        lines_to_keep.append(line)

            # Reopen file for writing
            with open(DICT_CSV, 'w', newline='', encoding='utf8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(lines_to_keep)

        elif self.merged_words:
            # Some existing words may have acquired additional English meanings, which we add
            # in place. Key is the Komusan word, value the English one.
            pairs = {out_entry.get('word', ''): out_entry.get('en', '')
                     for out_entry in out_entries}
            new_lines: list[list[str]] = []
            words_seen = set()

            with open(DICT_CSV, newline='', encoding='utf8') as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)

                for line in lines:
                    word = line[1]
                    words_seen.add(word)

                    if word in self.merged_words:
                        # Replace with merged entry
                        new_lines.append([pairs.get(word, ''), word])
                    else:
                        # Keep entry as is
                        new_lines.append(line)

            # Add all new (not yet seen) words after their English translations
            for word, english in pairs.items():
                if word not in words_seen:
                    new_lines.append([english, word])

            # Reopen file for writing
            with open(DICT_CSV, 'w', newline='', encoding='utf8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(new_lines)

        elif entries:
            with open(DICT_CSV, 'a', newline='', encoding='utf8') as csvfile:
                writer = csv.writer(csvfile)
                for out_entry in out_entries[:len(entries)]:
                    # Add new entries at end of file
                    writer.writerow([out_entry.get('en'), out_entry.get('word')])

    def add_entry_to_dict(self, entry: Optional[LineDict], export_needed: bool = True) -> None:
        """Add the selected candidate entry to the output dictionary and write it to disk.

        Set 'entry' to None to indicate that one or more entries were deleted (--delete option).

        If 'export_needed' is True (default), newly added entries will be exported field to
        ensure that fields are serialized in the correct order and that inappropriate fields
        are skipped. Set this to False if existing entries were modified in-place, so no
        re-export is needed.
        """
        self.add_entries_to_dict([] if entry is None else [entry], export_needed)

    @staticmethod
    def protect_ch_sh(word: str) -> str:
        """Convert 'ch' and 'sh' to upper case to protect them against seeming similar to c/s."""
        return word.replace('ch', 'CH').replace('sh', 'SH')

    def combine_entry(self, cand: Candidate, entry: LineDict) -> LineDict:
        """Combine a candidate an entry into a unified entry.

        The entry is modified in-place and also returned for convenience.
        """
        # Build influence list; order:
        # - first the donor language and any other languages that yield the same candidate,
        #   in alphabetic order
        # - then any other related languages, also in alphabetic
        first_set = cand.find_langs_with_identical_candidate()
        infl_set = set(cand.related_cands.keys())

        if cand.lang:
            # Will be empty (and is skipped) if the --word option is used
            first_set.add(cand.lang)
            infl_set.add(cand.lang)

        second_set = infl_set - first_set
        extra_originals: dict[str, str] = {cand.lang: cand.show_original}

        # Check if languages written in the Latin alphabet should be added to the list of
        # influences based on the spelling – e.g. 'relasi' is clearly related to 'relation',
        # but because the English pronunciation is so different, the latter is nevertheless not
        # listed in the related_cands; likewise for 'historia' and French 'histoire' or for
        # '-aje' in Komusan and Spanish. However, we convert 'ch' and 'sh' to upper case since
        # otherwise e.g. 'c' pronounced /k/ or /s/ would seem similar to 'ch' /tS/ which it
        # really isn't.
        this_word = self.protect_ch_sh(cand.export_word().lower())
        lang_codes_to_check = self._active_fallbacks & self._latin_fallback_set
        lang_codes_to_check |= self._latin_main_set

        for lang_code in lang_codes_to_check:
            if lang_code in infl_set:
                continue
            enfr_words = util.split_on_semicolons(entry.get(lang_code, ''))
            for enfr_word in enfr_words:
                enfr_word = util.discard_text_in_brackets(enfr_word)
                if self.calc_distance(this_word, self.protect_ch_sh(enfr_word.lower()))[1]:
                    second_set.add(lang_code)
                    extra_originals[lang_code] = enfr_word
                    break

        # After each source language, we include the original word(s) in parentheses
        infl_list = sorted(first_set) + sorted(second_set)
        formatted_infl_list = []

        for lang in infl_list:
            if lang in extra_originals:
                originals = extra_originals[lang]
            else:
                originals = ', '.join(
                    this_cand.show_original for this_cand in cand.related_cands[lang])
            if originals:
                formatted_infl_list.append(f'{lang} ({originals})')
            else:
                LOG.warn(f'No original found for {lang}.')
                formatted_infl_list.append(lang)

        entry.add('infl', '; '.join(formatted_infl_list))
        entry.add('word', cand.export_word())
        return entry

    @staticmethod
    def format_msg(msg: str) -> str:
        """Format a message by line wrapping it and adding a final dot, if missing."""
        if not msg:
            raise ValueError('format_msg: msg is empty!')
        # Append final punctuation, if missing
        if not msg[-1] in '.!?':
            msg += '.'
        return textwrap.fill(msg, width=78, subsequent_indent='  ')

    def validate_equals_gloss(self, word: str, gloss: str) -> Optional[str]:
        """Validate an '=' gloss such as '=Jungvo' for 'jungvo'.

        Returns a error message if validation fails, otherwise None.
        """
        rest = gloss[1:]
        if not rest.lower() in self.existing_words_dict:
            return f'Word "{rest}" doesn\'t exist in the dictionary'
        if rest.lower() != word.lower():
            return f'"{word}" and "{rest}" are different words'
        return None

    def check_gloss_parts(self, parts: Sequence[str]) -> Tuple[Optional[str], bool, bool]:
        """Check that all parts of a gloss exist.

        Returns a tuple of three values:

        1. An error message if validation has failed, otherwise None
        2. True iff the first part is a prefix
        3. True iff the last part is a suffix
        """
        last_idx = len(parts) - 1
        errmsg = None
        first_is_prefix = False
        last_is_suffix = False

        for idx, part in enumerate(parts):
            if not part:
                errmsg = 'part is empty'
                break
            if not part.lower() in self.existing_words_dict:
                # Check if it's an affix
                if idx == 0:
                    alt_part = part + '-'
                    first_is_prefix = True
                elif idx == last_idx:
                    alt_part = '-' + part
                    last_is_suffix = True
                else:
                    alt_part = ''
                if not (alt_part and alt_part.lower() in self.existing_words_dict):
                    errmsg = f'Part "{part}" doesn\'t exist in the dictionary'
                    break
        return errmsg, first_is_prefix, last_is_suffix

    @staticmethod
    def check_that_parts_fit(word: str,
                             parts: List[str],
                             first_is_prefix: bool,
                             last_is_suffix: bool) -> Optional[str]:
        """Check that the joined parts of a glass fit the actual word.

        Returns an error message if this is not the case, None otherwise.
        """
        joined_parts = ''.join(parts)
        # We delete spaces from the gloss, since e.g. "des sis+nari" becomes "dessisnari"
        # (keeping the space in such cases would be confusing)
        spaceless_parts = joined_parts.replace(' ', '').lower()
        if spaceless_parts != word.lower():
            return f'gloss doesn\'t correspond to word "{word}" (expected "{spaceless_parts}")'
        return None

    def validate_and_store_gloss(self, word: str, gloss: str, entry: LineDict) -> None:
        """Validate and store the gloss documenting how a compound is formed.

        The gloss should be made up two or more parts separated by '+'. All parts must already
        exist in the dictionary and together they must add up to 'word'. Dies with an error if
        this is not the case.

        As a special case, '=' followed by a single part can be used by add a word that is
        identical to another word except for a change in case (e.g. 'jungvo' is glossed as
        '=Jungvo').

        If some letters are written in parentheses, they are omitted from the compound.
        For example, "Merkur(i)+den" indicates that "Merkuri" + "den" are put together to form
        the shortened compound "merkurden".

        Alternatively, an informal explanation such as "contraction of ..." can be given
        (containing spaces and no plus sign). In such cases, no validation is performed.

        After validation, the gloss is added as 'gloss' field to 'entry'.
        """
        errmsg = None
        if util.gloss_is_informal(gloss):
            LOG.info(self.format_msg(f'Adding informal gloss "{gloss}" without validation.'))
        elif gloss.startswith('='):
            errmsg = self.validate_equals_gloss(word, gloss)
        else:
            if '(' in gloss:
                # Eliminate parentheses, but not their content, to look up the parts
                gloss_wo_parens = gloss.replace('(', '').replace(')', '')
                parts_to_lookup = gloss_wo_parens.split('+')
                # But to combine the parts into a whole, the text in the parentheses is removed
                # as well
                gloss_wo_text_in_parens = util.eliminate_parens(gloss)
                parts_to_combine = gloss_wo_text_in_parens.split('+')
            else:
                parts_to_lookup = gloss.split('+')
                parts_to_combine = parts_to_lookup

            if len(parts_to_lookup) == 1:
                errmsg = "gloss must have two or more parts ('+' separated)"
            else:
                errmsg, first_is_prefix, last_is_suffix = self.check_gloss_parts(parts_to_lookup)
            if not errmsg:
                errmsg = self.check_that_parts_fit(word, parts_to_combine,
                                                   first_is_prefix, last_is_suffix)
        if errmsg:
            sys.exit(f'error validating gloss "{gloss}": {errmsg}')
        entry.add('gloss', gloss)

    def validate_compound(self, word: str, entry: LineDict) -> None:
        """Ensure that a compound is valid.

        "..." may be used to indicate a gap in multi-word expressions, e.g. "da ... su".

        Dies with an error if validation fails because

        * one of the compound parts doesn't exist
        * the compound isn't made of several (hyphen or whitespace-separated) elements
        * it or a minimal pair of it already exists in the dictionary
        """
        # Some phrases end in a punctuation symbol
        if word and word[-1] in ('?', '!'):
            word = word[:-1]

        # Accept whitespace and '-' as separators
        parts = re.split(r'\s+|-', word)
        last_part_idx = len(parts) - 1

        # Make sure that all parts exist -- all except the last one may also be prefixes
        # (ending in -) and all except the first one may be suffixes (starting with -)
        missing_parts = []
        for idx, part in enumerate(parts):
            if part == '...':
                continue
            lower_part = part.lower()
            variants = {lower_part}
            if idx != last_part_idx:
                variants.add(lower_part + '-')  # It might be a prefix
            if idx != 0:
                variants.add('-' + lower_part)  # It might be a suffix
            if variants.isdisjoint(self.existing_words_dict.keys()):
                missing_parts.append(part)

        if missing_parts:
            if len(missing_parts) > 1:
                missing_str = ('Parts ' + ', '.join(f'"{part}"' for part in missing_parts)
                               + " don\'t")
            else:
                missing_str = f'Part "{missing_parts[0]}" doesn\'t'
            sys.exit(f"validation error: {missing_str} exist in the dictionary (or it's an affix "
                     'used in the wrong position)')

        # Make sure that a word or a minimal pair of it isn't yet in the dictionary
        norm_word = bu.normalize_word(word)
        if (norm_word in self.existing_norm_words
                and not (self.args.allowduplicates or entry.get('gloss', '').startswith('='))):
            sys.exit(f'validation error: compound "{word}" (normalized: {norm_word}) already '
                     'exists in the dictionary')

    def add_as_compound(self, entry: LineDict, word: str | None = None,
                        rationale: str | None = None) -> None:
        """Add the entry as a compound word as requested by the --compound argument.

        In Komusan, compounds always take hyphens between parts, e.g. "mi" and "su" form the
        compound "mi-su".

        If 'word' is None, it and rationale are read from the --compound argument instead.

        If 'rationale' is None, empty, or '-', no rationale is logged. This is useful for
        self-evident cases.

        Dies with an error if the parts of the compound don't yet exist in the dictionary.
        """
        if word is None:
            word, rationale = self.args.compound
        word = word.strip()
        if rationale:
            rationale = rationale.strip()
        self.validate_compound(word, entry)
        entry.add('word', word)
        self.add_entry_to_dict(entry)
        en_word = entry.get('en', '')
        full_rationale = '' if not rationale or rationale == '-' else f', rationale: {rationale}'

        if self.args.compound:
            # If the --compound argument was passed, we're done now
            LOG.info(self.format_msg(f'Added compound "{word}" ({en_word}) to the dictionary on '
                                     + str(datetime.now())[:19]
                                     + full_rationale))
            # Append all print and warn messages to the selection log
            LOG.append_all_messages(SELECTION_LOG)

    def prepare_add_as_meaning(self, entry: LineDict, merge_with: str,
                               merge_rationale: str | None = None,
                               premerge: bool = False) -> tuple[LineDict, str]:
        """Prepare to add this entry to the meaning of another word which already exists.

        If `premerge` is true, new translations re added before the existing ones, otherwise
        they are added after them.

        Returns a two-tuple:

        1. The resulting combined entry
        2. The rationale for merging, read from the 'merge_rationale' parameter,
           the -amr CLI argument or else (if neither is set) the results of the polysemy check
        """
        existing_entry = None
        other_entries = []

        for ex_en in self.existing_entries:
            if ex_en.get('word', '') == merge_with:
                existing_entry = ex_en
            else:
                other_entries.append(ex_en)

        if not existing_entry:
            sys.exit(f'error: Word "{merge_with}" doesn\'t exist in the dictionary')

        self.existing_entries = other_entries
        entries_ordered = [entry, existing_entry] if premerge else [existing_entry, entry]
        combined_entry = self.do_merge_entries(*entries_ordered)
        matchdict, entry_langs = self.do_polysemy_check(entry)
        langset = matchdict[merge_with]
        word_langs = self.langcode_sets.get(merge_with, set())
        shared_lang_count = len(entry_langs.intersection(word_langs))
        percentage = len(langset) / shared_lang_count * 100.0
        if merge_rationale:
            rationale = 'rationale: ' + merge_rationale
        elif self.args.amr:
            rationale = 'rationale: ' + self.args.amr
        else:
            rationale = (f'because {len(langset)} of {shared_lang_count} languages '
                         f'({percentage:.1f}%) use the same word for both concepts')
        return combined_entry, rationale

    def add_as_meaning(self, entry: LineDict) -> None:
        """Add this entry to the meaning of another word which already exists."""
        word = self.args.addmeaning.strip()
        combined_entry, rationale = self.prepare_add_as_meaning(entry, word)
        LOG.info(self.format_msg(f'Adding this to the meaning of "{word}" as requested, '
                                 + rationale))
        self.add_entry_to_dict(combined_entry)
        LOG.info('Dictionary updated on ' + str(datetime.now())[:19] + '.')
        # Append all print and warn messages to the selection log
        LOG.append_all_messages(SELECTION_LOG)

    def find_and_remove_komusan_entry(self, word: str) -> LineDict:
        """Find and retrieve the existing entry for the Komusan word 'word'.

        The entry is removed from the dictionary and returned. This allows modifying and then
        re-adding it, if desired.

        Dies with an error if the entry does not exist in the dictionary.
        """
        entry = None
        other_entries = []

        for ex_en in self.existing_entries:
            if ex_en.get('word', '') == word:
                entry = ex_en
            else:
                other_entries.append(ex_en)

        if not entry:
            sys.exit(f'error: Word "{word}" doesn\'t exist in the dictionary')
        self.existing_entries = other_entries
        return entry

    def add_english(self) -> None:
        """Add one of more words to the English translation of an existing entry.

        This implements the --addenglish option.
        """
        # pylint: disable=too-many-locals
        komusan, english = self.args.addenglish
        komusan = komusan.strip()
        english = english.strip()
        entry = self.find_and_remove_komusan_entry(komusan)

        # If the whole new text is wrapped in parentheses, it's just an explanation
        is_explanation = english.startswith('(') and english.endswith(')')
        sep = ' ' if is_explanation else ', '
        orig_engl = entry.get('en', '')
        new_engl = orig_engl + sep + english
        entry.add('en', new_engl, allow_replace=True)

        # If there are new translations, we set the sense of each of them to '-'
        if not is_explanation:
            # Check if we need to add each word to the sense to allow disambiguation
            orig_engl_split = util.split_on_commas(orig_engl)
            orig_sense = entry.get('sense', '')
            orig_sense_split = orig_sense.split(' | ')

            if len(orig_engl_split) != len(orig_sense_split):
                # Sense disambiguation is necessary since the numbers of existing words and
                # senses differ
                mod_subsenses = []
                for idx, subsense in enumerate(orig_sense_split):
                    # Old senses: add word in parentheses if it doesn't yet seem to be there
                    if util.split_text_and_explanation(subsense)[1] is None:
                        if idx >= len(orig_engl_split):
                            # Re-use the last original word for this sense
                            matching_en_word = orig_engl_split[-1]
                        else:
                            matching_en_word = orig_engl_split[idx]
                        mod_subsense = f'{subsense} ({matching_en_word})'
                    else:
                        mod_subsense = subsense
                    mod_subsenses.append(mod_subsense)
                initial_sense = ' | '.join(mod_subsenses)
                new_sense_entry = '- (-)'
            else:
                new_sense_entry = '-'
                initial_sense = orig_sense

            new_trans_count = len(util.split_on_commas(english))
            new_senses = (' | ' + new_sense_entry) * new_trans_count
            joint_sense = initial_sense + new_senses
            entry.add('sense', joint_sense, allow_replace=True)

        self.add_entry_to_dict(entry, False)
        LOG.info(self.format_msg(f'Added "{english}" to the English translation of "{komusan}" '
                                 f'on {util.current_datetime()}'))
        LOG.append_all_messages(SELECTION_LOG)

    def add_komusan(self) -> None:
        """Add a synonym to an existing Komusan entry.

        This implements the --addkomusan option.
        """
        old_word, new_word, rationale = self.args.addkomusan
        old_word = old_word.strip()
        new_word = new_word.strip()
        entry = self.find_and_remove_komusan_entry(old_word)

        # If it includes spaces, check that new_word is a valid compound
        if ' ' in new_word:
            self.validate_compound(new_word, entry)
        if self.args.gloss:
            self.validate_and_store_gloss(new_word, self.args.gloss.strip(), entry)

        joint_word = f'{new_word}; {old_word}' if self.args.first else f'{old_word}; {new_word}'
        entry.add('word', joint_word, allow_replace=True)
        self.old_word = old_word  # storing this since we'll need to adjust the csv file
        self.add_entry_to_dict(entry, False)
        position = 'before' if self.args.first else 'after'

        # Rationale may be omitted if a gloss is given
        rationale_msg = '' if self.args.gloss and rationale == '-' else f'; rationale: {rationale}'
        gloss_info = f' ({self.args.gloss})' if self.args.gloss else ''

        LOG.info(self.format_msg(f'Added "{new_word}"{gloss_info} as a synonym {position} '
                                 f'"{old_word}" on {util.current_datetime()}{rationale_msg}'))
        LOG.append_all_messages(SELECTION_LOG)

    def add_to_dict(self, cand_entry: LineDict, choice: int) -> None:
        """Add the selected candidate entry to the output dictionary.

        If choice is set to -1, this signals that the word specified through the --word option
        was used instead.

        Also adds a log of everything that happened to the selectionlog file.
        """
        self.add_entry_to_dict(cand_entry)
        if choice == -1:
            word_rationale = self.args.word[1].strip()
            LOG.info(self.format_msg(f'Specified word "{cand_entry["word"]}" added to the '
                                     f'dictionary on {util.current_datetime()}, rationale: '
                                     + word_rationale))
        else:
            LOG.info(f'Candidate #{choice} "{cand_entry["word"]}" added to the dictionary on '
                     + util.current_datetime() + '.')
            if self.args.sr:
                LOG.info(self.format_msg(f'Selection rationale: {self.args.sr}'))

        # Append all print and warn messages to the selection log
        LOG.append_all_messages(SELECTION_LOG)

    def print_cands(self, cand_list: Sequence[Candidate], min_length: int,
                    constraints: bu.Constraints | None = None) -> Sequence[Candidate]:
        """Returned an ordered list of candidates for selections.

        Candidates are sorted first by the number of related candidates in natural languages,
        then by the total score. Alphabetic order is used as tiebreaker.

        A number is added before each candidate that is not SKIPPED, and all these candidates will
        be added to the result list.

        Candidates with a distortion score of 0.6 or lower are SKIPPED. Candidates may also be
        SKIPPED due to constraints.

        The candidate with the best total penalty will be prefixed with a star.
        """
        if not cand_list:
            return []  # Nothing to print
        best_score = max(cand_list, key=lambda cand: cand.total_score).total_score

        result = []
        num = 1
        possible_failure = ''

        for cand in sorted(cand_list,
                           key=lambda cand: (-cand.count_related_natlang_cands(),
                                             -cand.total_score,
                                             cand.export_word())):
            norm_word = bu.normalize_word(cand.export_word())
            star = '*' if cand.total_score == best_score else ' '

            if constraints is not None:
                possible_failure = constraints.fails(cand)

            if norm_word in self.existing_norm_words and not self.args.allowduplicates:
                prefix = f'[SKIPPED (word exists already)]{star}  '
            elif len(cand.word) < min_length:  # Length in sounds, not letters
                prefix = f'[SKIPPED (too short)]{star} '
            elif cand.dscore <= 0.6:
                prefix = f'[SKIPPED (too distorted)]{star} '
            elif possible_failure:
                prefix = f'[SKIPPED ({possible_failure})]{star} '
            else:
                prefix = f'[{num}]{star} '
                num += 1
                result.append(cand)

            LOG.info(prefix + cand.show_info())

        return result

    def remember_word(self, word: str, entry: LineDict) -> None:
        """Add word to the sets of existing words, making it ineligible in further selections."""
        classes = set(util.split_on_commas(entry.get('class', '')))
        self.existing_words_dict[word.lower()] |= classes
        self.existing_norm_words.add(bu.normalize_word(word))

    def present_cands_for_selection(self, cand_list: Sequence[Candidate],
                                    candidates: dict[str, Sequence[Candidate]], entry: LineDict,
                                    constraints: bu.Constraints | None = None) -> None:
        """Print the list of candidates in order of preferences.

        Numbers are added before each eligible candidate, allowing the user to make their choice.

        Candidates are sorted by total score, with the number of related candidates as tiebreaker.

        Candidates will be marked as SKIPPED and not numbered if

        * they are identical to a word that already exists in the dictionary
        * they are too short (less than 3 sounds for content words)

        Only candidates that aren't SKIPPED can be selected.

        If no choice was made by the user, this will also print the entry that would result if
        the first candidate is selected. Or if a choice was already made, that entry will be
        written to the output file and the choice recorded in the selection log, completing the
        selection process.
        """
        if self.args.allowduplicates:
            LOG.info('Allowing candidates that are duplicates of already existing words as '
                     'requested.')

        # The constraints object can overwrite the target class
        if constraints is not None and constraints.target_class:
            entry.add('class', constraints.target_class, allow_replace=True)

        min_length = self.min_length(entry, constraints)
        sorted_eligible_list = self.print_cands(cand_list, min_length, constraints)

        if self.args.select is not None:
            choice = self.args.select
            if choice <= 0 or choice > len(sorted_eligible_list):
                sys.exit(f'error: Choose a candidate in the range from 1 to '
                         f'{len(sorted_eligible_list)}, not {choice}')
            if choice > 1 and not self.args.sr:
                sys.exit('error: You must specify a rationale for your choice (-sr option)')

            # Add selected candidate to the dictionary
            cand_entry = self.combine_entry(sorted_eligible_list[choice - 1], entry)
            self.add_to_dict(cand_entry, choice)
        elif self.args.word:
            # Add specified word to the dictionary
            self.store_specified_word(candidates, entry)
        elif constraints and constraints.compound:
            # Validate the requested compound and print it, or prepare it for commit
            compound = constraints.compound
            self.validate_compound(compound, entry)
            entry.add('word', compound)

            if self.args.commit:
                LOG.info(f'Choosing compound "{compound}".\n')
                self.chosen_candidates.append(entry)
            else:
                LOG.info(
                    '++ The following entry will result if the specified compound is added:')
                cand_entry = self.export_entry(entry)
                LOG.info(linedict.stringify_dict(cand_entry).strip())

            # Remember that the chosen word exists and so is ineligible in further selections
            self.remember_word(compound, entry)
        elif constraints and constraints.merge_with:
            # Merging this meaning with an existing word, following a recommendation by the
            # polysemy check
            combined_entry, rationale = self.prepare_add_as_meaning(
                entry, constraints.merge_with, constraints.merge_rationale, constraints.premerge)
            merge_type = 'Premerging' if constraints.premerge else 'Merging'

            if self.args.commit:
                LOG.info(
                    f'{merge_type} with "{constraints.merge_with}" as requested {rationale}.\n')
                self.chosen_candidates.append(combined_entry)
                self.merged_words.add(constraints.merge_with)
            else:
                LOG.info(
                    f'++ {merge_type} with "{constraints.merge_with}" as requested ({rationale}) '
                    'will result in the following entry:')
                cand_entry = self.export_entry(combined_entry)
                LOG.info(linedict.stringify_dict(cand_entry).strip())
        else:
            # Print sample entry or prepare it for commit
            if sorted_eligible_list:
                chosen_cand = None

                if constraints and constraints.added:
                    # Select the candidate added per constraint
                    for pos, cand in enumerate(sorted_eligible_list):
                        if not cand.lang:
                            # The added candidate has an empty language field, so this is the one
                            chosen_cand = cand
                            chosen_pos = pos
                            choice_info = 'added candidate'
                            break
                elif self.auxlangs:
                    # Select best auxlang candidate, if its total score is >= 0.5
                    # (but skipping those with no related natlang candidates)
                    min_ts = 0.5
                    for pos, cand in enumerate(sorted_eligible_list):
                        identical_count = len(cand.find_langs_with_identical_candidate())
                        if (cand.lang in self.auxlangs and cand.total_score >= min_ts
                                and cand.has_suitable_related_natlang_cands()):
                            chosen_cand = cand
                            chosen_pos = pos
                            choice_info = 'best eligible auxlang candidate'
                            break
                        elif identical_count >= 3 and cand.total_score >= min_ts:
                            # Candidates shared by 3 or more (natural) languages are likewise
                            # eligible, provided their total score is >= 0.5
                            chosen_cand = cand
                            chosen_pos = pos
                            choice_info = 'first eligible candidate shared by 3+ languages'
                            break

                    if not chosen_cand:
                        LOG.info(f'No eligible auxlang candidate with a total score >= {min_ts} '
                                 'and related candidates found – will use the best natlang '
                                 'candidate instead.')

                if chosen_cand is None:
                    # Choose the first entry (possibly just as example)
                    chosen_cand = sorted_eligible_list[0]
                    chosen_pos = 0
                    choice_info = 'first eligible candidate'

                combined_entry = self.combine_entry(chosen_cand, entry)
                chosen_word = chosen_cand.export_word()

                if self.args.commit:
                    LOG.info(f'Choosing #{chosen_pos + 1}: {chosen_word}, the {choice_info}.\n')
                    self.chosen_candidates.append(combined_entry)
                else:
                    LOG.info(f'++ The following entry will result if #{chosen_pos + 1}, the '
                             f'{choice_info}, is selected:')
                    cand_entry = self.export_entry(combined_entry)
                    LOG.info(linedict.stringify_dict(cand_entry).strip())

                # Remember that the chosen word exists and so is ineligible in further selections
                self.remember_word(chosen_word, entry)
            else:
                LOG.info('++ Word has no eligible candidates!')

    def add_specified_word(self, candidates: dict[str, Sequence[Candidate]], word: str,
                           source: str) -> None:
        """Add the candidate specified by the --word option or an Add: constraint.

        'word' is the word to add.

        'source' is a short description of its origin, e.g. "--word argument" or
        "Add: constraint".

        An empty language code is used as key.

        This also validates that the word is phonetically valid and doesn't yet exist in the
        dictionary.
        """
        cand = Candidate(word, 0, '', word, None, self.auxlangs)
        val_err = cand.validate()
        if val_err:
            sys.exit(f'error: {source} is phonetically invalid: {val_err}')
        norm_word = bu.normalize_word(cand.export_word())
        if norm_word in self.existing_norm_words and not self.args.allowduplicates:
            sys.exit(f'error: {source} "{cand.export_word()}" (normalized: {norm_word}) '
                     'exists already in the dictionary')
        candidates[''] = [cand]

    def build_candidates(self, entry: LineDict, constraints: bu.Constraints | None = None
                         ) -> dict[str, Sequence[Candidate]]:
        """Build and return candidate words for each languages."""
        self._active_fallbacks.clear()
        candidates: dict[str, Sequence[Candidate]] = {}
        lang_codes = self._sourcelang_list

        if self.args.consider:
            extra_lang_codes = util.split_on_commas(self.args.consider)
            LOG.info(f'Also considering candidates from {", ".join(extra_lang_codes)} as '
                     'requested.')
            lang_codes = lang_codes + extra_lang_codes
        for langcode in lang_codes:
            cands = self.build_candidates_for_lang(langcode, langcode, entry)
            if cands:
                candidates[langcode] = cands
            elif langcode in self._main_to_fallbacks:
                for fallback_code in self._main_to_fallbacks[langcode]:
                    # Some fallback languages (such as 'tpi') have their own mapping,
                    # while others (such as 'ur') use the mapping of the main language
                    if fallback_code in self.convdicts:
                        conv_dict_name = fallback_code
                    else:
                        conv_dict_name = langcode

                    cands = self.build_candidates_for_lang(fallback_code, conv_dict_name, entry)
                    if cands:
                        LOG.info(f'Using fallback candidate(s) from {fallback_code}')
                        candidates[fallback_code] = cands
                        self._active_fallbacks.add(fallback_code)
                        break
        if constraints and constraints.added:
            self.add_specified_word(candidates, constraints.added, 'Add: constraint')
        elif self.args.word:
            self.add_specified_word(candidates, self.args.word[0].strip(), '--word argument')
        return candidates

    def do_polysemy_check(self, entry: LineDict) -> Tuple[dict[str, Set[str]], Set[str]]:
        """Check whether one might add this meaning to an existing word.

        Returns:

        1. A dictionary of words in our languages to sets of languages which use the same word
           for both meanings. Matching is case-insensitive.
        2. The set of languages for which the meaning has translations.

        The check considers all our source languages, including fallback languages and the
        default auxlangs typically used to generate candidates, if present. Other languages are
        not considered even if they are present in the dictionary (e.g. Esperanto, Italian,
        Polish).
        """
        # Mapping from our words to languages which share this meaning
        matchdict: dict[str, Set[str]] = defaultdict(set)
        entry_langs = set()

        for key, translations in entry.items():
            if key not in self._full_sourcelang_set:
                continue  # Only source languages are considered
            entry_langs.add(key)

            # Remove IPA or romanizations in brackets
            translations = util.discard_text_in_brackets(translations)

            for trans in util.split_on_semicolons(translations):
                full_trans = f'{key}:{trans}'.lower()
                our_words = self.polyseme_dict.get(full_trans, set())
                for our_word in our_words:
                    matchdict[our_word].add(key)
                    ##LOG.info(f'{key} uses the same word')

        return matchdict, entry_langs

    def check_polysemy(self, entry: LineDict, do_print: bool = True) -> tuple[str | None,
                                                                              float | None]:
        """Check whether one might add this meaning to an existing word.

        If yes, print suitable suggestions if 'do_print' is True.

        Also returns the best suggestion if at least 40% of languages use the same word
        for both concepts, together with the shared percentage, e.g. ("basa", 67.7) meanin
        that  67.7 of the shared languages use words also used for "basa". If there is no
        such suggestion, (None, None) is returned instead.
        """
        matchdict, entry_langs = self.do_polysemy_check(entry)
        requested_check = bool(self.args.polycheck)

        if requested_check and not matchdict:
            # Invoked through --polycheck option, so we need to print a status update anyway
            LOG.info(self.format_msg("Polysemy check: Don't consider merging these two concepts "
                                     '– no languages (0.0%) use the same word for both concepts.'))

        for word, langset in sorted(matchdict.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            word_langs = self.langcode_sets.get(word, set())
            shared_lang_count = len(entry_langs.intersection(word_langs))

            if not shared_lang_count:
                if requested_check:
                    LOG.info("Sorry, but the two words don't have any languages in common!")
                continue

            percentage = len(langset) / shared_lang_count * 100.0

            if not do_print:
                # Since we don't have to print anything, we can return now with the first
                # (best) candidate
                if percentage >= 40:
                    return (word, percentage)
                else:
                    return (None, None)

            if percentage < 33.3 and not requested_check:
                ## LOG.info(f'--- {word}: {percentage:.1f}% ---')
                continue
            if percentage > 66.6:
                intro = 'Urgently consider'
            elif percentage >= 50:
                intro = 'Do consider'
            elif percentage > 33.3:
                intro = 'Consider'
            else:
                # Only relevant if this is a requested check (--polycheck), where results should
                # be printed in any case
                intro = "Don't consider"

            # The recommended action depends on whether or not we are in a requested check
            if requested_check:
                action = 'merging these two concepts'
            else:
                action = f'adding this meaning to "{word}"'

            langlist = ', '.join(sorted(langset))
            LOG.info(self.format_msg(f'Polysemy check: {intro} {action} – {len(langset)} of '
                                     f'{shared_lang_count} languages ({percentage:.1f}%) use '
                                     f'the same word for both concepts ({langlist}).'))

        return (None, None)

    def store_specified_word(self, candidates: dict[str, Sequence[Candidate]],
                             entry: LineDict) -> None:
        """Add the word specified by the --word option to the dictionary.

        This word must now already exist in the candidates dictionary using an empty string as
        language code.

        If the specified word is NOT related to the words candidates yielded by any of our
        source languages, it is rejected with an error message.
        """
        cand_entry = self.combine_entry(candidates[''][0], entry)
        # Ensure that 'infl' field isn't empty
        if not cand_entry.get('infl', ''):
            sys.exit(f'error: --word argument "{cand_entry.get("word", "")}" is not related to '
                     'any word from our source languages')
        self.add_to_dict(cand_entry, -1)

    def select_candidate(self, candidates: dict[str, Sequence[Candidate]], entry: LineDict,
                         constraints: bu.Constraints | None = None) -> None:
        """Calculate penalties and let the user select one candidate.

        If the --class option is specified, the class of the entry will be set accordingly.

        If the --compound option is used, the specified compound will be added instead.
        """
        # Adjust class if the --class option is used
        if self.args.cls:
            entry.add('class', self.args.cls.strip(), allow_replace=True)

        if self.args.compound is not None:
            self.add_as_compound(entry)
            return
        if self.args.addmeaning is not None:
            self.add_as_meaning(entry)
            return

        # Check whether one might add this meaning to an existing word (unless the --copy
        # option or Merge constraint was used, which makes this pointless/unnecessary)
        if not (self.args.copy or (constraints and constraints.merge_with)):
            self.check_polysemy(entry)

        # Calculate similarity penalties
        for lang in candidates:
            self.calc_sim_penalties(lang, candidates)
        cand_list = self.store_normalized_sim_penalties(candidates)
        self.present_cands_for_selection(cand_list, candidates, entry, constraints)

    def filter_entries_by_kind(self, entries: List[LineDict]) -> List[LineDict]:
        """Remove entries of the wrong kind to ensure a balanced dictionary.

        Also remove spurious entries occasionally produced by the wiktextract parser.

        Sometimes there were entries with the same translations and sense, but different
        word classes, see https://github.com/tatuylonen/wiktextract/issues/57.
        This function removes any such spurious entries, returning only entries that
        should actually be added.

        That issue is fixed now, but this code stays around in case that some similar problem
        shows up.

        Note that the result list may be empty.
        """
        entries_by_key: dict[str, List[LineDict]] = defaultdict(list)

        for entry in entries:
            if self.get_kind(entry) not in self.kinds_to_add:
                # Entry has the wrong kind (e.g. 'noun' if we want to add an adjective)
                continue
            entry_key = self.mk_entry_key(entry)
            entries_by_key[entry_key].append(entry)

        result = []

        for entry_key, entry_list in entries_by_key.items():
            if len(entry_list) > 1:
                LOG.info(f'Note: found {len(entry_list)} entries for "{entry_key}" -- will just '
                         'keep the first noun')
                for entry in entry_list:
                    if entry.get('class') == 'noun':
                        result.append(entry)
                        break
                else:
                    LOG.warn('No noun found!')
            else:
                result += entry_list

        return result

    def select_cand_dict_to_handle_first(self,
                                         cand_dict_list: List[dict[str, Sequence[Candidate]]],
                                         entries: List[LineDict]) -> Tuple[
                                             dict[str, Sequence[Candidate]], LineDict]:
        """Select the candidate dictionary that should be added first.

        Returns the selected candidate dictionary and the corresponding entry (the element
        at the same position of the 2nd argument).

        To make the choice, all candidates are sorted alphabetically (using the form:
        "word/langcode", where word is the external form of the word and langcode is the
        language code).
        """
        # Sanity check
        if len(cand_dict_list) != len(entries):
            raise ValueError('Both arguments to select_cand_dict_to_handle_first must have the '
                             'same number of elements')

        if len(cand_dict_list) == 1:
            ## Trivial choice
            return cand_dict_list[0], entries[0]

        list_of_cand_lists = []

        # Build lists of candidates and sort them alphabetically
        for cand_dict in cand_dict_list:
            cand_list = []
            for cands in cand_dict.values():
                for cand in cands:
                    cand_list.append(f'{cand.export_word()}/{cand.lang}')
            cand_list.sort()
            list_of_cand_lists.append(cand_list)

        # Choose the list that comes alphabetically first and its position
        pos, first_list = min(enumerate(list_of_cand_lists), key=itemgetter(1))
        selected_cand_dict = cand_dict_list[pos]
        selected_entry = entries[pos]
        en_word = selected_entry['en']
        LOG.info(self.format_msg(f'Will handle "{en_word}" first because it has the '
                                 f'alphabetically first candidate: {first_list[0]}.'))
        return selected_cand_dict, selected_entry

    def find_entry(self, entry_map: dict[int, List[LineDict]], word: str, sense: str) -> LineDict:
        """Find the entry with the specified English word and word sense.

        The sense may include the word class, separated by '//' after the actual sense;
        alternatively the --type option may be used to specify the word class. In either case,
        the entry must have the specified word class. Otherwise the first matching entry will
        be returned, regardless of word class (POS).

        Dies with an error if the entry cannot be found.
        """
        cls = None
        if self.args.typ:
            cls = self.args.typ
        if '//' in sense:
            sense, cls = sense.split('//', 1)

        for entry_list in entry_map.values():
            for entry in entry_list:
                if (util.discard_text_in_brackets(entry.get('en', '')) == word
                        and entry.get('sense') == sense):
                    if cls is None or entry.get('class') == cls:
                        return entry
        sys.exit(f'error: No matching entry found for "{word}" ({sense})')

    def look_up_entry(self, entry_map: dict[int, List[LineDict]], word: str,
                      sense: str) -> LineDict:
        """Find the entry with the specified English word and word sense, handling '–' gracefully.

        If the sense is '–', this is interpreted as just an English word without a
        corresponding Wiktionary concept, so a small LineDict containing nothing but the
        English word and a dummy sense is returned.

        Otherwise the 'find_entry' result is returned.
        """
        if sense == '–':
            entry = LineDict()
            entry.add('sense', '–')
            entry.add('en', word)
            return entry
        return self.find_entry(entry_map, word, sense)

    def process_aux_line(self, english: str, senses: str, constraint_str: str,
                         aux_candidates: dict[str, str], line_no: int,
                         entry_map: dict[int, List[LineDict]]) -> None:
        """Process a line from the auxfile, choosing the best among the listed auxlang candidates.

        See the docs for the "process_auxfile" function for an explanation of the arguments.
        """
        # Clear the cache of candidates generates for earlier words
        self.candi_cache.clear()

        # Parse and print constraints, if any
        constraint_str = constraint_str.strip()
        if constraint_str:
            constraints = bu.Constraints(constraint_str)
            LOG.info(str(constraints) + '.')
        else:
            constraints = None

        # There may be an explanatory comment in parenthesis at the end of the English word
        english_words, comment = util.split_text_and_explanation(english)

        # Some affixes have only a comment-like explanation, no actual word – in that case
        # we restore the comment as word
        if comment and not english_words:
            english_words = f'({comment})'
            comment = None

        # There may be several comma-separated English words, and several pipe-separated senses
        en_list = util.split_on_commas(english_words)
        sense_list = util.split_on_pipes(senses)
        # Pop first English word and its sense
        this_word = en_list.pop(0)
        this_sense = sense_list.pop(0)

        # Look up the first concept
        entry = self.look_up_entry(entry_map, this_word, this_sense)

        # Merge additional concepts, if given
        while (en_list or sense_list):
            if sense_list:
                this_sense = sense_list.pop(0)
                # Check if the sense includes the word, separated by '::'
                if '::' in this_sense:
                    this_word, this_sense = this_sense.split('::', 1)
                elif en_list:
                    # Match with next English word
                    # (otherwise, if there is none, we re-use the last one)
                    this_word = en_list.pop(0)

                other_entry = self.look_up_entry(entry_map, this_word, this_sense)
                entry = self.do_merge_entries(entry, other_entry)
            elif en_list:
                LOG.warn(f'English words without corresponding senses in line #{line_no} will '
                         f'be ignored: {", ".join(en_list)}')
                break

        # Add the auxlang translations (semicolon-separated)
        for auxlang, candidates in aux_candidates.items():
            candidates = candidates.replace(",", ";")
            entry.add(auxlang, candidates)

        if comment:
            # Add the comment back to the English word
            entry.append_to_val('en', f' ({comment})')

        # Generate and rank candidates, choosing the most suitable one
        cand_dict = self.build_candidates(entry, constraints)
        self.select_candidate(cand_dict, entry, constraints)

    def process_auxfile(self, entry_map: dict[int, List[LineDict]]) -> None:
        """Read concepts with translations into several auxlangs from a CSV files.

        For each line in the file, the auxlang translation that's most similar to the
        words from the source languages will be chosen and added to our dictionary.

        The file must have at least five columns:

        * The English word – several words may be separated by commas, and there may be an
          explanatory comment in parenthesis at the end of the value
        * The matching sense in Wiktionary (separated by pipe characters if there are several
          concepts to add; the corresponding words may be comma-separated, otherwise they
          senses are interpreted as different senses of the same word). it's also possible to
          include the word itself, followed by '::' and the sense definition, e.g.
          "woman::adult female person" (useful e.g. when listing several senses of the same
          word, adjacent to senses for other words). If a sense is '–' (en dash), the corresponding
          word is just added as translation without looking for a Wiktionary concept.
        * A possible set of constrains (semicolon-separated) such as "Syllables:1.5" (case is
          ignored)
        * The translations into two or more auxlangs (several words may be separated by commas)

        Explanations in parentheses in the English word and the auxlang translations will be
        ignored.

        The first line must be a header line. Two first two rows are ignored, but the additional
        ones give the names of the auxlangs to consider.

        Once a line is too short or the English word is empty, processing stops.
        """
        auxfile = self.args.auxfile
        LOG.info(f'Processing entries in auxfile {auxfile}.')

        with open(auxfile, newline='', encoding='utf8') as csvfile:
            reader = csv.reader(csvfile)
            for line_no, row in enumerate(reader, start=1):
                if line_no == 1:
                    # Read names of auxlangs
                    auxlangs = row[3:]
                    # Store them as attribute and add them to our list of source languages
                    # (lower-cased)
                    self.auxlangs = [auxlang.lower() for auxlang in auxlangs]
                    self._sourcelang_list.extend(self.auxlangs)
                    continue

                # Process entry (may represent several related concepts)
                english = util.get_elem(row, 0)
                if not english:
                    LOG.info(f'No English word found in line {line_no} of {auxfile} – '
                             'stopping processing.')
                    break

                senses = util.get_elem(row, 1)
                constraint_str = util.get_elem(row, 2)
                aux_candidates = {}

                for auxpos, auxlang in enumerate(auxlangs, start=3):
                    auxcand = util.get_elem(row, auxpos)
                    if auxcand and auxcand not in ('-', '–'):
                        aux_candidates[auxlang.lower()] = auxcand

                self.process_aux_line(english, senses, constraint_str, aux_candidates, line_no,
                                      entry_map)

        # If any words were chosen and committed, export them and update the selection log
        if self.chosen_candidates:
            self.add_entries_to_dict(self.chosen_candidates)
            # Append all print and warn messages to the selection log
            LOG.info(f'Choices committed on {util.current_datetime()}.')
            LOG.append_all_messages(SELECTION_LOG)

    def find_concepts(self, concept_count: int) -> None:
        """Find 'concept_count' new concepts to add and print them to stdout.

        Output is pipe-separated: english|sense|class.

        The concepts with the highest number of translations not yet represented in our
        dictionary are chosen first, but distinguished by the kind of word, a rough word class
        grouping. The target distribution is:

          * 40% nouns (including proper nouns)
          * 20% adjectives and adverbs
          * 40% verbs and other words

        Based on the kind distribution in our existing dictionary, the new concepts will be
        chosen to as to get near to this target distribution (e.g. if nouns are underrepresented,
        more nouns will be chosen).
        """
        LOG.info(f'Finding {concept_count} new concepts to add.')
        counter = self.count_existing_kinds(self.existing_entries)

        # Calculate ideal target distribution
        target_sum = sum(counter.values()) + concept_count
        target_counts: dict[Kind, int] = {}
        target_counts[Kind.NOUN] = round(target_sum * .4)
        target_counts[Kind.ADJ] = round(target_sum * .2)
        target_counts[Kind.VERB] = target_sum - target_counts[Kind.NOUN] - target_counts[Kind.ADJ]

        # Calculate overshoot, since some kinds may already be overrepresented
        overshoot = 0
        overshot_kinds = set()

        for kind in counter.keys():
            diff = counter[kind] - target_counts[kind]
            if diff > 0:
                overshoot += diff
                overshot_kinds.add(kind)
                target_counts[kind] = counter[kind]

        if overshoot:
            # Subtract one of each from NOUN, ADJ, again NOUN, VERB that's not overrepresented
            # unless the correct target count has been reached
            kind_list = [Kind.NOUN, Kind.ADJ, Kind.NOUN, Kind.VERB]
            kind_list = [kind for kind in kind_list if kind not in overshot_kinds]
            remaining_overshoot = overshoot

            for kind in cycle(kind_list):
                if target_counts[kind] > counter[kind]:
                    # We still can safely reduce the target count of this kind
                    target_counts[kind] -= 1
                    remaining_overshoot -= 1
                    if not remaining_overshoot:
                        break

        # Calculate how many concepts of each kind we need to find, dropping those we don't need
        needed_kinds = {kind: target_counts[kind] - counter.get(kind, 0) for kind in target_counts}
        needed_kinds = {kind: count for kind, count in needed_kinds.items() if count != 0}

        # Sanity checks
        needed_sum = sum(needed_kinds.values())
        if needed_sum != concept_count:
            raise RuntimeError(f'Internal error: total needed concept count is {needed_sum} '
                               f'instead of {concept_count}')
        min_needed = min(needed_kinds.values())
        if min_needed <= 0:
            raise RuntimeError(f'Internal error: non-positive value {min_needed} in '
                               'needed_kinds dict')

        counter_formatted = ', '.join(f'{count}x {key.name.lower()}' for key, count in sorted(
            counter.items(), key=lambda pair: (-pair[1], pair[0])))
        needed_kinds_formatted = ', '.join(f'{count}x {key.name.lower()}' for key, count in sorted(
            needed_kinds.items(), key=lambda pair: (-pair[1], pair[0])))
        msg = f'Existing distribution: {counter_formatted}; needed counts: {needed_kinds_formatted}'
        if overshoot:
            msg += f'; original overshoot: {overshoot}'
        LOG.info(msg + '.')

        # Open output CSV file for writing and add header, after moving an existing version away
        outfilenname = f'top{concept_count}.csv'
        util.rename_to_backup(outfilenname)
        with open(outfilenname, 'w', newline='', encoding='utf8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow('English Sense Constraints'.split()
                            + [lang.capitalize() for lang in sorted(util.DEFAULT_AUXLANGS)])

            # Find concepts to add
            for _count, entries_by_count in sorted(self.sort_entries_by_transcount().items(),
                                                   reverse=True):
                for entry in entries_by_count:
                    # Check if we need an entry of this kind
                    kind = self.get_kind(entry)
                    if kind in needed_kinds:
                        english = util.discard_text_in_brackets(entry.get('en', ''))
                        sense = entry.get('sense', '')

                        # Check if the polysemy check suggestions a merger
                        merge_word, merge_percentage = self.check_polysemy(entry, False)
                        if merge_word:
                            merge_constraint = f'Merge:{merge_word} ({merge_percentage:.1f}%)'
                        else:
                            merge_constraint = ''

                        # Write row
                        row = [english, sense]
                        if merge_constraint:
                            row.append(merge_constraint)
                        writer.writerow(row)

                        # Update remaining needed counts, removing kinds that are no longer needed
                        # -- entries that shall be merged don't count
                        if not merge_constraint:
                            needed_kinds[kind] -= 1
                            if needed_kinds[kind] <= 0:
                                del needed_kinds[kind]

                    if not needed_kinds:
                        break
                if not needed_kinds:
                    break

    def process_requested_entry(self, entry_map: dict[int, List[LineDict]]) -> None:
        """Process the entry requested by the --add command."""
        word, sense, rationale = self.args.add
        # This allows adding a space in front of the word starts with a hyphen (otherwise
        # argparse won't accept it)
        word = word.strip()

        with_opt = ' with copy option turned on' if self.args.copy else ''
        if self.args.schwastrip:
            if with_opt:
                with_opt = with_opt.replace('option', 'and schwastrip options')
            else:
                with_opt = ' with schwastrip option turned on'

        LOG.info(self.format_msg(
            f'Processing entry "{word}" ({sense}) as requested{with_opt}, rationale: {rationale}'))
        entry = self.find_entry(entry_map, word, sense)

        if self.args.merge:
            cand_dict, entry = self.merge_entries(entry, entry_map)
        else:
            cand_dict = self.build_candidates(entry)
        self.select_candidate(cand_dict, entry)

    def mk_requested_polycheck(self, entry_map: dict[int, List[LineDict]]) -> None:
        """Check whether it is reasonable to consider two concepts as meanings of the same word.

        This is triggered by the --polycheck option and will print a suitable recommendation
        based on how many of the widely spoken languages use the same word for them.
        """
        word_1, sense_1, word_2, sense_2 = self.args.polycheck
        word_1 = word_1.strip()
        word_2 = word_2.strip()

        if word_1 == word_2 and sense_1 == sense_2:
            sys.exit('error: There is no point in comparing an entry with itself!')

        entry_1 = self.find_entry(entry_map, word_1, sense_1)
        entry_2 = self.find_entry(entry_map, word_2, sense_2)

        # Rebuild polyseme_dict and langcode_sets, using just the first entry as contents
        entry_1.add('word', POLY_DUMMY)
        self.polyseme_dict = self.fill_polyseme_dict([entry_1])
        self.langcode_sets = self.fill_langcode_sets([entry_1])

        # Do check on second entry
        self.check_polysemy(entry_2)

    @staticmethod
    def do_merge_entries(entry: LineDict, other_entry: LineDict) -> LineDict:
        """Merge the given entry (entry) with the one specified by the --merge option (other_entry).

        Translations will be merged by adding all those of the second entry except for
        those already listed in the first entry; they are considered to be semicolon-separated.

        Senses are merged by separated them with a ' | ' (a pipe character surrounded by
        spaces).

        In general, only the first class it kept if they are different. However, if one of
        the classes is an affix, both are kept (comma-separated).

        If both entries have the same word class, it won't be duplicated; otherwise, the two
        classes will be separated by a comma.

        The "transcount" field is merged by keeping the larger value.
        """
        # pylint: disable=too-many-branches, too-many-locals, too-many-statements
        combined_entry = LineDict()

        for key, value in entry.items():
            value_2 = other_entry.get(key, '')
            if not value_2:
                joint_value = value
            elif key == 'class':
                if value == value_2:
                    joint_value = value
                elif value.endswith('fix') or value_2.endswith('fix'):
                    if ',' in value:
                        # The first is a list such as 'noun, suffix': only append the second
                        # value if it's not yet part of the list
                        existing_classes = set(util.split_on_commas(value))
                        if value_2 in existing_classes:
                            joint_value = value
                        else:
                            joint_value = value + ', ' + value_2
                    else:
                        joint_value = value + ', ' + value_2
                else:
                    # Just keep the first value
                    joint_value = value
            elif key == 'sense':
                joint_value = value + ' | ' + value_2
            elif key == 'transcount':
                joint_value = str(max(int(value), int(value_2)))
            else:
                words = set(util.split_on_semicolons(value))
                joint_word_list = [value]
                for word in util.split_on_semicolons(value_2):
                    if word not in words:
                        joint_word_list.append(word)
                joint_value = '; '.join(joint_word_list)
            combined_entry.add(key, joint_value)

        # Check if sense disambiguation is possible without problems
        sense = combined_entry['sense']
        senses = sense.split(' | ')
        en_word = combined_entry['en']
        en_words = util.split_on_semicolons(en_word)

        if len(en_words) > 1 and len(senses) != len(en_words):
            # This can happen if the new word is already part of the existing translations, e.g.
            # en: declare, declaration  (new "declaration" ignored since it's already listed)
            # sense: to make a declaration | act or process of declaring | written or oral
            #        indication of a fact, opinion, or belief
            # We disambiguate this by adding the word in parentheses at the end of each sense,
            # resulting in
            # sense: to make a declaration (declare) | act or process of declaring (declaration)
            #        | written or oral indication of a fact, opinion, or belief (declaration)
            orig_en_word = util.discard_text_in_brackets(entry.get('en', ''))
            orig_en_words = util.split_on_semicolons(orig_en_word)

            new_subsenses = []
            for idx, subsense in enumerate(senses[:-1]):
                # Old senses: add word in parentheses if it doesn't yet seem to be there
                if util.split_text_and_explanation(subsense)[1] is None:
                    if idx >= len(orig_en_words):
                        # Re-use the last original word for this sense
                        matching_en_word = orig_en_words[-1]
                    else:
                        matching_en_word = orig_en_words[idx]
                    # If the matching word contains an explanation in parentheses, we strip it
                    matching_en_word = util.eliminate_parens(matching_en_word)
                    new_subsense = f'{subsense} ({matching_en_word})'
                else:
                    new_subsense = subsense
                new_subsenses.append(new_subsense)

            # Finally add the sense from the new entry
            new_en_word = util.discard_text_in_brackets(other_entry.get('en', ''))
            new_subsense = f"{other_entry.get('sense', '')} ({new_en_word})"
            new_subsenses.append(new_subsense)
            joint_sense = ' | '.join(new_subsenses)
            combined_entry.add('sense', joint_sense, allow_replace=True)

        # Copy any additional entries from the second entry
        for key, value in other_entry.items():
            if key not in combined_entry:
                combined_entry.add(key, value)
        return combined_entry

    def merge_entries(self, entry: LineDict, all_entries: dict[int, List[LineDict]]) -> Tuple[
            dict[str, Sequence[Candidate]], LineDict]:
        """Merge the given entry with the one specified by the --merge option.

        See 'do_merge_entries' for details.
        """
        word, sense, rationale = self.args.merge
        if self.args.addmeaning:
            sys.exit("error: please don't use --merge and --addmeaning together; instead add "
                     'each meaning separately')
        # This allows adding a space in front of the word starts with a hyphen (otherwise
        # argparse won't accept it)
        word = word.strip()
        first_word = entry.get('en', '')
        first_sense = entry.get('sense', '')
        LOG.info(self.format_msg(f'Merging "{first_word}" ({first_sense}) and "{word}" ({sense}) '
                                 f'as requested, rationale: {rationale}'))
        other_entry = self.find_entry(all_entries, word, sense)
        combined_entry = self.do_merge_entries(entry, other_entry)
        cand_dict = self.build_candidates(combined_entry)
        return cand_dict, combined_entry

    def process_entry_batch(self, entries: List[LineDict], transcount: int,
                            all_entries: dict[int, List[LineDict]]) -> int:
        """Process a batch of entries, building candidate words and selecting the best ones.

        Returns the number of entries handled.
        """
        entries = self.filter_entries_by_kind(entries)
        if not entries:
            return 0

        entry_str = 'entry' if len(entries) == 1 else 'entries'
        LOG.info(f'Processing {len(entries)} {entry_str} with {transcount} translations.')
        cand_dict_list: List[dict[str, Sequence[Candidate]]] = []
        for entry in entries:
            cand_dict_list.append(self.build_candidates(entry))
        cand_dict, entry = self.select_cand_dict_to_handle_first(cand_dict_list, entries)
        if self.args.merge:
            cand_dict, entry = self.merge_entries(entry, all_entries)
        self.select_candidate(cand_dict, entry)
        return len(entries)

    def delete_word(self) -> None:
        """Delete the specified word and optionally its derivatives from the dictionary.

        Derivatives are deleted if the --delete is used, but not if the --delete-only option
        is used.

        The word to delete and the reason for the deletion are read from the specified option.
        """
        # pylint: disable=too-many-branches, too-many-locals
        if self.args.delete_only:
            word, reason = self.args.delete_only
            delete_derivatives = False
            runs_required = 1
        else:
            word, reason = self.args.delete
            delete_derivatives = True
            runs_required = 2

        word = word.strip()
        word_lower = word.lower()
        words_to_delete: Set[str] = set([word])
        words_to_delete_lower: Set[str] = set([word_lower])

        if delete_derivatives:
            if ',' in word_lower:
                # We also add each of the comma-separated synonyms separately, to find their
                # derivatives
                for synonym in util.split_on_commas(word_lower):
                    words_to_delete_lower.add(synonym)

            # Also add a hyphen-stripped versions in case of affixes, to find uses in glosses
            for this_word in words_to_delete_lower.copy():
                if this_word.startswith('-') or this_word.endswith('-'):
                    words_to_delete_lower.add(this_word.strip('-'))

        word_seen = False

        # If --delete is used, we run this code twice to also catch indirect derivatives,
        # e.g. "jen brasili" is derived from "brasili" which is derived from "Brasil"
        for _idx in range(runs_required):
            entries_to_keep: List[LineDict] = []
            for entry in self.existing_entries:
                this_word = entry.get('word', '')
                this_word_lower = this_word.lower()
                this_gloss = entry.get('gloss', '')
                if ' ' in this_word_lower or '-' in this_word_lower:
                    parts = re.split(r'[ -]+', this_word_lower)
                else:
                    parts = []

                if word_lower == this_word_lower or (delete_derivatives and any(
                        deletable_word in parts for deletable_word in words_to_delete_lower)):
                    if word == this_word:
                        word_seen = True
                    else:
                        words_to_delete.add(this_word)
                        words_to_delete_lower.add(this_word.lower())
                elif delete_derivatives and this_gloss:
                    parts = this_gloss.lower().split('+')
                    if any(deletable_word in parts for deletable_word in words_to_delete_lower):
                        words_to_delete.add(this_word)
                        words_to_delete_lower.add(this_word.lower())

                if this_word not in words_to_delete:
                    entries_to_keep.append(entry)
            self.existing_entries = entries_to_keep

        if not word_seen:
            sys.exit(f'error: "{word}" not found in the dictionary')

        self.add_entry_to_dict(None)
        words_joined = '"' + '", "'.join(words_to_delete) + '"'
        LOG.info(self.format_msg(
            f'Deleted {words_joined} on {util.current_datetime()}; reason: {reason}'))
        # Append all print and warn messages to the selection log
        LOG.append_all_messages(SELECTION_LOG)

    def run(self) -> None:
        """Main function: build our vocabulary or perform the specified command(s)."""
        if self.args.delete or self.args.delete_only:
            self.delete_word()
            return
        if self.args.addenglish:
            self.add_english()
            return
        if self.args.addkomusan:
            self.add_komusan()
            return

        count_map = self.sort_entries_by_transcount()
        addcount = 0

        if self.args.add:
            self.process_requested_entry(count_map)
        elif self.args.auxfile:
            self.process_auxfile(count_map)
        elif self.args.findconcepts:
            self.find_concepts(self.args.findconcepts)
        elif self.args.polycheck:
            self.mk_requested_polycheck(count_map)
        else:
            # If no specific entry was requested, we add one of the entries with the highest
            # number of translations
            for count, entries in sorted(count_map.items(), key=lambda pair: -pair[0]):
                addcount += self.process_entry_batch(entries, count, count_map)
                if addcount >= 1:
                    break  # We process just one word at a time


##### ArgumentParser and main entry point #####

def build_arg_parser() -> argparse.ArgumentParser:
    """Build a parser for the arguments this script can handle."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--select', type=int, help='the candidate to add to the dictionary')
    parser.add_argument('-sr', help='the selection rationale -- must be specified whenever any '
                                    'other but the 1st candidate is selected')
    parser.add_argument('-a', '--add', metavar=('WORD', 'SENSE', 'RATIONALE'), nargs=3,
                        help='the entry to add next, requires three arguments: the English word, '
                             'the specific word sense, and a rationale for why this word is to '
                             'be added now')
    parser.add_argument('-ae', '--addenglish', metavar=('KOMUSAN', 'ENGLISH'), nargs=2,
                        help='Add one or more words to the English translation of an existing '
                             "entry; multiple translations are comma-separated; it's also "
                             'possible to include an explanation in parentheses or to put the '
                             'whole addition in parentheses; sample including both: "nobody, '
                             'anyone, anybody (in negated sentences)"')
    parser.add_argument('-ak', '--addkomusan', metavar=('OLD_WORD', 'NEW_WORD', 'RATIONALE'),
                        nargs=3,
                        help='Add NEW_WORD as a synonym to the Komusan entry OLD_WORD; requires '
                             "a rationale as third argument (can be set to '-' if a gloss is "
                             'specified instead)')
    parser.add_argument('-am', '--addmeaning', metavar='WORD',
                        help='Add this entry to the meaning of another word which already exists '
                             "in the dictionary -- unless the merger was suggested by the "
                             'polysemy check, the reason should be added using the -amr argument')
    parser.add_argument('-amr', help='the rationale for calling --addmeaning -- if not specified, '
                                     'it will be assumed that the decision was due to the '
                                     'polysemy check')
    parser.add_argument('-ad', '--allowduplicates', action='store_true',
                        help='Allow candidates that are duplicates of already existing words')
    parser.add_argument('-as', '--allowshort', metavar='RATIONALE',
                        help='Allow candidates that would otherwise be considered too short; '
                             'requires a rationale for this decision')
    parser.add_argument('-cl', '--class', dest='cls', metavar='CLASS',
                        help='Set the class of the new entry to the specified value.')
    parser.add_argument('-c', '--compound', metavar=('WORD', 'RATIONALE'), nargs=2,
                        help='Add the next entry as compound instead of choosing one of the '
                             'candidates; requires two arguments: the compound to add and a '
                             'rationale for why this word is to be formed in this manner (if the '
                             "latter is obvious, e.g. in the case of phrases, it can be set to '-')"
                        )
    parser.add_argument('--consider', metavar='LANGS',
                        help='Also consider candidates from the listed language code or '
                             '(comma-separated) codes when building candidates. This allows '
                             'adding words from languages that are not usually sources. Note: '
                             'If you add an xx-ipa field to extradict.txt, it will be used to '
                             'make the candidate, otherwise the raw (original) form of the '
                             'word will be used.')
    parser.add_argument('--copy', action='store_true',
                        help='This can be used together with --add to reprocess an entry that '
                             'already exists in the dictionary (e.g. it was used to add the '
                             'entry "human being" (person) to "san" in addition to "insan")')
    parser.add_argument('--core', action='store_true',
                        help='Tag this entry as "core" vocabulary (this is a shortcut for '
                             '--field tags core)')
    parser.add_argument('--delete', metavar=('WORD', 'REASON'), nargs=2,
                        help='Delete the specified word (as well as all of its derivatives!) '
                             'from the dictionary; requires a reason for the deletion.')
    parser.add_argument('--delete-only', metavar=('WORD', 'REASON'), nargs=2,
                        help='Delete the specified word, but not its derivatives; requires a '
                             'reason for the deletion. When using this, you have to ensure that '
                             'you won\'t leave the dictionary in an inconsistent state!')
    parser.add_argument('-f', '--field', metavar=('NAME', 'VALUE'), action='append', nargs=2,
                        help='Add an additional field to the entry; this option can be called '
                             'several times (with different names); it is useful to add fields '
                             'that are present in some, but not all entries (e.g. "sample", '
                             '"tags", "value")')
    parser.add_argument('-1', '--first', action='store_true',
                        help='If this is combined with the --addkomusan option, NEW_WORD will be '
                             'added in front of OLD_WORD rather than after it')
    parser.add_argument('-g', '--gloss',
                        help="shows the parts ('+' separated) of a compound that's a single word, "
                             'e.g. "dudes" should be glossed as "du+des"; alternatively, an '
                             'informal explanation such as "contraction of ..." can be given '
                             '(containing spaces)')
    parser.add_argument('-m', '--merge', metavar=('WORD', 'SENSE', 'RATIONALE'), nargs=3,
                        help="the entry that'll be added next will be merged with the specified "
                             'entry, that is, all translations will be combined (eliminating '
                             'duplicates) and candidates will be derived from all of them; '
                             'arguments as for --add')
    parser.add_argument('-pc', '--polycheck', metavar=('WORD_1', 'SENSE_1', 'WORD_2', 'SENSE_2'),
                        nargs=4,
                        help='check whether it is reasonable to consider two concepts as meanings '
                             'of the same word (explicit polysemy check)')
    parser.add_argument('--schwastrip', action='store_true',
                        help='Normally this option should not be used, but it can be used in '
                             'exceptional cases to strip the filler vowels (schwas) inserted '
                             'for phonetic reasons at the start or end of some candidates. '
                             'NOTE that this can lead to invalid candidates!!')
    parser.add_argument('-t', '--type', dest='typ', metavar='CLASS',
                        help='Sometimes the same word/sense combination occurs twice in the '
                             'dictionary, with different word classes. In this case, this '
                             'argument can be used to specify which of them should be selected. '
                             'Don\'t confuse this with the --class option, which can be used to '
                             'change the class of a word added to the dictionary!')
    parser.add_argument('--tags',
                        help='Add the specified tag or tags (multiple tags should be '
                             'comma-separated and listed in alphabetic order)')
    parser.add_argument('-w', '--word', metavar=('WORD', 'RATIONALE'), nargs=2,
                        help="Use the specified word to represent the entry that'll be added "
                             "next – this allows adding a form that doesn't exactly correspond "
                             'to any of the proposed candidates. Note that the internally used '
                             'form must be used, e.g. "C" instead of "ch". A rationale '
                             'explaining the chosen form must be supplied as 2nd argument.')
    parser.add_argument('-x', '--auxfile',
                        help='Read concepts with translations into several auxlangs from a CSV '
                             "file. For each line in the file, the auxlang translation that's "
                             'most similar to the words from the source languages will be chosen '
                             'and added to our dictionary.')
    parser.add_argument('-co', '--commit', action='store_true',
                        help='Commit the changes made via --auxfile option to the dictionary '
                             'instead of merely printing them for inspection.')
    parser.add_argument('-fc', '--findconcepts', metavar='NUMBER', type=int, nargs='?', const=15,
                        help='Find new concepts to add (15 by default), starting with the most '
                             'frequently translated ones and ensuring an adequate coverage of '
                             'different kinds of words. Output is pipe-separated: word|sense|class.'
                        )
    return parser


if __name__ == '__main__':
    # pylint: disable=invalid-name
    builder = VocBuilder(build_arg_parser().parse_args())
    builder.run()

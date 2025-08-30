#!/usr/bin/env python3
"""Export the dictionary into a grouped wordlist.

Can be invoked with one argument that specified the language code of the translations to add,
e.g. 'en' for 'English' or 'ja' for 'Japanese'. English is the default.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, TextIO
from warnings import warn

from linedict import LineDict
import linedict
import metadata
import util


##### Constants #####

# Names of input and output files
DICT_FILE = util.DICT_FILE
# '(LANG)' in this filename will be expanded by the language code
OUTFILE = 'wordlist-(LANG).txt'

# The order in which classes should be printed
GROUP_ORDER = tuple('noun name adj adv verb aux particle quant sel pron num prep prep_phrase '
                    'conj prefix suffix intj phrase'.split())

# A few word classes are replaced with shorter names when showing them in word class lists
SHORT_CLASS_NAMES = {
    'name': 'PN',
    'noun': 'n',
    'particle': 'par',
    'prefix': 'Pref',
    'suffix': 'Suf',
    'verb': 'v'
}


##### Types #####

@dataclass
class WordInfo():
    """The information about a word we want to serialize."""
    classes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    word: str = ''
    trans: str = ''
    gloss: Optional[str] = None
    infl: List[str] = field(default_factory=list)
    numvalue: Optional[float] = None

    @staticmethod
    def parse_value(entry: LineDict) -> Optional[float]:
        """
        Parse the 'value' field of an entry into a float.

        Returns None if there is no such field or if it is empty.

        Fractions of the form "x/y" are instead parsed as "y.x" (e.g. 1/3 becomes 3.1), so that
        they will be sorted behind the corresponding base number.

        Prints a warning and returns None if the content of the 'value' field is not a number.
        """
        raw = entry.get('value')
        if not raw:
            return None
        if '/' in raw:
            # It's a fraction
            num, denom = raw.split('/', 1)
            raw = f'{denom.strip()}.{num.strip()}'
        try:
            if '^' in raw:
                # Power notation
                base, exp = raw.split('^', 1)
                return pow(int(base), int(exp))
            return float(raw)
        except ValueError:
            warn(f'Value "{raw}" is not a number')
            return None

    @staticmethod
    def from_entry(entry: LineDict, lang: str = 'en') -> WordInfo:
        """Create a WordInfo object from a LineDict.

        The created entry includes translations in the specific languages.
        """
        return WordInfo(
            util.split_on_commas(entry.get('class', '')),
            util.split_on_commas(entry.get('tags', '')),
            entry.get('word', ''),
            entry.get(lang, '???'),
            entry.get('gloss'),
            util.split_on_commas(entry.get('infl', '')),
            WordInfo.parse_value(entry)
        )

    def __str__(self) -> str:
        """Return a string representation in DokuWiki format."""
        class_info = ''
        if len(self.classes) > 1:
            # The list of classes is only printed if the word belongs to several ones
            formatted_classes = [util.capitalize(SHORT_CLASS_NAMES.get(cls, cls))
                                 for cls in self.classes]
            class_info = ' <sup>' + '/'.join(formatted_classes) + '</sup>'

        if self.gloss and ('+' in self.gloss or self.gloss.startswith('=')):
            gloss_info = f' ({self.gloss})'
        else:
            gloss_info = ''

        infl_info = f' <sup>(sources: {", ".join(self.infl)})</sup>' if self.infl else ''
        return f'  * **{self.word}** – {self.trans}{class_info}{gloss_info}{infl_info}'


##### Main class #####

class WordlistMaker:
    """Create and print the wordlist."""

    def __init__(self, lang: str) -> None:
        """Create a new instance, using 'lang' as language."""
        self.lang = lang
        self.value_provider = metadata.ValueProvider(lang)

    def print_header(self, outfile: TextIO) -> None:
        """Print the output file header."""
        title = self.value_provider.lookup('Title')
        intro = self.value_provider.lookup('Intro')
        wordlist_intro = self.value_provider.lookup('WordlistIntro')
        # Replace '(TODAY)' placeholder in the intro text by the current date in ISO format
        intro = intro.replace('(TODAY)', str(date.today()))
        print(f'====== {title} ======\n\n**{intro}**\n\n{wordlist_intro}', file=outfile)

    def print_group(self, outfile: TextIO, cls: str, word_infos: List[WordInfo]) -> None:
        """
        Print a group of words that all the belong to the same word class.

        The word class header is printed first, followed by one line for each word.

        In general, words aren't sorted – they should have been added in dictionary order,
        which means they are already sorted. However, if the first of them has a 'numvalue'
        field, they are sorted in the order of this field, in so far as it is present.
        Any words that lack it are added after them. In these cases, and in case of numerical
        ties, the original order is preserved.
        """
        # Sort using numvalue field, if present
        if word_infos[0].numvalue is not None:
            infinity = float('inf')
            word_infos = sorted(word_infos, key=lambda word:
                                word.numvalue if word.numvalue is not None else infinity)

        trans_cls = self.value_provider.lookup(cls)
        print(f'\n===== {trans_cls} =====\n', file=outfile)

        for word_info in word_infos:
            print(word_info, file=outfile)

    def create_wordlist(self) -> None:
        """Create and print the wordlist, adding translations in the requested language.
        """
        # Replace '(LANG)' in generic OUTFILE by language code
        outfilename = OUTFILE.replace('(LANG)', self.lang)
        with open(outfilename, 'w', encoding='utf8') as outfile:
            self.print_header(outfile)
            # Read and convert entries
            entries = linedict.read_dicts_from_file(DICT_FILE)
            class_dict: Dict[str, List[WordInfo]] = defaultdict(list)

            for entry in entries:
                word_info = WordInfo.from_entry(entry, self.lang)
                # Group words by first class
                class_dict[word_info.classes[0]].append(word_info)

            # Iterate word classes in group order
            for cls in GROUP_ORDER:
                word_infos = class_dict.pop(cls, [])
                if word_infos:
                    self.print_group(outfile, cls, word_infos)

            # Print any left-over classes, with a warning
            for cls, word_infos in sorted(class_dict.items()):
                warn(f'Unexpected word class: {cls}')
                self.print_group(outfile, cls, word_infos)
            # XXX Tags are ignored for now, but should be added and used in a later version


##### Main entry point #####

if __name__ == '__main__':
    LANG = util.retrieve_single_arg(default='en')
    MAKER = WordlistMaker(LANG)
    MAKER.create_wordlist()

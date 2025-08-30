#!/usr/bin/env python3
"""Parse Wikipedia's "List of languages by total number of speakers" to determine source languages.

The latest data from Ethnologue is used. Only the biggest languages are listed on that page.

Up to two languages per language branch (subfamily) are added to the file "sourcelangs.csv", while
any further languages from the same branch are added to the file "morelangs.csv" instead. For
language families not already represented among the top-10 languages, only two languages from the
whole family (instead of individual branches) are considered, with the rest added to "morelangs.csv"
instead.

Languages that are closely related (such as Hindi and Urdu or the various variants of Arabic) are
written in a single line (slash-separated) and their speaker counts added together.

The output files contain a header line, and they are both sorted by the total number of speakers.

If possible, the ISO 639-1 code (2 letters) is used to identify a languages, otherwise its 639-3
code (3 letters). The script in which a languages is written (e.g. Latin or Cyrillic) is also
added to the output files. The information is read from the "codescripts.csv" file, which must
exist locally.

Speaker counts refer to the (estimated) total number of speakers, in million.

Required libraries: requests.
"""

from collections import Counter
import csv
from dataclasses import dataclass
import dataclasses
import logging
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import urllib.parse

import requests

import util


##### Dataclass #####

@dataclass(frozen=True, eq=True)
class LanguageInfo:
    """Wraps information on a language."""

    name: str
    """Its English name."""

    iso_code: str
    """Its ISO 639-1 or 639-3 code (2 or 3 letters)."""

    family: str
    """Its language family, e.g. "Indo-European"."""

    branch: Optional[str]
    """Its language branch or subfamily, e.g. "Germanic"."""

    speakers: float
    """Its estimated total number of speakers, in million."""

    script: Optional[str] = None
    """The script used for writing, e.g. "Latin" or "Cyrillic"."""

    related: Optional[str] = None
    """"Optional name of related languages, given by Wikedia after "excl."

    For example, "Urdu" for Hindi, "creole languages" for Spanish.
    """

    @staticmethod
    def header_row() -> List[str]:
        """Create a row suitable for use as a header line in a CSV file."""
        return ['Language', 'ISO 639', 'Family', 'Branch', 'Speakers (million)', 'Script']

    def to_row(self) -> List[str]:
        """Export the fields of this instance as a list suitable for writing into a CSV file.

        The "related" entry (if any) is not exported.
        """
        return [self.name, self.iso_code, self.family, self.branch or '–',
                # Use 'g' format to avoid a trailing '.0' for round numbers
                '{:g}'.format(self.speakers), util.or_empty(self.script)]


##### Helper functions #####

def extract_fragment(text: str, start: str, end: str,
                     return_rest: bool = False) -> Union[Optional[str],
                                                         Tuple[Optional[str], Optional[str]]]:
    """From 'text', return the fragment between 'start' and 'end'.

    If 'start' or 'end' occur multiple times, the first occurrence of 'start' and the first
    occurrence of 'end' after it will be used.

    If 'return_rest', the rest of text (after 'end') will be return as second element of a
    two-tuple.

    Any whitespace around the return value(s) is stripped.

    If 'start' or 'end' is not found in 'text' or if they don't occur in the expected order,
    None is returned if 'return_rest' is False, (None, None) otherwise.

    Examples:
        >>> extract_fragment('Hello (do I really mean it ? ) world! ', '(', ')')
        'do I really mean it ?'
        >>> extract_fragment('Hello (do I really mean it ? ) world! ', 'do', '?', True)
        ('I really mean it', ') world!')
        >>> extract_fragment('Hello (do I really mean it ? ) world! ', 'you', '?')
        None
    """
    start_index = text.find(start)
    end_index = text.find(end, start_index)
    if start_index == -1 or end_index == -1:
        return (None, None) if return_rest else None
    fragment = text[start_index + len(start):end_index].strip()
    if return_rest:
        rest = text[end_index + len(end):].strip()
        return (fragment, rest)
    return fragment


def extract_first_wikilink(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract the first wikilink from a MediaWiki text fragment and return it.

    Return a 3-tuple:

    1. The linked text
    2. The link itself, if different from the linked text, otherwise None
    3. The rest of the text fragment, after the link.

    Any text before the first link is silently discarded.

    Example:
        >>> extract_first_wikilink('the [[ISO 639:eng|English]] language')
        ('English', 'ISO 639:eng', ' language')
    """
    full_link, rest = extract_fragment(text, '[[', ']]', True)  # type: ignore
    if full_link is not None and '|' in full_link:
        link, linktext = full_link.split('|')
    else:
        link = None
        linktext = full_link
    return linktext, link, rest


def split_first_line_from_rest(text: str) -> List[str]:
    """Split the first line of a multiline text from the rest, returning both as a two-element list.

    Any other whitespace is first stripped.
    """
    return text.strip().split('\n', maxsplit=1)


def title_to_filename(title: str) -> str:
    """Change a name to the form under which its Wikipedia article can be found.

    Each whitespace sequence is replaced with an underscore (_), and special characters
    (diacritics etc.) are URL-encoded.
    """
    # Parenthesis as in "Georgia (country)" should NOT be quoted, otherwise the country
    # won't be resolved correctly
    return urllib.parse.quote(re.sub(r'\s+', '_', title), safe='/()')


def dl_wikipedia_page(title: str) -> str:
    """Down a page from the English Wikipedia and return its contents (in wikitext format)."""
    response = requests.get('https://en.wikipedia.org/w/api.php?action=query',
                            params={'prop': 'revisions',
                                    'rvprop': 'content',
                                    'format': 'json',
                                    'titles': title,
                                    'rvslots': 'main'})
    response.raise_for_status()
    data = response.json()

    # Extract the needed JSON element
    pages = data.get('query').get('pages')
    first_page = next(iter(pages.values()))
    wikitext = first_page.get('revisions')[0].get('slots').get('main').get('*')
    return wikitext


##### Main class and entry point #####

class LangListParser:
    """Parses Wikipedia's "List of languages by total number of speakers"."""

    def __init__(self) -> None:
        """Create a new instance."""
        # Mapping from language code to information about that language
        self.languages: Dict[str, LanguageInfo] = {}
        # Families with a language in the top 10
        self.top_families: Set[str] = set()
        self.iso3_to_iso1_map, self.script_map = self._read_codescripts_file()

    @staticmethod
    def _read_codescripts_file() -> Tuple[Dict[str, str], Dict[str, str]]:
        """Read the "codescripts.csv" file and return info based on it.

        Specifically, two dictionaries are returned:

        1. A mapping from ISO 639-3 code (3 letters) to ISO 639-1 (2 letters) codes, for
           languages that have both.
        2. A mapping fromm ISO 639 codes to the script used by the language; if both codes are
           defined for a language, its script can be found under both
        """
        iso3_to_iso1_map = {}
        script_map = {}

        with open(util.CODESCRIPTS_FILE, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip header row
            for row in reader:
                # Map rows to local variables
                iso1, iso3, script = row
                # Add code mapping, if both codes exist
                if iso1 != '–' and iso3 != '–':
                    iso3_to_iso1_map[iso3] = iso1

                # Add entries for script
                script_map[iso1] = script
                script_map[iso3] = script

        # Delete spurious entry for '–' (not a real code) and return both dicts
        del script_map['–']
        return iso3_to_iso1_map, script_map

    @staticmethod
    def _normalize_to_millions(raw_num: str) -> float:
        """Normalize a number in millions or billions to just the raw count in millions.

        If the number doesn't correspond to the expected format, a warning is logged and 0 is
        returned.

        Examples:
            >>> self._normalize_to_millions('1.456 billion')
            1456
            >>> self._normalize_to_millions('310 million')
            310
        """
        parts = raw_num.rsplit(' ', 1)
        if len(parts) == 2:
            if parts[1] == 'billion':
                return float(parts[0]) * 1000
            if parts[1] == 'million':
                return float(parts[0])
        logging.warning(f'"{raw_num}" cannot be parsed as a number in million or billion')
        return 0

    def _parse_wikipedia_list(self) -> Dict[str, LanguageInfo]:
        """Parses Wikipedia's "List of languages by total number of speakers".

        Returns a mapping from language names to LanguageInfo objects.
        """
        results: Dict[str, LanguageInfo] = {}
        # Download page and extract just the fragment containing the actual list
        raw_language_list = dl_wikipedia_page('List of languages by total number of speakers')
        raw_language_list = extract_fragment(raw_language_list, "L1+L2", '==')  # type: ignore
        pos = 0  # Position in the list

        # Split table into entries
        for langdata in raw_language_list.split('|-'):
            # We skip the very first entry, which is just (part of the) header
            if pos == 0:
                pos += 1
                continue

            # Split data in first line and rest; extract name and code from the former
            first_line, rest_of_entry = split_first_line_from_rest(langdata)
            name, iso_code, rest_of_first_line = extract_first_wikilink(first_line)

            # The linked text should have a form like "ISO 639:eng", so the actual ISO code
            # follows after the colon
            iso_code = (iso_code.split(':', 1)[1]
                        if iso_code is not None and ':' in iso_code
                        else '??')

            # If there's a note like "excl. [[English-based creole languages|creole languages]]"
            # or "excl. [[Urdu]]", we extract that for the related field
            if rest_of_first_line is not None and 'excl.' in rest_of_first_line:
                exclnode = extract_fragment(rest_of_first_line, 'excl.', ')')
                related = extract_first_wikilink(exclnode)[0]  # type: ignore
            else:
                related = None

            # First wikilink in rest of entry is the language family
            family, _, rest_of_entry = extract_first_wikilink(rest_of_entry)  # type: ignore

            # Second one should be the branch, but we extract it only for families that have a
            # language in the top 10 (actually top 11 since experience shows that two of them
            # will be merged)
            if pos <= 11 and family is not None:
                self.top_families.add(family)

            if family in self.top_families:
                # Extract branch
                branch, _, rest_of_entry = extract_first_wikilink(rest_of_entry)  # type: ignore
            else:
                branch = None

            # Remove trailing '|}' from last entry
            if rest_of_entry.endswith('|}'):
                rest_of_entry = rest_of_entry[:-2].strip()

            # Extract total speaker count (always in the last column) and convert it to millions
            speakers_raw = rest_of_entry[rest_of_entry.rfind('|') + 1:].strip()
            speakers = self._normalize_to_millions(speakers_raw)

            # Create entry and increment position counter
            if name:
                results[name] = LanguageInfo(util.or_default(name, '??'),
                                             util.or_default(iso_code, '??'),
                                             util.or_default(family, '??'),
                                             branch, speakers, related=related)
            pos += 1

        return results

    def _combine_related(self, langdict: Dict[str, LanguageInfo]) -> Set[LanguageInfo]:
        """Combine related languages into a single entry.

        For example, Hindi and Urdu as well as Western Punjabi and Eastern Punjabi will be combined
        (marked as "related"), as will the various varieties of Arabic.

        Additionally this function does two more things:

        1. The 3-letter 639-3 language code is replaced with the corresponding 2-letter 639-1,
           if one exists (e.g. "deu" becomes "de")
        2. The used script (e.g. Latin) is added
        """
        results = set()
        handled_names = set()  # Set of already combined language names
        # Variables for languages that are listed repeatedly and need special treatment
        arabic = None
        punjabi = None

        for langname, langinfo in sorted(langdict.items()):
            # Replace 639-3 with 639-1 code if possible
            iso1_code = self.iso3_to_iso1_map.get(langinfo.iso_code)
            if iso1_code is not None:
                langinfo = dataclasses.replace(langinfo, iso_code=iso1_code)

            # Add the script if known
            script = self.script_map.get(langinfo.iso_code)
            if script is not None:
                langinfo = dataclasses.replace(langinfo, script=script)

            if langname in handled_names:
                continue  # Already handled
            elif 'Arabic' in langname:
                if arabic is None:
                    # Create entry for Arabic
                    arabic = dataclasses.replace(langinfo, name='Arabic', iso_code='ar',
                                                 related=None, script=self.script_map.get('ar'))
                else:
                    # Combine entries
                    arabic = dataclasses.replace(arabic,
                                                 speakers=arabic.speakers + langinfo.speakers)
            elif 'Punjabi' in langname:
                if punjabi is None:
                    # Create entry for Punjabi
                    punjabi = dataclasses.replace(langinfo, name='Punjabi', iso_code='pa',
                                                  related=None)
                else:
                    # Combine entries
                    punjabi = dataclasses.replace(punjabi,
                                                  speakers=punjabi.speakers + langinfo.speakers)
            else:
                if langinfo.related:
                    if langinfo.related in langdict:
                        # Combine directly related languages like Hindi/Urdu
                        handled_names.add(langinfo.related)
                        related_lang = langdict[langinfo.related]
                        related_iso_code = self.iso3_to_iso1_map.get(related_lang.iso_code,
                                                                     related_lang.iso_code)
                        langinfo = dataclasses.replace(
                            langinfo,
                            name=f'{langinfo.name}/{related_lang.name}',
                            iso_code=f'{langinfo.iso_code}/{related_iso_code}',
                            speakers=langinfo.speakers + related_lang.speakers,
                            related=None
                        )
                    elif langinfo.related == 'Malay':
                        # Malay is not sufficiently widespread to be listed, hence we add it
                        # manually
                        langinfo = dataclasses.replace(
                            langinfo,
                            name=f'{langinfo.name}/{langinfo.related}',
                            iso_code=f'{langinfo.iso_code}/ms',
                            related=None
                        )
                    elif not (langinfo.related.islower()            # type: ignore
                              or 'dialects' in langinfo.related):   # type: ignore
                        # If the related entry has only lower-case letters (e.g. 'creole
                        # languages') or looks like 'other Persian dialects' or similar,
                        # it can be skipped -- otherwise we warn
                        logging.warning(f"Don't know how to handle relationship of {langname} to "
                                        f'langinfo.related')
                results.add(langinfo)

        # Add the specially treated languages and return
        if arabic is not None:
            results.add(arabic)
        if punjabi is not None:
            results.add(punjabi)
        return results

    def _postprocess_lang(self, langinfo: LanguageInfo) -> LanguageInfo:
        """Postprocess a language info to fix and enrich its details.

        Specifically, a few needlessly long language names are shortened ("Standard German"
        becomes "German", "Iranian Persian" becomes "Persian").
        """
        name_parts = langinfo.name.split(maxsplit=1)

        if len(name_parts) >= 2 and name_parts[0] in ('Standard', 'Iranian'):
            # Remove "Standard" and "Iranian" from start of name
            langinfo = dataclasses.replace(langinfo, name=name_parts[1])

        return langinfo

    @staticmethod
    def _export_to_csv(filename, langinfos: Sequence[LanguageInfo]) -> None:
        """Export a list of LanguageInfo's into a CSV file."""
        util.rename_to_backup(filename)
        with open(filename, 'w', newline='') as csvfile:
            # Create write and write header line
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(LanguageInfo.header_row())

            for langinfo in langinfos:
                csvwriter.writerow(langinfo.to_row())

    def parselanguagelist(self) -> None:
        """Parses Wikipedia's "List of languages by total number of speakers".

        Extracts and combines the relevant information and groups the languages into two output
        files, as documented in the script header.
        """
        langdict = self._parse_wikipedia_list()
        combined_lang_set = self._combine_related(langdict)
        combined_lang_set = {self._postprocess_lang(langinfo) for langinfo in combined_lang_set}
        mainlist = []
        rest = []
        subfamily_counter: Counter[Tuple[str, Optional[str]]] = Counter()

        # Sort by speaker count (descending) and as fallback alphabetically
        for langinfo in sorted(combined_lang_set,
                               key=lambda langinfo: (-langinfo.speakers, langinfo.name)):
            # Split into main list (2 languages per (sub)family) and the rest
            subfamily = (langinfo.family, langinfo.branch)
            if subfamily_counter[subfamily] < 2:
                mainlist.append(langinfo)
                subfamily_counter[subfamily] += 1
            else:
                rest.append(langinfo)

        # Export to CSV
        self._export_to_csv(util.SOURCELANGS_FILE, mainlist)
        self._export_to_csv('morelangs.csv', rest)


if __name__ == '__main__':
    lang_parser = LangListParser()
    lang_parser.parselanguagelist()

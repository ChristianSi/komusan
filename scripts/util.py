"""Utility functions."""

import csv
from contextlib import contextmanager
from datetime import datetime
import logging
import os
from os import path
import re
from shutil import copyfile
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar
import unicodedata
from warnings import warn


##### Constants and logging #####

# Generic type variable
T = TypeVar('T')  # pylint: disable=invalid-name

# Names of locally used files
CODESCRIPTS_FILE = 'codescripts.csv'
SOURCELANGS_FILE = 'sourcelangs.csv'
EXTRA_DICT = 'extradict.txt'
DICT_FILE = 'dict.txt'
TERM_DICT = 'termdict.txt'

# Directory and file parts of the preparsed kaikki.org dump (created by wiktextract)
KAIKKI_EN_DIR = 'https://kaikki.org/dictionary/English'
KAIKKI_EN_FILE = 'kaikki.org-dictionary-English.jsonl'

# The 3 auxlangs frequently represented in Wiktionary (ignoring Volapük which is only of
# historical interest)
COMMON_AUXLANGS = frozenset('eo ia io'.split())

# The auxlangs we use by default to draw our own candidates from
DEFAULT_AUXLANGS = frozenset('globasa glosa lidepla'.split())

# Configure logging – generally show debug messages, but suppress them for some modules
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)


##### Types #####

class RecordingLogger():
    """A very simple logger that keeps a record of everything logged.

    All messages are immediately print to stdout.

    The 'append_all_messages' method allows appending all messages recorded to far to a file.
    """

    def __init__(self) -> None:
        """Create a new instance."""
        self._messages: List[str] = []

    def info(self, msg: str) -> None:
        """Print an info message to stdout and record it for later."""
        print(msg)
        self._messages.append(msg)

    def warn(self, msg: str) -> None:
        """Print an warn message to stdout and record it for later.

        All warn messages will be prefixed by 'WARNING: '.
        """
        self.info('WARNING: ' + msg)

    def append_all_messages(self, filename: str) -> None:
        """Append all logged messages to 'filename'.

        If the file exists already, two empty lines will be inserted between the existing
        content and the new messages.

        If it doesn't exist, it will be created.

        If there are no message to append, this method does nothing. If there are any,
        the list of messages is cleared after printing it.
        """
        if not self._messages:
            return
        file_exists = path.exists(filename)

        # Create a backup copy if the file exists
        copy_to_backup(filename)

        with open(filename, 'a', encoding='utf8') as outfile:
            if file_exists:
                outfile.write('\n\n')
            for msg in self._messages:
                outfile.write(msg + '\n')

        self._messages = []


##### General utility functions #####

def or_empty(text: Optional[str]) -> str:
    """Returns the given string, or an empty string if its None."""
    return text if text is not None else ''


def or_default(text: Optional[str], default: str) -> str:
    """Returns the given string, or the `default` string if it's None."""
    return text if text is not None else default


##### CSV-related utility functions #####

@contextmanager
def open_csv_reader(filename: str, skip_header: bool = True) -> Iterator[Any]:
    """Open a CSV file for reading.

    If `skip_header` is true (default), the first row is skipped as header row.
    """
    with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        if skip_header:
            next(reader)
        yield reader


@contextmanager
def open_csv_writer(filename: str, create_backup: bool = True) -> Iterator[Any]:
    """Open a CSV file for writing.

    If `create_backup` is true (default) and there is already a file with the specified file, that
    file will be preserved by adding '.bak' to its name.
    """
    rename_to_backup(filename)
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        yield writer


##### Data-related utility functions #####

def coalesce(*args: Optional[T]) -> Optional[T]:
    """Return the first of its arguments that is not None.

    If all arguments are None, then None is returned.
    """
    for arg in args:
        if arg is not None:
            return arg
    return None


def current_datetime() -> str:
    """Return the current date and time as a formatted string, following ISO conventions."""
    return str(datetime.now())[:19]


def extract_key_val(row: Sequence[str], filename: str,
                    ignore_extra_fields: bool = True) -> Tuple[str, str]:
    """
    Extracts a key/value pair from a sequence of strings.

    The first element is treated as key and the second as value.

    Prints a warning if 'row' contains less than two elements.

    If 'ignore_extra_fields' is set False, a warning is also printed if 'row' contains more
    than two elements. (By default, additional elements are ignored.)
    """
    fieldcount = len(row)
    if fieldcount != 2:
        if fieldcount < 2 or not ignore_extra_fields:
            warn(f'Error parsing {filename}: Row "{",".join(row)}" has {fieldcount} '
                 'fields instead of 2')
    key = row[0] if fieldcount else ''
    val = row[1] if fieldcount >= 2 else ''
    return (key, val)


def read_dict_from_csv_file(filename: str, delimiter: str = ',', skip_header_line: bool = True,
                            converter: Callable[[Sequence[str], str], Tuple[str, T]] =
                            extract_key_val) -> Dict[str, T]:  # type: ignore
    """Read a dictionary from a two-column CSV file.

    If 'skip_header_line' is True, the first line is considered a header and skipped.

    The 'converter' function is invoked to convert each row into a key/value pair; it receives
    the 'filename' as second argument, for logging purposes. Defaults to the 'extract_key_val'
    function, which will treat the first field as key and the second as value.

    Prints a warning if any key occurs more than once.
    """
    result: Dict[str, T] = {}
    with open(filename, newline='', encoding='utf-8') as csvfile:
        dictreader = csv.reader(csvfile, delimiter=delimiter)

        if skip_header_line:
            next(dictreader)

        for row in dictreader:
            key, val = converter(row, filename)
            if key in result:
                warn(f'Error parsing {filename}: Key {key} occurs more than once')
            else:
                result[key] = val
    return result


def get_elem(seq: Sequence[str], idx: int, default: str = '') -> str:
    """Safely retried a string from a sequence

    'default' is returned if 'seq' ends before the requested 'idx' position.
    """
    if len(seq) > idx:
        return seq[idx]
    return default


##### IO-related utility functions #####

def copy_to_backup(currentname: str) -> None:
    """Create a backup copy of a file that has '.bak' appended to its name.

    If the file doesn't exist, this function does nothing. If the backup file exists already,
    it will be deleted and overwritten.
    """
    if path.exists(currentname):
        copyfile(currentname, currentname + '.bak')


def dump_file(text: str, filename: str) -> None:
    """Store 'text' in a file called 'filename'."""
    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write(text)


def read_file(filename: str) -> str:
    """Read and return the contents of 'filename'."""
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


def rename_file_if_exists(currentname: str, newname: str) -> None:
    """Rename a file, if it exists. If it doesn't exist, this function does nothing.

    If 'newname' exists already, it will be deleted and overwritten.
    """
    if path.exists(currentname):
        os.replace(currentname, newname)


def rename_to_backup(currentname: str) -> None:
    """Rename a file by adding '.bak' to its name.

    If the file doesn't exist, this function does nothing. If the backup file exists already,
    it will be deleted and overwritten.
    """
    rename_file_if_exists(currentname, currentname + '.bak')


##### Text-related utility functions #####

def capitalize(text: str) -> str:
    """Return a copy of 'text' with its first character capitalized.

    In contrast to the build-in '.capitalize()' method, the rest of the string is left unchanged.

    If 'text' is empty, it is returned as-is.
    """
    if text:
        return text[0].upper() + text[1:]
    return text


def eliminate_parens(text: str) -> str:
    """Eliminate those parts of a text written in parentheses."""
    return re.sub(r'\s*\([^)]*\)', '', text).strip()


def discard_text_in_brackets(text: str) -> str:
    """
    Remove all text contained within square brackets and any whitespace preceding them.

    Returns the input string with all bracketed text and preceding whitespace removed.
    If if those contain any square brackets, the original string is returned unchanged.
    """
    return re.sub(r'\s*\[.*?\]', '', text)


def extract_text_in_brackets(text: str, otherwise_return_fully: bool = True) -> Optional[str]:
    """Extract text between square brackets if present.

    If there is no such text and 'otherwise_return_fully' is true, 'text' is returned completely.
    Otherwise, None is return.
    """
    start = text.find('[')
    end = text.find(']')
    if start != -1 and end > start:
        return text[start + 1:end]
    return text if otherwise_return_fully else None


def format_compact_string_list(strlist: List[str], prefix: str = '  ', sep: str = ',',
                               max_line_length: int = 78) -> str:
    """Format a string list into a compact way.

    Returns a multi-line string that puts each many items of the list into each line as fit
    without surpassing the 'max_line_length'. Items are separated by 'sep' followed by either
    a space or a newline. Each line is prefixed by 'prefix'.

    Note that the list items themselves are never broken, therefore a very long item might
    cause 'max_line_length' to be surpassed.
    """
    if not strlist:
        return ''

    result_arr = []
    # First item in the first line (appended regardless of length)
    line = prefix + strlist.pop(0) + sep

    for item in strlist:
        if len(line) + len(item) + len(sep) < max_line_length:
            line += ' ' + item + sep
        else:
            # Append full line and start new one
            result_arr.append(line)
            line = prefix + item + sep

    # Discard trailing separator
    result_arr.append(line[:-len(sep)])
    return '\n'.join(result_arr)


def gloss_is_informal(gloss: str) -> bool:
    """Check whether a gloss is informal (just an explanation of a word's origins).

    An informal gloss contains spaces; it doesn't contain a plus sign (+) and it doesn't
    start with an equals sign (=).
    """
    return ' ' in gloss and not ('+' in gloss or gloss.startswith('='))


def has_latin_letter(text: str) -> bool:
    """Checks whether text contains at least one Latin letter."""
    return any(unicodedata.name(char).startswith("LATIN") for char in text if char.isalpha())


def normalize(text: str) -> str:
    """Normalize whitespace in a string.

    Any leading or trailing whitespace is discarded; each internal whitespace sequence
    is replaced by a single space.
    """
    return ' '.join(text.strip().split())


def split_on_sep(text: Optional[str], sep: str) -> List[str]:
    """Split a string into parts separated by 'sep'.

    The separator may include optional outer whitespace which is ignored when splitting,
    but kept/restored when the separator is encountered within parentheses

    Whitespace at the beginning and end of all parts is discarded.

    Separators within regular parentheses (such as these) are ignored when splitting.

    If 'text' is None or empty, an empty list will be returned.

    Otherwise, if there are no separators, the returned list will have just one element.
    """
    if not text:
        return []
    text = text.strip()
    actual_sep = sep.strip()
    sep_in_parens = rf'\([^)]*{re.escape(actual_sep)}[^)]*\)'
    need_to_handle_parens = bool(re.search(sep_in_parens, text))
    result = re.split(rf'\s*{re.escape(actual_sep)}\s*', text)

    if need_to_handle_parens:
        # Remerge elements if the separator occurs within parentheses
        wait_for_end = False
        merged_result: List[str] = []
        for elem in result:
            if wait_for_end:
                # Merge with preceding element
                merged_result[-1] += sep + elem
                if ')' in elem and '(' not in elem:
                    wait_for_end = False
            else:
                if '(' in elem and ')' not in elem:
                    wait_for_end = True
                merged_result.append(elem)
        return merged_result

    return result


def split_on_commas(text: Optional[str]) -> List[str]:
    """Split a string into its comma-separated parts, ignoring commas in parentheses."""
    return split_on_sep(text, ', ')


def split_on_pipes(text: Optional[str]) -> List[str]:
    """Split a string into its pipe-separated (|) parts, ignoring pipes in parentheses."""
    return split_on_sep(text, ' | ')


def split_on_semicolons(text: Optional[str]) -> List[str]:
    """Split a string into its semicolon-separated parts, ignoring semicolons in parentheses."""
    return split_on_sep(text, '; ')


def split_text_and_explanation(text: str) -> Tuple[str, Optional[str]]:
    """Split a text that ends in explanation enclosed in parentheses into its parts.

    If the text has the form 'main text (explanation'), the parts 'main text' and 'explanation'
    will be returned as a tuple.

    If the text doesn't end in a parentheses, a tuple of the original 'text' followed by None will
    be returned.
    """
    text = text.strip()
    if text.endswith(')') and '(' in text:
        opening_paren = text.rfind('(')
        main_text = text[:opening_paren].strip()
        explanation = text[opening_paren + 1:-1].strip()
        return main_text, explanation
    return text, None


##### CLI-related utility functions #####

def retrieve_single_arg(default: Optional[str] = None):
    """Returns a single command-line argument.

    If two or more arguments were specified, exits with an error message.

    If no argument was specified and an explicit 'default' is set, it is returned.
    Otherwise, exits with an error message.
    """
    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            sys.exit('error: Too many arguments')
        return sys.argv[1]
    if default is not None:
        return default
    sys.exit('error: One argument required, but none given')

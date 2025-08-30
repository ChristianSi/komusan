""" LineDict and related types and functions. """

from abc import ABC, abstractmethod
from typing import Dict, Iterator, Mapping, MutableMapping, Optional, Sequence
from warnings import warn

import util


class ToStringDict(ABC):
    """Interface for objects that can be converted into a dictionary of string/string pairs."""
    # pylint: disable=too-few-public-methods

    @abstractmethod
    def to_dict(self) -> Mapping[str, str]:
        """Convert this object into a dictionary of string/string pairs."""


class LineDict(MutableMapping[str, str], ToStringDict):
    """
    A dictionary that contains line number information to generate meaningful messages.

    Since the dictionary is read from a text file, all keys and values must be strings.

    To add key/value, call add(key, value, lineno).

    Accessing keys and values works in the usual way.

    To get the line number of a key, call lineno(key).

    The get the line number of the first key/value pair, call first_lineno().
    """

    def __init__(self, filename: Optional[str] = None) -> None:
        """Create a new instance.

        The optional 'filename' argument is stored as a field (self.filename) and used in
        error messages.
        """
        self.filename = filename
        self._store: Dict[str, str] = {}
        self._first_lineno: Optional[int] = None
        self._lines: Dict[str, int] = {}

    def errprefix(self, lineno: int = -1) -> str:
        """Return the prefix to use for error messages (warnings).

        If self.filename is known, it is included in the prefix.
        If lineno is set to a non-negative value, it is likewise included.
        """
        if self.filename:
            lineinfo = f', line {lineno}' if lineno >= 0 else ''
            return f'Error parsing {self.filename}{lineinfo}: '
        lineinfo = f' on line {lineno}' if lineno >= 0 else ''
        return f'Error{lineinfo}: '

    def __getitem__(self, key: str) -> str:
        """Return the value for a key."""
        return self._store[key]

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the keys in this dictionary."""
        return iter(self._store)

    def __len__(self) -> int:
        """Return the number of key/value pairs in this dictionary."""
        return len(self._store)

    def add(self, key: str, value: str, lineno: int = -1, allow_replace: bool = False) -> None:
        """Add a key/value pair to the dictionary.

        If the key already exists in a dictionary, it will be replaced. Unless 'allow_replace'
        is True, a warning will be printed.

        If this method is called for the first time, 'lineno' will also be stored as firstline.
        Note that subsequent calls will not override this value, even if they come with a
        smaller 'lineno' value.
        """
        if key in self._store and not allow_replace:
            warn(f'{self.errprefix(lineno)}Duplicate key "{key}"')
        self._store[key] = value
        self._lines[key] = lineno
        if self._first_lineno is None:
            self._first_lineno = lineno

    def append_to_val(self, key: str, append_text: str) -> None:
        """Appends 'append_text' at the end of the value stored for 'key'.

        Throws a KeyError if that key doesn't yet exist in the dictionary.
        """
        self._store[key] += append_text

    def __setitem__(self, key, item):
        """Add a key/value pair to the dictionary using the standard dict protocol.

        Note that it's necessary to call 'add' instead to properly set a line number.
        If called like this, the 'first_lineno' will be re-used for the new entry.
        """
        self.add(key, item, util.coalesce(self._first_lineno, -1))

    def __delitem__(self, key):
        """Delete the specified key. Raises a KeyError if *key* is not in the map."""
        del self._store[key]

    def first_lineno(self) -> int:
        """Return the number of the first key added to this dictionary.

        Returns -1 if the dictionary is empty.
        """
        return util.coalesce(self._first_lineno, -1)  # type: ignore

    def lineno(self, key: str) -> int:
        """Return the line number associated with the key.

        Returns -1 if the key is unknown.
        """
        return util.coalesce(self._lines.get(key), -1)  # type: ignore

    def to_dict(self) -> MutableMapping[str, str]:
        """Return this dictionary itself."""
        return self

    def __repr__(self) -> str:
        """Return an unambiguous string with the class name and contents."""
        return f'{self.__class__.__name__}({self._store!r})'

    def __str__(self) -> str:
        """Return a user-friendly string of key: value pairs, newline-separated."""
        return '\n'.join(f'{key}: {value}' for key, value in self._store.items())


def dict_from_str(entry: str, firstline: int = 1, filename: Optional[str] = None) -> LineDict:
    """Parse a multi-line string into a dictionary structure.

    Each line must contain a key/value pair, separated by ':'. Whitespace around the
    separator or at the end of the line is ignored.

    There shouldn't be any empty or whitespace-only lines in the string. If such a line is
    encountered, a warning is printed and it is otherwise ignored.

    If a line does not contain a colon, it is ignored and a is printed.
    In case of duplicate keys, a warning is printed and the last value is used.
    If a key is preceded by a single space or by tabs, a warning is printed, but the key is
    accepted (without the initial whitespace).

    Empty keys are not allowed – in such cases, a warning is printed and the line is discarded.

    Lines starting with '#' (after possible whitespace) are considered comments and ignored.

    Continuation lines start with at least two spaces, followed by at least one printable
    character. They continue the value started or continued on the preceding line.
    The first two spaces on the line are discarded, while the rest of the line is added to
    the value, separated by a line break (which is therefore preserved).

    If a dictionary starts with a continuation line or if a such line follows a comment, a
    warning is printed and the line is ignored.

    It is an error to start a line with a single space or a tab followed by printable
    characters, unless the rest of the line is a comment. In such cases, a warning is printed
    and the line is ignored.

    'firstline' is the line number of the first line, used to add line number information
    to the dictionary and to generate meaningful warning messages.
    """
    lines = entry.splitlines()
    lineno = firstline - 1
    result = LineDict(filename)
    last_key_added = None  # None: initial value, '': last line was comment or empty

    for line in lines:
        lineno += 1
        line = line.rstrip()

        if line.startswith('  '):
            # Continuation line
            if last_key_added:
                value = line[2:]
                result.append_to_val(last_key_added, '\n' + value)
            else:
                if last_key_added == '':
                    error_cond = 'after a comment or empty line'
                else:
                    error_cond = 'at the start of a dictionary'
                warn(f'{result.errprefix(lineno)}Unexpected continuation line {error_cond}: '
                     f'"{line.strip()}"')
            continue

        if not line or line.lstrip().startswith('#'):
            # Skip empty and comment lines – the former trigger a warning
            if not line:
                warn(f'{result.errprefix(lineno)}Line is empty')
            last_key_added = ''
            continue
        if line[0] in (' ', '\t'):
            warn(f'{result.errprefix(lineno)}Invalid leading whitespace: "{line}"')
            line = line.lstrip()

        parts = line.split(':', 1)

        if len(parts) < 2:
            warn(f'{result.errprefix(lineno)}No valid key/value pair found: "{line}"')
            continue

        key = parts[0].strip()
        value = parts[1].strip()
        last_key_added = key

        if not key:
            warn(f'{result.errprefix(lineno)}Ignoring invalid empty key: "{line}"')
            continue

        result.add(key, value, lineno)

    return result


def read_dicts_from_file(filename: str) -> Sequence[LineDict]:
    """Read multiple dicts from a text file and return them in order.

    Dicts are separated by one or more empty lines and must comply with the format described in
    'dict_from_str'.
    """
    lineno = 1
    result = []

    with open(filename, newline='', encoding='utf8') as infile:
        # Split into entries (separated by empty lines)
        entries = infile.read().split('\n\n')

        for entrystr in entries:
            entry = dict_from_str(entrystr, lineno, filename)
            if entry:
                result.append(entry)
            lineno += entrystr.count('\n') + 2

    # If the last element is empty, we remove it altogether (this will happen esp. in case of
    # empty files)
    if result and not result[-1]:
        result.pop()

    return result


def read_single_dict_from_file(filename: str) -> LineDict:
    """Read a single dicts from a text file and return it.

    The dict must comply with the format described in 'dict_from_str'.
    """
    with open(filename, newline='', encoding='utf8') as infile:
        return dict_from_str(infile.read(), 1, filename)


def dump_dicts(dict_objects: Sequence[ToStringDict], filename: str) -> None:
    """Dump a sequence of dictionary-serializable objects into a file.

    Dictionaries are serialized using the format described in 'dict_from_str' and
    separated by an empty line. Keys are serialized in the order they have in the
    dict.  All whitespace in keys and values is normalized.
    """
    first = True
    with open(filename, 'w', encoding='utf8') as outfile:
        for obj in dict_objects:
            if first:
                first = False
            else:
                outfile.write('\n')
            outfile.write(stringify_dict(obj))


def stringify_dict(dict_object: ToStringDict) -> str:
    """Convert a dictionary-serializable object into a string.

    The object is serialized using the format described in 'dict_from_str'. Keys are
    serialized in the order they have in the dict.  All whitespace in keys and values is
    normalized.
    """
    dct = dict_object.to_dict()
    result = ''
    for key, value in dct.items():
        norm_key = util.normalize(key)
        norm_value = util.normalize(value)
        result += f'{norm_key}: {norm_value}\n'
    return result

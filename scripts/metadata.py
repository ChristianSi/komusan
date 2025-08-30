"""Classes and functions for managing metadata."""

from warnings import warn

import linedict


class ValueProvider():
    """Provide translations of metadata into the configured language."""
    # pylint: disable=too-few-public-methods

    def __init__(self, lang: str, values_file: str = 'transvalues.txt') -> None:
        """Create a new instance, using 'lang' as language.

        'values_file' is the file from which value strings and their translations will be read.
        """
        self.lang = lang
        self.values_file = values_file
        dct: dict[str, str] = {}
        line_dicts = linedict.read_dicts_from_file(values_file)

        for line_dict in line_dicts:
            # store mapping from "value" to 'lang' code
            value = line_dict.get('value')
            trans = line_dict.get(lang)
            if value is None:
                warn(f'{values_file} entry starting on line {line_dict.first_lineno()} lacks a '
                     '"value" field')
                continue
            if trans is None:
                warn(f'{values_file} entry starting on line {line_dict.first_lineno()} lacks a '
                     f'translation into "{lang}"')
                continue
            dct[value] = trans
        self.dict = dct

    def lookup(self, value: str) -> str:
        """Lookup the translation into the configured language of the specified 'value'.

        If no translation is listed in the 'values_file', the value itself is returned, with
        ' (?)' appended.
        """
        result = self.dict.get(value)
        if result is None:
            result = value + ' (?)'
        return result

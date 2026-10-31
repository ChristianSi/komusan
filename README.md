# Komusan, an easy and fair language for global communication

**Komusan** is meant to become an easy-to-learn, logical and well-balanced
auxiliary language for global usage. Its name is shortened from **komun
lisan** "common tongue", which is a poetic way to say 'common language'.
Komusan's goal is to allow people who otherwise don't share a common
language to communicate effectively and on an equal footing.

Its grammar is largely isolating and it is based on three other auxiliary
language (auxlangs): [Globasa](https://www.globasa.net/eng),
[Lidepla](https://en.wikipedia.org/wiki/Lingwa_de_planeta), and
[Glosa](https://en.wikipedia.org/wiki/Glosa), aiming to combine the best of
their traits and elements without needlessly reinventing the wheel.

## Source languages

Komusan's words are built from 24 source languages, listed in
[data/sourcelangs.csv](data/sourcelangs.csv). They are the most widely
spoken languages according to the total speaker counts (first plus second
language speakers) from *Ethnologue*, as given in Wikipedia's [List of
languages by total number of
speakers](https://en.wikipedia.org/wiki/List_of_languages_by_total_number_of_speakers).
To keep the selection balanced, at most two languages from the same language
branch (or family) are included. The selection is made by the script
`scripts/parselanguagelist.py`.

The list was first determined in September 2024, based on the *Ethnologue*
2024 data. In October 2026, it was rechecked against the *Ethnologue* 2026
data; for this, a hard limit of 24 languages was added to keep the original
number. The recheck confirmed the original set of languages – only their
speaker counts and order changed – which suggests that it's a fairly stable
choice. Without the limit, Yoruba would have been added as the 25th
language.

## License

The software in this repository is licensed under the [ISC
license](https://en.wikipedia.org/wiki/ISC_license), a permissive free
software license. See the file [LICENSE.txt](LICENSE.txt) for the full
license text.

The language Komusan – including the dictionary contained in this
repository and all related data files – is placed in the public domain as
per the [CC0 1.0 Universal (CC0 1.0) Public Domain
Dedication](https://creativecommons.org/publicdomain/zero/1.0/deed.en).

## Files in this repository

The rest of file document explains the other files that can be found in
this repository, including their purposes and their formats.

### The data directory

This directory contains files used to generate the Komusan dictionary as
well as automatically generated files on the website. Most files in it can
be read as plain text, but they are written in specific formats that make
them easily parsable by computers.

### The scripts directory

This directory contains the Python scripts used to add new words to the
Komusan dictionary, to print statistics on it, and convert it in a more
human-accessible form. For more details, see the intro comments within the
various `py` (Python) files.

Additionally, this directory also contains a few supporting files that
allow checking the Python for inconsistencies and other issues.

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

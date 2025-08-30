"""Utilities for helping with vocabulary building."""

from __future__ import annotations
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import util


##### Constants and precompiled regexes #####

# Used for logging
LOG = util.RecordingLogger()

# A pseudo language code used for added hybrid candidates
ADDED_LANG = 'added'

# Regex patterns describing how a syllable in a candidate word should look like
INITIAL_NON_SEMIVOWELS = 'bCdfghjklmnprsStvz'
SEMIVOWELS = 'wy'
INITIAL_CONSONANTS = INITIAL_NON_SEMIVOWELS + SEMIVOWELS
ALL_CONSONANTS = INITIAL_CONSONANTS + 'N'
ALL_CONSONANTS_SET = frozenset(ALL_CONSONANTS)
ALL_NON_SEMIVOWELS = INITIAL_NON_SEMIVOWELS + 'N'
# Set of word-final consonants (excluding the semivowels which are instead treated as part of
# failing diphthongs)
WORD_FINAL_CONSONANTS = 'klmnNprst'
SYLLABLE_FINAL_CONSONANTS = WORD_FINAL_CONSONANTS + 'bdg'
SECOND_CONSONANTS = 'lrwy'
ALL_CONSONANTS_RE = re.compile(rf'[{ALL_CONSONANTS}]')
# Consonants not allowed at the end of a word (except for semivowels which may occur as part
# of a failing diphtong)
NOT_WORD_FINAL_NON_SEMIVOWEL = 'CSbdfghjvz'   # Consonants not allowed at the end of a word
# Consonants not allowed at the end of a syllable (except for semivowel)
NOT_SYLLABLE_FINAL_NON_SEMIVOWEL = 'CSfhjvz'
NOT_SECOND_CONSONANTS = 'bCdfghjkmnpsStvz'   # Consonants not allowed after another one
SIMPLE_VOWELS = 'aeiou'
# Vowels used in the internal representation
INTERNAL_VOWELS = SIMPLE_VOWELS + 'ə'
# Pipe-separated list of falling diphthongs (internal representation)
FALLING_DIPHTHONGS = 'ay|aw|ew|oy'
# 'iw' and 'uy' as falling diphthongs (not followed by a vowel)
IU_FALLING_DIPHTHONGS_RE = re.compile(
    rf'(iw|uy)(?!([{SIMPLE_VOWELS}]))')
# These falling diphthongs are likewise not allowed unless a vowel follows
ILLEGAL_FALLING_DIPHTHONGS = 'ey|iy|ow|uw'
ILLEGAL_FALLING_DIPHTHONGS_RE = re.compile(
    rf'({ILLEGAL_FALLING_DIPHTHONGS})(?!([{SIMPLE_VOWELS}]))')
# Patterns matching the semivowels in cases where they should be spelled as vowels (after vowels
# unless another vowel follows)
W_RE = re.compile(rf'([{INTERNAL_VOWELS}])w(?![{INTERNAL_VOWELS}])')
Y_RE = re.compile(rf'([{INTERNAL_VOWELS}])y(?![{INTERNAL_VOWELS}])')
INTERNAL_VOWEL_RE = re.compile(rf'[{INTERNAL_VOWELS}]')
##CAND_VOWEL_FRAG = f'(?:a[IU]|oI)|[{SIMPLE_VOWELS}ə]'
# Capturing and non-capturing regexes matching a candidate vowel
##CAND_VOWEL_RE_CAPT = re.compile('(' + CAND_VOWEL_FRAG + ')')
##CAND_VOWEL_RE_NONCAPT = re.compile('(?:' + CAND_VOWEL_FRAG + ')')
# The two consonants are captured in two subgroups
ILLEGAL_CONS_PAIR_RE = re.compile(
    f'([{NOT_SYLLABLE_FINAL_NON_SEMIVOWEL}])([{NOT_SECOND_CONSONANTS}])')
# Matches 'ks' at the start of a word
INITIAL_KS_RE = re.compile(r'^ks')
# Matches 'ts' at the start of words and after syllable-final consonants.
TS_COMBI_RE = re.compile(rf'(?:^|(?<=[{SYLLABLE_FINAL_CONSONANTS}]))ts')
# R or S followed by a nasal and or T and another consonant that's not allowed here -- the first
# two are captured in one subgroup, the final consonant in another
ILLEGAL_RN_TRIPLE_RE = re.compile(f'([rs][mnt])([{NOT_SECOND_CONSONANTS}])')
# Matches sequences of syllable-final consonants followed by one of 'sk/sp/st' before another
# letter -- the last two are captured (and hence kept) together
ILLEGAL_SKPT_TRIPLE_RE = re.compile(f'([{SYLLABLE_FINAL_CONSONANTS}])(s[kpt].)')
# Sequences of three consonants at the start of a word, the middle of which is a syllable-final
# consonant – the first one and the last two are captured in separate subgroups, since we
# can insert a filler vowel between them
INITIAL_CONS_TRIPLE_RE = re.compile(
    rf'\b([{INITIAL_CONSONANTS}])([{SYLLABLE_FINAL_CONSONANTS}][{INITIAL_CONSONANTS}])')
# Sequences of two syllable-final consonants followed by another one are also illegal
# -- the first two and the last one are captured in separate subgroups
ILLEGAL_CONS_TRIPLE_RE = re.compile(
    f'([{SYLLABLE_FINAL_CONSONANTS}][{SYLLABLE_FINAL_CONSONANTS}])([{NOT_SECOND_CONSONANTS}])')
# If two consonants that might start a syllable are followed by another one, we keep the first
# two together (first subgroup) – we don't accept semivowels as second consonant here, since
# they might be converted to the corresponding vowels instead
ILLEGAL_CONS_TRIPLE2_RE = re.compile(
    f'([{INITIAL_NON_SEMIVOWELS}][lr])([{ALL_NON_SEMIVOWELS}])')
# 'ny' before another consonant, which is captured in a subgroup
NY_BEFORE_CONSONANT = re.compile(f'ny([{INITIAL_CONSONANTS}])')
# A double consonant such as 'nn' or 'rr' -- the consonant itself is captured in a subgroup
DOUBLE_CONS_RE = re.compile(rf'([{ALL_CONSONANTS}])\1')
# Matches 'n' and 'r' between Pinyin vowels (lowercase only) – the first vowel as captured in
# a group, the rest in another. Also captured the sequence 'nr' in this position, which will be
# divided between the two groups.
NR_BETWEEN_VOWELS = re.compile(r'([aeiouü]n?)([nr][aeiouü])')
# Matches 'i' and 'u' (first group) before another Spanish vowel (second group) -- but not in
# cases where another semivowel is allowed after two consonants
IU_BEFORE_VOWEL = re.compile(
    rf'(?<![{INITIAL_CONSONANTS}][{SECOND_CONSONANTS}])([iu])([aeiouáéíóú])')
# 'h' before another consonant, excluding semivowels (the consonant is captured)
H_BEFORE_CONSONANT = re.compile(f'h([{ALL_NON_SEMIVOWELS}])')
# An R-colored schwa at the end of English words (in IPA)
FINAL_R_COLORED_SCHWA = re.compile(r'(ɚ|əɹ|\(ə\)ɹ)$')
# Words ending in a certain vowel(a, e, u) after a single final consonant or another vowel)
A_AFTER_FINAL_CONSONANT = re.compile(rf'[^{ALL_CONSONANTS}][{WORD_FINAL_CONSONANTS}]?a$')
E_AFTER_FINAL_CONSONANT = re.compile(rf'[^{ALL_CONSONANTS}][{WORD_FINAL_CONSONANTS}]?e$')
U_AFTER_FINAL_CONSONANT = re.compile(rf'[^{ALL_CONSONANTS}][{WORD_FINAL_CONSONANTS}]?u$')
# Matches 'c' before any character except 'e' and 'i'
##C_BEFORE_NON_EI = re.compile(r'c(?![ei])')
# Matches 'ng' not followed by a vowel of one of the consonants allowed after 'g'
##ILLEGAL_NG = re.compile(rf'ng(?![lrv{SIMPLE_VOWELS}])')

# Vowels with acute accents (incl. y), used for some languages
ACUTE_ACCENTS = 'áéíóúý'
ACUTE_ACCENT_RE = re.compile(rf'[{ACUTE_ACCENTS}]')
# Equivalent of the above, but without accents
VOWELS_WO_ACCENT = 'aeiouy'
VOWEL_WO_ACCENT_RE = re.compile(rf'[{VOWELS_WO_ACCENT}]')

# A normal vowel followed by one marked with an acute accent -- both captured as groups
ACUTE_ACCENT_AFTER_VOWEL = re.compile(rf'([{VOWELS_WO_ACCENT}])([{ACUTE_ACCENTS}])')

# Sequences of common punctuation
PUNCTUATION_RE = re.compile(r'[-,.…;:!?"/()]+')

# Replacements needed to convert the internal form used for candidate word to the external form
# used in the dictionary.
EXPORT_REPL = {
    'sh': "s-h",  # Insert hyphen to prevent misreading
    'C': "ch",
    'Ng': 'ng',
    'Nk': 'nk',
    'N': 'ng',
    'S': 'sh',
    'ə': 'e',
}

# Replacements needed for normalization (to avoid minimal pairs).
NORM_REPL = {
    'v': 'u',
    'w': 'u',
    'y': 'i',
    'z': 's',
    '-': '',
}


##### Types #####

@dataclass
class Conversion():
    """The result of a phonetic conversion."""
    output: str
    penalty: bool  # Whether a penalty should be applied


@dataclass
class Candidate():
    """A word in (nearly) our phonology considered a candidate for selection.

    The phonology used here has a few differences to the generally used one:

    * The semivowels are always written as "w" and "y".
    * "C", "S" and "N" are used instead of "ch", "sh" and "ng" to follow the "one sound, one
      letter" principle
    * Vowels inserted to fit our phonology are written "ə" instead of "e" – such vowels
      are ignored when comparing similarities between candidate pairs
    """
    # pylint: disable=too-many-instance-attributes
    word: str
    penalty: int  # The penalty incurred when converting the source word into our phonology
    lang: str               # The code of the language from which the candidate was adapted
    original: str = ''      # The original word, romanization, or IPA (empty if not set)

    # The actual original word if 'original' is the romanization or IPA
    true_original: str | None = None

    # The names of any auxlangs that might be related
    auxlangs: Collection[str] = field(default_factory=list)

    raw_psim: int = -1        # The raw similarity penalty (distance to other candidates)
    simscore: float = -1.0    # The similarity score, normalized from 0 (worst) to 1 (best)
    related_cands: Dict[str, List[Candidate]] = field(
        default_factory=lambda: defaultdict(list), compare=False)
    filled = False      # set to True by insert_filler_vowels

    @property
    def dscore(self) -> float:
        """The distortion score, from 1 (no distortion) to 0 (highly distorted)

        Calculated by diving the raw (conversion) penalty by 5 and subtracting it from 1,
        but with 0 as lower bound.
        """
        return max(1 - self.penalty / 5.0, 0.0)

    @property
    def total_score(self) -> float:
        """The total score: simscore times dscore."""
        return self.simscore * self.dscore

    def export_word(self) -> str:
        """Return the external form of the word that will be used in the dictionary."""
        return export_word(self.word)

    def count_related_natlang_cands(self) -> int:
        """Return the number of natural languages with related candidates.

        All languages not listed in 'auxlangs' are considered natural.
        """
        return sum(1 for lang in self.related_cands if lang not in self.auxlangs)

    def has_suitable_related_natlang_cands(self) -> bool:
        """Check whether this candidate has natlang candidates that make it suitable to be elected.

        Generally, *any* related natlang candidate is enough for this.

        But for Glosa, we additionally require that there must be one related candidate from
        a non-Romance language, since otherwise Glosa's Greek/Latin/Romance-based vocabulary
        could easily cause the Romance source language to gain undue influence.
        """
        dont_count = list(self.auxlangs)
        if self.lang == 'glosa':
            dont_count += ['es', 'fr']
        for lang in self.related_cands:
            if lang not in dont_count:
                return True
        return False

    def find_langs_with_identical_candidate(self) -> Set[str]:
        """Find languages whose candidate are identical to this one.

        This checks the 'related_cands' field and returns a set of zero or more language codes.
        """
        result = set()
        word = self.export_word()

        for lang, cand_list in self.related_cands.items():
            for cand in cand_list:
                if cand.export_word() == word:
                    result.add(lang)
                    break

        return result

    @property
    def show_original(self) -> str:
        """Show the original word.

        If both 'true_original' (usually in a non-Latin script) and 'original'
        (the romanization) are set, they'll be slash-separated.

        'original' is omitted, for English words,  as it's usually the IPA, which is less
        interesting. It's likewise omitted if 'true_original' includes Latin letters, since in
        that case 'original' tends to indicate a more precise pronunciation rather than being
        the romanization.
        """
        if self.true_original and (self.lang == 'en' or util.has_latin_letter(self.true_original)):
            return self.true_original
        originals = [orig for orig in (self.true_original, self.original) if orig]
        result = '/'.join(originals).strip()
        # Sometimes there is still whitespace around slashes which we trim
        result = re.sub(r'\s*/\s*', '/', result)
        return result

    @property
    def syllables(self) -> float:
        """Returns the number of syllables in this candidates.

        If the candidate starts and ends with a vowel, the syllable count is reduced by 0.5
        For example, "nana", "anan", and "nanan" are counted as having 2 syllables, but "ana"
        has only 1.5.
        """
        count: float = count_vowels_internal(self.word)
        if count and self.word[0] in INTERNAL_VOWELS and self.word[-1] in INTERNAL_VOWELS:
            count -= 0.5
        return count

    def insert_filler_vowels(self) -> None:
        """Insert filler vowels where necessary to make the phonology of this candidate valid.

        The penalty is increased accordingly.

        Also cleans up the phonology in some other ways, e.g. by simplifying 'tx' to 'C' and
        by eliminating double consonants.

        Note that this method will run only ONCE per candidate – subsequent invocations will
        simply do nothing. The reason for this is that otherwise the distortion penalties
        would be overcounted.
        """
        # pylint: disable=too-many-branches, too-many-statements
        if self.filled:
            return  # Avoid having this method run twice
        # Replace punctuation by spaces
        text = PUNCTUATION_RE.sub(' ', self.word)

        words = text.split()
        new_words = []

        for word in words:
            # In the mapping, we use 'X' for IPA /x/ -- it becomes 'h' before vowels (where that
            # sound is allowed, no penalty), 'k' otherwise (with penalty).
            word = re.sub(rf'X([{INTERNAL_VOWELS}])', r'h\1', word)
            if self.lang == 'es':
                # In Spanish, 'h' is also used before semivowels
                word = re.sub(rf'X([{SEMIVOWELS}])', r'h\1', word)
            x_count = word.count('X')
            if x_count:
                word = word.replace('X', 'k')
                self.penalty += x_count

            # Simplify tx/tc to c
            word = word.replace('tx', 'c').replace('tc', 'c')
            # Likewise 'iy' to 'i' and 'uw' to 'u'
            word = word.replace('iy', 'i').replace('uw', 'u')

            # Eliminate double consonants
            word = DOUBLE_CONS_RE.sub(r'\1', word)

            # 'ny' before a consonant is simplified by dropping the 'y' (without a penalty,
            # since the pronunciation difference is pretty small) – except in German, where it
            # tends to represent 'ni'
            if self.lang != 'de':
                word = NY_BEFORE_CONSONANT.sub(r'n\1', word)

            # Semivowels between consonants are changed to the corresponding vowel
            # (without a penalty)
            word = re.sub(f'([{ALL_CONSONANTS}])y(?=[{ALL_CONSONANTS}])', r'\1i', word)
            word = re.sub(f'([{ALL_CONSONANTS}])w(?=[{ALL_CONSONANTS}])', r'\1u', word)

            # 'ks' at the start of words is simplified to 's', likewise 'dz' to 'z' -- incurs a
            # penalty since one sound is lost)
            word, count = INITIAL_KS_RE.subn('s', word)
            self.penalty += count
            word, count = re.subn(r'^dz', 'z', word)
            self.penalty += count

            # 'ts' at the start of words and after syllable-final consonants is simplified to 's'
            # (incurs a penalty since one sound is lost)
            word, count = TS_COMBI_RE.subn('s', word)
            self.penalty += count

            # Final 'iy' is simplified to 'i' (without penalty)
            if word.endswith('iy'):
                word = word[:-1]

            # 'iw' and 'uy' as falling diphthongs are converted to the related vowel pairs 'iu'
            # and 'ui' (without a penalty)
            word = IU_FALLING_DIPHTHONGS_RE.sub(lambda m: 'iu' if m.group(1) == 'iw' else 'ui',
                                                word)

            # Final 'ey' in French words ending in -eil or -ille(s) (e.g. vieille, conseil) is
            # changed to the double vowel 'ei' (without penalty)
            if (self.lang == 'fr' and word[-1] == 'y'
                    and re.search(r'(eil|illes?)\b', self.original)
                    and len(word) >= 2 and word[-2:] in ILLEGAL_FALLING_DIPHTHONGS):
                word = word[:-1] + 'i'

            # Other illegal falling diphthongs are simplified to just the first vowel
            # (with a penalty)
            word, num_replacements = ILLEGAL_FALLING_DIPHTHONGS_RE.subn(lambda m: m.group(1)[0],
                                                                        word)
            if num_replacements:
                self.penalty += num_replacements

            # Final 'h' is dropped (with penalty)
            if word.endswith('h'):
                word = word[:-1]
                self.penalty += 1

            # As is 'h' before another consonant (excluding semivowels)
            word, count = H_BEFORE_CONSONANT.subn(r'\1', word)
            self.penalty += count

            # Final N (ng) needs a vowel before it (penalties for added vowels are calculated
            # below)
            word = re.sub(rf'([{ALL_NON_SEMIVOWELS}])N$', r'\1əN', word)

            # Simplify nN to N
            word = word.replace('nN', 'N')

            # Add vowel before syllable-initial N or Ng
            word = re.sub(rf'(^|[{SYLLABLE_FINAL_CONSONANTS}])(Ng?[{SIMPLE_VOWELS}lr])', r'\1ə\2',
                          word)

            # After a vowel, simplify "SC" to "sC" to make it valid across a syllable boundary,
            # with a penalty (occurs in some, especially Russian words)
            word, num_replacements = re.subn(rf'([{INTERNAL_VOWELS}])SC', r'\1sC', word)
            if num_replacements:
                self.penalty += num_replacements

            # Simplify any remaining instances of "SC" to just "C", with a penalty
            sc_count = word.count('SC')
            if sc_count:
                word = word.replace('SC', 'C')
                self.penalty += sc_count

            # Final 'z' after a vowel becomes 's' (with penalty)
            if re.search(rf'[{INTERNAL_VOWELS}]z$', word):
                word = word[:-1] + 's'
                self.penalty += 1

            # Insert filler vowels between illegal pairs and triples of consonants
            word = word.replace('rld', 'rəld')
            word = ILLEGAL_SKPT_TRIPLE_RE.sub(r'\1ə\2', word)

            # In German words, the frequent combination "Sv" (schw) before a vowel becomes
            # "Sw" to avoid a filler vowel -- without a penalty, since they are fairly similar
            if self.lang == 'de':
                word = re.sub(rf'Sv([{INTERNAL_VOWELS}])', r'Sw\1', word)

            # We run this repeatedly because sometimes there are three (or more?) matching
            # consonants in a row
            count = 1
            while count:
                word, count = ILLEGAL_CONS_PAIR_RE.subn(r'\1ə\2', word)

            word = ILLEGAL_RN_TRIPLE_RE.sub(r'\1ə\2', word)
            word = re.sub(r'([np])mn', r'\1mən', word)
            word = word.replace('stl', 'stəl')
            word = word.replace('stv', 'stəv')
            word = word.replace('smr', 'səmr')

            # Prepend a filler vowel before initial "s", followed by a consonant pair allowed to
            # start a syllable (e.g. "street")
            word = re.sub(rf'^(s[{INITIAL_NON_SEMIVOWELS}][{SECOND_CONSONANTS}])', r'ə\1', word)

            word = INITIAL_CONS_TRIPLE_RE.sub(r'\1ə\2', word)
            word = ILLEGAL_CONS_TRIPLE_RE.sub(r'\1ə\2', word)
            word = ILLEGAL_CONS_TRIPLE2_RE.sub(r'\1ə\2', word)

            # Simplify initial 'wh' to 'w' (without penalty) -- those are usually imports from
            # English where the 'h' is not actually pronounced
            word = re.sub(r'^wh', 'w', word)

            # Convert initial semivowels followed by consonant not allowed in 2nd position to
            # the corresponding vowel
            word = re.sub(rf'^([{SEMIVOWELS}])([{NOT_SECOND_CONSONANTS}])',
                          lambda m: ('u' if m.group(1) == 'w' else 'i') + m.group(2),
                          word)

            # Semivowels occurring as third consonant in a syllable (e.g. in "try" or "plw")
            # are converted to the corresponding vowel (without a penalty)
            word = re.sub(
                rf'(^|[{SYLLABLE_FINAL_CONSONANTS}])([{INITIAL_NON_SEMIVOWELS}][lr])([wy])',
                lambda m: m.group(1) + m.group(2) + ('u' if m.group(3) == 'w' else 'i'),
                word)
            word = re.sub(
                rf'([{INTERNAL_VOWELS}{SEMIVOWELS}][{NOT_SYLLABLE_FINAL_NON_SEMIVOWEL}][lr])'
                r'([wy])',
                lambda m: m.group(1) + ('u' if m.group(2) == 'w' else 'i'),
                word)

            # But if two of them occur in a row (e.g. "gwy" or "hyw"), the first one becomes
            # a vowel
            word = re.sub(
                rf'(^|[{SYLLABLE_FINAL_CONSONANTS}])([{INITIAL_NON_SEMIVOWELS}])(wy|yw)',
                lambda m: m.group(1) + m.group(2) + ('uy' if m.group(3) == 'wy' else 'iw'),
                word)
            word = re.sub(
                rf'([{INTERNAL_VOWELS}{SEMIVOWELS}][{NOT_SYLLABLE_FINAL_NON_SEMIVOWEL}])(wy|yw)',
                lambda m: m.group(1) + ('uy' if m.group(2) == 'wy' else 'iw'),
                word)

            # "Nək" after a syllable-final consonant is simplified to "nək", since usually the 'N'
            # was just the result of a phonetic assimilation between the two consonants
            word = re.sub(rf'([{SYLLABLE_FINAL_CONSONANTS}])Nək', r'\1nək', word)

            # If a word has just two consonants, add a vowel between them if that results in a
            # legal word (e.g. "st" -> "set")
            if (len(word) == 2 and word[0] in INITIAL_NON_SEMIVOWELS
                    and word[1] in WORD_FINAL_CONSONANTS and word[1] not in SECOND_CONSONANTS):
                word = word[0] + 'ə' + word[1]

            # Prepend a vowel if that leads to a valid sequence in a word starting with two
            # consonants
            if (len(word) >= 2 and word[0] in SYLLABLE_FINAL_CONSONANTS
                    and word[1] in NOT_SECOND_CONSONANTS):
                word = 'ə' + word

            # Most consonants are not allowed at the end of words, nor are pairs of consonants;
            # final semivowels are only allowed in certain falling diphthongs
            if (word and word[-1] in NOT_WORD_FINAL_NON_SEMIVOWEL) or (
                    len(word) > 2 and (word[-2] in INITIAL_CONSONANTS
                                       and word[-1] in SECOND_CONSONANTS
                                       or word[-2] in SYLLABLE_FINAL_CONSONANTS
                                       and word[-1] in INITIAL_CONSONANTS
                                       or word[-1] in SEMIVOWELS
                                       and word[-2:] not in FALLING_DIPHTHONGS)):
                word += 'ə'

            # Add a final vowel if the word contains consonants, but no vowels
            if ALL_CONSONANTS_RE.search(word) and not INTERNAL_VOWEL_RE.search(word):
                word = word + 'ə'

            # Just 'N' needs a vowel before (rather than after) it
            if word == 'Nə':
                word = 'əN'

            # A spurious 'ə' may have been inserted between 'Ng' -- we can safely remove it
            word = word.replace('Nəg', 'Ng')

            # Eliminate double consonants again (since new ones might have been added by the
            # previous rules)
            word = DOUBLE_CONS_RE.sub(r'\1', word)

            # Eliminate the double 'ii' that occurs in some Russian words
            if self.lang == 'ru':
                word = word.replace('ii', 'i')

            self.penalty += word.count('ə')
            new_words.append(word)
        self.word = ' '.join(new_words)
        self.filled = True

    def validate(self) -> Optional[str]:
        """Check that this candidate is valid.

        Returns an error message if this is not the case.
        """
        # Check that all characters are valid
        unexpected_set = set(self.word) - set(ALL_CONSONANTS + INTERNAL_VOWELS + ' ')
        if unexpected_set:
            err_msg = (f'{self.lang} candidate contains unexpected characters: '
                       f'{self.word} ({''.join(sorted(unexpected_set))})')
            LOG.warn(err_msg)
            return err_msg

        words = self.word.split()

        for word in words:
            # Split at vowels (including falling diphthongs) and examine the non-vowel parts
            # (which may be empty)
            parts = re.split(rf'(?:{FALLING_DIPHTHONGS}|[{INTERNAL_VOWELS}])', word)

            last_idx = len(parts) - 1
            for idx, part in enumerate(parts):
                if not part:
                    continue
                if idx == last_idx and (len(part) > 1 or part not in WORD_FINAL_CONSONANTS):
                    return f'"{part}" is not allowed at the end of words ({word})'
                if idx > 0 and part[0] in SYLLABLE_FINAL_CONSONANTS:
                    part = part[1:]  # discard consonant ending the preceding syllable
                if not part:
                    continue
                if (len(part) > 2 or part[0] not in INITIAL_CONSONANTS
                        or (len(part) == 2 and part[1] not in SECOND_CONSONANTS)):
                    return f'"{part}" is not allowed at the start of syllables ({word})'
        return None

    def __str__(self) -> str:
        """Return a compact string representation."""
        return f'{self.lang}:{self.export_word()}'

    def show_info(self) -> str:
        """Return a detailed string representation."""
        result = (f'{self.lang}:{self.export_word()} (T:{self.total_score:.3f} – '
                  f'C:{self.dscore:.3f}×S:{self.simscore:.3f}, ')

        identical_cand_count = len(self.find_langs_with_identical_candidate())
        if identical_cand_count:
            if identical_cand_count == 1:
                result += '1 identical candidate, '
            else:
                result += f'{identical_cand_count} identical candidates, '

        if self.related_cands:
            related_natlang_cands = self.count_related_natlang_cands()
            result += f'{related_natlang_cands} related natlang candidate'
            if related_natlang_cands != 1:
                result += 's'

            language_str = 'language' if len(self.related_cands) == 1 else 'languages'
            result += f', related candidates in {len(self.related_cands)} {language_str}: '
            rel_cands: List[Candidate] = []
            for rel_lang in sorted(self.related_cands):
                rel_cands += self.related_cands[rel_lang]
            result += ', '.join(str(cand) for cand in rel_cands) + ')'
        else:
            result += 'no related candidates)'

        return result


class Constraints:
    """Stores and validates the constraints a candidate must fulfill to be eligible for selection.
    """

    def __init__(self, constraint_str: str):
        """Create a new instance, based on the contents of a "Constraint" value in the auxfile.

        Note that constraints are case-sensitive. Multiple constraints are separated by semicolons.
        """
        self.max_syllables = None
        self.allowed_langs = None
        self.allowed_langs_rationale = None
        # A mapping from candidates to be skipped to the reason for skipping them
        self.skip = {}
        self.added = None
        self.adding_rationale = None
        self.allowshort = False
        self.compound = None
        self.compound_rationale = None
        self.chosen = None
        self.chosen_rationale = None
        self.target_class = None
        self.merge_with = None
        self.premerge = False

        constraints = util.split_on_semicolons(constraint_str)
        for constraint in constraints:
            # Maximum syllable count, e.g. "Syllables:1" or "Syllables:2.5"
            value = constraint.removeprefix('Syllables:').strip()
            if value != constraint:
                self.max_syllables = float(value)
                continue

            # Only languages from the listed set are accepted, with the rationale specified in
            # parentheses, e.g.
            # "Allow langs: globasa glosa tl vi (We prefer a particle rather than an affix)"
            value = constraint.removeprefix('Allow langs:').strip()
            if value != constraint:
                allowed_langs, self.allowed_langs_rationale = util.split_text_and_explanation(value)
                # Multiple language codes are whitespace-separated
                self.allowed_langs = set(allowed_langs.split())
                continue

            # Allow short: Relax the minimum length requirement of content words, instead
            # accepting candidates of arbitrary length
            if constraint == 'Allow short':
                self.allowshort = True
                continue

            # Choose the specified candidate, skipping all others, with the rationale specified
            # in parentheses, e.g.
            # "Choose:ka (closely related to the first candidate -ika, but shorter)"
            value = constraint.removeprefix('Choose:').strip()
            if value != constraint:
                self.chosen, self.chosen_rationale = util.split_text_and_explanation(value)
                continue

            # Skip a candidate, with the rationale given in parentheses, e.g.
            # "Skip:un (only used before vowels)". Can be specified repeatedly
            value = constraint.removeprefix('Skip:').strip()
            if value != constraint:
                skipped_word, rationale = util.split_text_and_explanation(value)
                self.skip[skipped_word] = rationale
                continue

            # The following aren't actual constraints, but influence the selection and target entry

            # Add a hybrid candidate that should then also be chosen, with the rationale given
            # in parentheses, e.g. 'Add:me (Derived by blending the candidates "men" and "ma"
            # since neither is suitable by itself)'
            value = constraint.removeprefix('Add:').strip()
            if value != constraint:
                self.added, self.adding_rationale = util.split_text_and_explanation(value)
                continue

            # Add this as a compound (whose parts must already exist), optionally with the
            # rationale given in parentheses, e.g. "Compound:li-su" or
            # "Compound:li-su (This is the logical way to express this)"
            value = constraint.removeprefix('Compound:').strip()
            if value != constraint:
                self.compound, self.compound_rationale = util.split_text_and_explanation(value)
                continue

            # Change the class (POS) of the generated entry, e.g. "Set class:particle"
            value = constraint.removeprefix('Set class:').strip()
            if value != constraint:
                self.target_class = value
                continue

            # Merge this concept with an existing Komusan word, e.g. "Merge:ku"
            value = constraint.removeprefix('Merge:').strip()
            if value != constraint:
                self.merge_with = value
                continue

            # Premerge: like Merge, but put the new translations before instead of after the
            # existing ones
            value = constraint.removeprefix('Premerge:').strip()
            if value != constraint:
                self.merge_with = value
                self.premerge = True
                continue

            raise ValueError(f'Unknown constraint: {constraint}')

    def fails(self, cand: Candidate) -> str:
        """Check whether a candidate fails any of the defined constraints.

        If yes, a descriptive string is returned, e.g. "too long".

        If not, the empty string is returned.
        """
        if self.max_syllables is not None and cand.syllables > self.max_syllables:
            return 'too long'

        if self.allowed_langs and cand.lang and cand.lang not in self.allowed_langs:
            # In addition to the explicitly allowed languages we also allow those without a
            # language code (added via Add: constraint)
            return self.allowed_langs_rationale or ''

        cand_word = cand.export_word()
        if cand_word in self.skip:
            return self.skip[cand_word] or ''

        if self.chosen and cand_word != self.chosen:
            return 'not the chosen candidate'

        return ''

    def __str__(self):
        """Return a string representation."""
        constraints = []
        if self.max_syllables is not None:
            constraints.append(f'up to {self.max_syllables} syllables')

        if self.allowshort:
            constraints.append('short candidates are allowed')

        if self.allowed_langs:
            constraints.append(f'allowed languages: {", ".join(sorted(self.allowed_langs))} '
                               f'(rationale: {self.allowed_langs_rationale})')

        if self.skip:
            constraints.append(f'skipped candidates: {", ".join(sorted(self.skip))}')

        if self.added:
            constraints.append(f'added candidate: {self.added} (rationale: '
                               f'{self.adding_rationale})')

        if self.chosen:
            constraints.append(f'chosen candidate: {self.chosen} (rationale: '
                               f'{self.chosen_rationale})')

        if self.compound:
            if self.compound_rationale:
                rationale = f' (rationale: {self.compound_rationale})'
            else:
                rationale = ''
            constraints.append(f'model as compound: {self.compound}{rationale}')

        if self.target_class:
            constraints.append(f'target class: {self.target_class}')

        if self.merge_with:
            merge_type = 'premerge' if self.premerge else 'merge'
            constraints.append(f'{merge_type} with: {self.merge_with}')

        if constraints:
            return "Constraints: " + '; '.join(constraints)
        return 'No Constraints'


##### Helper functions #####

@lru_cache(maxsize=None)
def export_word(word: str) -> str:
    """Convert a word from the internal form to the external form used in the dictionary."""
    # Adapt semivowel (diphthong) spellings
    word = W_RE.sub(r'\1u', word)
    word = Y_RE.sub(r'\1i', word)

    # Change internal to external representations
    for key, value in EXPORT_REPL.items():
        word = word.replace(key, value)
    return word


@lru_cache(maxsize=None)
def normalize_word(word: str) -> str:
    """Normalize a word to avoid adding minimal pairs in the dictionary.

    * Case is converted to lower-case.
    * Final 'ng' is converted to just 'n'.
    * 'w' and 'y' are converted to their vowel equivalents, and 'v' likewise to 'u'.
    * 'z' is converted to 's'.
    * Whitespace and hyphens (marking affixes and separating word parts) are deleted.
    """
    word = word.lower()
    word = re.sub(r'\s+', '', word)
    if word.endswith('ng'):
        word = word[:-1]
    for key, value in NORM_REPL.items():
        word = word.replace(key, value)
    return word


def count_vowels_internal(word: str) -> int:
    """Count the vowels in the internal representation of a Kikomun word.

    Note that this won't work in the external representation, where some vowel letters
    actually represent semivowels.
    """
    return sum(1 for char in word.lower() if char in INTERNAL_VOWELS)


def extract_phonetic_conversion_rule(row: Sequence[str], filename: str) -> Tuple[str, Conversion]:
    """Extract a phonetic conversion rule from a line in a CSV file.

    The line should have 3 fields:

    1. The input characters (IPA)
    2. The output characters (our phonology)
    3. Whether the conversion should incur a penalty ('1') or not ('0')
    """
    if len(row) != 3:
        LOG.warn(f'Error parsing {filename}: Row "{",".join(row)}" has {len(row)} fields '
                 'instead of 3.')
    inchar = util.get_elem(row, 0)
    outchar = util.get_elem(row, 1)
    rawpenalty = util.get_elem(row, 2, '1')
    if rawpenalty == '0':
        penalty = False
    else:
        penalty = True
        if rawpenalty != '1':
            LOG.warn(f'Unexpected penalty in {filename}: "{rawpenalty}" instead of "0" or "1".')
    return inchar, Conversion(outchar, penalty)

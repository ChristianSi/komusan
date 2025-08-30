#!/usr/bin/env perl
# List the words in the dictionary, including their English translations and their sources.
# Reads "dict.txt" unless another file name is given as command-line argument.

use strict;
use warnings;

@ARGV = ('dict.txt') unless @ARGV;

# Initialize variables
my ($word, $infl, $en, $infl_formatted);
my $has_infl = 0;  # Flag to check if infl is present

# Read input file line by line
while (<>) {
    next unless /^(word|infl|en):/;  # Skip unrelated lines
    chomp;

    if (/^word:\s*(.*)/) {
        $word = $1;  # Capture the word
    }
    elsif (/^infl:\s*(.*)/) {
        $infl = $1;  # Capture and format list of influences
        $infl =~ s/([^\s;]+)\s*\(([^)]+)\)/$1: $2/g;
        $infl =~ s/([^\s;]+)\s*$([^)]+)$/$1: $2/g;  # infl
        $has_infl = 1;  # Set infl flag
    }
    elsif (/^en:\s*(.*)/) {
        $en = $1;  # Capture the English translation
        $en =~ s/\s*;\s*/, /g;  # Format English translation

        # Print the output, including sources (infl) if they exist
        $infl_formatted = $has_infl ? " (*sources:* $infl)" : '';
        print "* **$word** – $en$infl_formatted\n";

        # Reset variables for the next entry
        ($word, $infl, $en) = (undef, undef, undef);
        $has_infl = 0;
    }
}

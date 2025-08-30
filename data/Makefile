all: walsfeatures.csv

termdict:
	../scripts/updatekaikki.py
	../scripts/parsewikt.py

walsfeatures.csv: sourcelangs.csv cldf/values.csv
	../scripts/walsfeaturefreq.py

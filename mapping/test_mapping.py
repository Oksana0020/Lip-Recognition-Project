"""
Bozkurt Viseme Mapping Loader
Script loads Bozkurt viseme–phoneme mapping from CSV file and
constructs a dictionary that maps individual phonemes to their corresponding
viseme classes. The resulting mapping is used during dataset preprocessing
to convert phoneme labels into viseme labels.
"""

import csv
from pathlib import Path

mapping_file = Path('./mapping/bozkurt_viseme_map.csv')
viseme_map = {}

with mapping_file.open('r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Row: {row}")
        viseme_class = row['viseme_class'].strip()
        phonemes_str = row['phonemes'].strip()
        print(f"  Class: '{viseme_class}', Phonemes string: '{phonemes_str}'")
        phonemes = [p.strip() for p in phonemes_str.split(',')]
        print(f"  Split phonemes: {phonemes}")
        for phoneme in phonemes:
            viseme_map[phoneme] = viseme_class

print(f'\nTotal mappings: {len(viseme_map)}')
print(f"All mapped phonemes: {sorted(viseme_map.keys())}")
print(f"AA in map: {'AA' in viseme_map}")
print(f"B in map: {'B' in viseme_map}")

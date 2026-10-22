"""Rebuild res/networks/networks.csv, the list of every network in res/networks.

One row per BIF in res/networks/BIF: its source (bnRep when the name starts with
"bnrep_", bnlearn otherwise), nodes and arcs counted from the BIF, the bnRep
application area and structure origin, whether a gold-standard .xbif exists, and
how many client datasets BBDD/<net>/ holds. Area and structure are kept from the
current networks.csv (bnrep_export.R fills them for new bnRep networks).

Usage (from the repository root): python res/networks/generators/networks_manifest.py
"""
import csv
import glob
import os
import re

ROOT = 'res/networks'
MANIFEST = os.path.join(ROOT, 'networks.csv')

old = {}
if os.path.exists(MANIFEST):
    with open(MANIFEST, encoding='utf-8') as f:
        old = {r['name']: r for r in csv.DictReader(f)}

rows = []
for path in sorted(glob.glob(os.path.join(ROOT, 'BIF', '*.bif')), key=str.lower):
    name = os.path.basename(path)[:-4]
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    nodes = len(re.findall(r'^\s*variable\s', text, re.M))
    arcs = sum(len([p for p in m.group(1).split('|')[1].split(',') if p.strip()])
               for m in re.finditer(r'probability\s*\(\s*([^)]*)\)', text) if '|' in m.group(1))
    prev = old.get(name, {})
    rows.append({
        'name': name,
        'source': 'bnRep' if name.startswith('bnrep_') else 'bnlearn',
        'nodes': nodes,
        'arcs': arcs,
        'area': prev.get('area', ''),
        'structure': prev.get('structure', ''),
        'xbif': int(os.path.exists(os.path.join(ROOT, 'xbif', name + '.xbif'))),
        'samples': len(glob.glob(os.path.join(ROOT, 'BBDD', name, name + '.*.csv*'))),
    })

with open(MANIFEST, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
print(f'{len(rows)} networks written to {MANIFEST}')

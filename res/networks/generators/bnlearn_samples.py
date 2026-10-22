"""Sample the client datasets of the bnlearn networks.

Each network gets 100 datasets of 5000 rows drawn from its BIF by forward
sampling, written as res/networks/BBDD/<net>/<net>.<i>.csv.gz. Existing files
are kept, so an interrupted run resumes.

The datasets in the repository were drawn with this procedure (python bnlearn,
methodtype='bayes') but without a fixed seed, so a rerun gives different
samples, not the same files.

Usage (from the repository root): python res/networks/generators/bnlearn_samples.py [net ...]
"""
import os
import sys

import bnlearn as bn
from tqdm import tqdm

NETWORKS = ['alarm', 'andes', 'asia', 'barley', 'cancer', 'child', 'diabetes', 'earthquake',
            'hailfinder', 'hepar2', 'insurance', 'link', 'mildew', 'munin', 'munin1', 'munin2',
            'munin3', 'munin4', 'pathfinder', 'pigs', 'survey', 'water', 'win95pts']
N_DATASETS = 100
N_ROWS = 5000

for network in sys.argv[1:] or NETWORKS:
    print('\n' + network)
    model = bn.import_DAG('res/networks/BIF/' + network + '.bif', checkmodel=False)
    out_dir = 'res/networks/BBDD/' + network
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(N_DATASETS)):
        path = out_dir + '/' + network + '.' + str(i) + '.csv.gz'
        if os.path.exists(path):
            continue
        sample = bn.sampling(model, n=N_ROWS, methodtype='bayes')
        sample.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0})

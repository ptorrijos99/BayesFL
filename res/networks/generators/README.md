# Networks and their datasets

Layout of `res/networks/`:

| Path | Content |
|---|---|
| `BIF/<net>.bif` | Network (structure and CPTs) in textual BIF |
| `xbif/<net>.xbif` | The same network in XMLBIF, read by the Java code as gold standard |
| `net/`, `txt/` | Other exports of the bnlearn networks |
| `BBDD/<net>/<net>.<i>.csv.gz` | Client datasets: 100 samples of 5000 rows per network |
| `BBDD_old/`, `BBDD_noIID/` | Datasets of earlier experiments (flat layout; non-IID splits) |
| `networks.csv` | One row per network: source, nodes, arcs, bnRep area and structure origin, whether an `.xbif` exists, number of datasets |

Datasets are stored gzipped. The Java readers (`Utils.readData`, `Weka_Instances`)
take the path of the plain file and read `<path>.gz` when the plain file is
missing, so experiment code keeps using `.csv` and `.arff` paths.

## Sources

- **bnlearn**: networks from the [bnlearn repository](https://www.bnlearn.com/bnrepository/).
  Datasets drawn by forward sampling with python bnlearn (`bnlearn_samples.py`).
- **bnRep**: the 202 discrete networks of the R package
  [bnRep](https://cran.r-project.org/package=bnRep) 0.0.6, exported with
  `bnrep_export.R` (state and node names sanitised, undefined CPT columns set
  to uniform, samples drawn with `bnlearn::rbn`). Its summary table lists a
  second `asia`, but the package ships only the 8-node one.
  Prefixed `bnrep_` because bnRep and bnlearn share names (`earthquake`).
  `bnrep_forest` and `bnrep_tbm` have no `.xbif`: the pgmpy conversion of their
  BIFs fails.

No sampler fixed a seed, so the datasets in the repository cannot be redrawn
identically; they are the reference data of the experiments.

## Scripts

Run from the repository root.

| Script | Purpose |
|---|---|
| `bnlearn_samples.py [net ...]` | Datasets of the bnlearn networks |
| `bnrep_export.R [minNodes] [nClients] [nRows] [onlyNet]` | BIF and datasets of the bnRep networks |
| `bif_to_xbif.py <bifDir> <outDir> [--only NAME]` | BIF to XMLBIF (pgmpy) |
| `networks_manifest.py` | Rebuilds `networks.csv` from the files |
| `sachs_discretize.R` | Discretisation of the Sachs et al. (2005) data |

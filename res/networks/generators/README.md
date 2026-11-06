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
  `bnrep_export.R`. Prefixed `bnrep_` because bnRep and bnlearn share names
  (`earthquake`). pgmpy cannot convert `bnrep_forest` and `bnrep_tbm`, so they
  have no `.xbif`.

No sampler fixed a seed, so the datasets cannot be redrawn identically. Both
sets are on Zenodo: [bnlearn](https://doi.org/10.5281/zenodo.14917795) and
[bnRep](https://doi.org/10.5281/zenodo.23561524).

Rows of 34 client files (andes.46 and 33 files of 17 bnRep networks) were split
or merged by concurrent writers when they were saved, and andes.46 was cut at
2819 rows. `repair_clients.R` dropped the damaged rows and topped each file up
to 5000 rows with seeded draws from the network; valid rows were kept as they
were. The repaired files replace the originals in later versions of both
Zenodo records (version 1.1); the links above resolve to the latest version.

## Scripts

Run from the repository root.

| Script | Purpose |
|---|---|
| `bnlearn_samples.py [net ...]` | Datasets of the bnlearn networks |
| `bnrep_export.R [minNodes] [nClients] [nRows] [onlyNet]` | BIF and datasets of the bnRep networks |
| `bif_to_xbif.py <bifDir> <outDir> [--only NAME]` | BIF to XMLBIF (pgmpy) |
| `networks_manifest.py` | Rebuilds `networks.csv` from the files |
| `repair_clients.R <net> <client> [...]` | Repairs client datasets with damaged rows |
| `sachs_discretize.R` | Discretisation of the Sachs et al. (2005) data |

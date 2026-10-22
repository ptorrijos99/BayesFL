# Export bnRep networks into the layout the Java pipeline expects.
#
# For each selected network this writes
#   res/networks/BIF/bnrep_<name>.bif        (structure + CPTs, BIF 0.15)
#   res/networks/BBDD/bnrep_<name>/bnrep_<name>.<i>.csv.gz   (client samples)
# and records the bnRep area and structure origin in res/networks/networks.csv.
#
# The .xbif that the Java side reads as the gold standard is produced from the
# .bif by bif_to_xbif.py, because bnlearn writes textual BIF and weka's
# BIFReader wants XMLBIF. networks_manifest.py then recounts nodes and arcs.
#
# Everything is prefixed "bnrep_" for two reasons: provenance is visible in every
# result row, and bnRep ships an `earthquake` with 40 nodes while bnlearn's has
# 5 — without the prefix the export would silently destroy existing data.
#
# No seed is set: the samples in the repository cannot be redrawn identically.
#
# Usage (from the repository root):
#   Rscript res/networks/generators/bnrep_export.R [minNodes] [nClients] [nRows] [onlyNet]
# The environment variable BNREP_MAX_NODES caps the size of the selected
# networks (e.g. minNodes 0 and BNREP_MAX_NODES=20 for the small ones).

args <- commandArgs(trailingOnly = TRUE)
minNodes <- if (length(args) > 0) as.integer(args[1]) else 20
nClients <- if (length(args) > 1) as.integer(args[2]) else 100
nRows    <- if (length(args) > 2) as.integer(args[3]) else 5000
onlyNet  <- if (length(args) > 3) args[4] else NA

suppressMessages(library(bnRep))
suppressMessages(library(bnlearn))
data(bnRep_summary)

s <- bnRep_summary
maxNodes <- as.integer(Sys.getenv("BNREP_MAX_NODES", "1000000"))
sel <- s[s$Type == "Discrete" & s$Nodes > minNodes & s$Nodes <= maxNodes, ]
if (!is.na(onlyNet)) sel <- sel[sel$Name == onlyNet, ]
cat("exporting", nrow(sel), "networks\n")

bifDir <- "res/networks/BIF"
dir.create(bifDir, recursive = TRUE, showWarnings = FALSE)

# bnRep state names contain spaces and punctuation ("Last day", "One or less").
# pgmpy's BIF parser splits parent-state tuples on whitespace and then tries to
# read a state name as a probability, so the conversion to XMLBIF fails. Levels
# are therefore sanitised BEFORE anything is written, by rebuilding the bn.fit
# from its own CPTs -- that way the .bif and the sampled CSVs carry identical
# category names, which the Java gold-standard loader requires.
# volleyball names a parent "NEUROTICISM" in its children's CPTs while the node
# is "neuroticism", and leaves its root CPTs without dimension names. When the
# CPT dimension names do not match the nodes, they are matched to the nodes
# ignoring case and the DAG is rebuilt from the CPTs. The same path renames
# nodes whose names hold spaces or symbols ("sampling place" in dioxins), which
# BIF parsers reject, replacing them by "_".
cptsOf <- function(bn) {
  nds <- nodes(bn)
  clean <- gsub("[^A-Za-z0-9_.-]", "_", nds)
  probs <- lapply(nds, function(nd) bn[[nd]]$prob)
  names(probs) <- nds
  ok <- identical(clean, nds) && all(vapply(nds, function(nd) {
    nm <- names(dimnames(probs[[nd]]))
    !is.null(nm) && nm[1] == nd && all(nm[-1] %in% nds)
  }, logical(1)))
  if (ok) return(list(dag = bn.net(bn), probs = probs))
  stopifnot(!anyDuplicated(tolower(clean)))
  cat("   CPT dimension or node names repaired\n")
  d <- empty.graph(clean)
  out <- list()
  for (k in seq_along(nds)) {
    p <- probs[[nds[k]]]
    nm <- names(dimnames(p))
    pa <- if (length(dim(p)) > 1) clean[match(tolower(nm[-1]), tolower(nds))] else character(0)
    if (anyNA(pa)) stop("unmatched parent of ", nds[k])
    names(dimnames(p)) <- c(clean[k], pa)
    for (q in pa) d <- set.arc(d, q, clean[k])
    out[[clean[k]]] <- p
  }
  list(dag = d, probs = out)
}

sanitiseLevels <- function(bn) {
  cp <- cptsOf(bn)
  d <- cp$dag
  cpts <- lapply(nodes(d), function(nd) {
    p <- cp$probs[[nd]]
    # A few networks were fitted by maximum likelihood and leave NA in the
    # columns of parent configurations absent from their data (poultry,
    # student1). Those columns become uniform, as in bn.fit's own default.
    if (anyNA(p)) {
      k <- dim(p)[1]
      cols <- matrix(p, nrow = k)
      cols[, colSums(is.na(cols)) > 0] <- 1 / k
      p[] <- cols
      cat("  ", nd, ": NA columns set to uniform\n")
    }
    dn <- dimnames(p)
    nm <- names(dn)
    # Replacing punctuation can collapse two distinct levels onto the same name
    # ("Below 0" and "Below-0", or "<0" and ">0"), which bnlearn rejects as
    # duplicate dimension names. make.unique keeps them distinct. Level ORDER is
    # identical wherever a variable appears, so a parent gets the same mapping in
    # its own CPT and in each child's.
    dimnames(p) <- lapply(dn, function(v) make.unique(gsub("[^A-Za-z0-9]", "_", v), sep = "_"))
    names(dimnames(p)) <- nm
    p
  })
  names(cpts) <- nodes(d)
  custom.fit(d, cpts)
}

manifest <- data.frame()
for (k in seq_len(nrow(sel))) {
  nm <- sel$Name[k]
  out <- paste0("bnrep_", nm)
  ok <- tryCatch({
    data(list = nm, package = "bnRep", envir = environment())
    bn <- sanitiseLevels(get(nm, envir = environment()))

    # bnRep objects are bn.fit; a few may carry zero-probability rows that break
    # sampling, so failures are recorded rather than aborting the whole export.
    write.bif(file.path(bifDir, paste0(out, ".bif")), bn)

    dataDir <- file.path("res/networks/BBDD", out)
    dir.create(dataDir, recursive = TRUE, showWarnings = FALSE)

    # Resuming is only safe if the existing files were sampled at the SAME size.
    # Skipping on mere existence would silently mix 1000-row and 5000-row clients
    # in one network, which no downstream analysis could detect.
    existing <- list.files(dataDir, pattern = "[.]csv[.]gz$", full.names = TRUE)
    if (length(existing) > 0) {
      have <- length(readLines(gzfile(existing[1]), warn = FALSE)) - 1L
      if (have != nRows) {
        cat("  rows differ (have", have, "want", nRows, ") - regenerating", out, "\n")
        unlink(existing)
      }
    }

    for (i in 0:(nClients - 1)) {
      f <- file.path(dataDir, paste0(out, ".", i, ".csv.gz"))
      if (file.exists(f)) next
      smp <- rbn(bn, nRows)
      con <- gzfile(f, "w")
      write.csv(smp, con, row.names = FALSE, quote = FALSE)
      close(con)
    }
    TRUE
  }, error = function(e) { cat("  FAILED", nm, ":", conditionMessage(e), "\n"); FALSE })

  if (ok) {
    manifest <- rbind(manifest, data.frame(
      name = out, nodes = sel$Nodes[k], arcs = sel$Arcs[k],
      area = as.character(sel$Area[k]), structure = as.character(sel$Structure[k])))
    cat(sprintf("%3d/%3d %-28s %4d nodes %4d arcs\n", k, nrow(sel), out,
                sel$Nodes[k], sel$Arcs[k]))
  }
}

# Area and structure go to networks.csv; nodes, arcs and the other columns are
# recounted from the files by networks_manifest.py.
manifestFile <- "res/networks/networks.csv"
all <- if (file.exists(manifestFile)) read.csv(manifestFile, stringsAsFactors = FALSE) else
  data.frame(name = character(), source = character(), nodes = integer(), arcs = integer(),
             area = character(), structure = character(), xbif = integer(), samples = integer())
for (k in seq_len(nrow(manifest))) {
  row <- manifest[k, ]
  if (!(row$name %in% all$name)) {
    all[nrow(all) + 1, ] <- list(row$name, "bnRep", row$nodes, row$arcs, "", "", 0L, 0L)
  }
  all$area[all$name == row$name] <- row$area
  all$structure[all$name == row$name] <- row$structure
}
write.csv(all, manifestFile, row.names = FALSE, quote = FALSE, na = "")
cat("\nexported", nrow(manifest), "networks; run bif_to_xbif.py and networks_manifest.py next\n")

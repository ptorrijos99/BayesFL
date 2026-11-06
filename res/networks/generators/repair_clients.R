# Repair client datasets damaged when they were written.
#
# Some client CSVs of res/networks/BBDD hold rows that were split or merged by
# concurrent writers (fields from two rows on one line, or a row cut in two), and
# andes.46 was truncated after 2819 rows. Tetrad either stops at such a line or
# reads it silently with a spurious category, so every pipeline that reads these
# files is affected.
#
# For each file this keeps, in their original order, the rows that have the
# header's number of fields and a valid state in every column, drops the others,
# and brings the file back to exactly nRows rows: extra rows are cut from the end
# and missing ones are drawn from the network (res/networks/BIF/<net>.bif) by
# forward sampling with a seed derived from the network and client. Valid rows
# are never changed, so the first N rows of a file only change when a damaged
# line lay among them.
#
# Usage (from the repository root):
#   Rscript res/networks/generators/repair_clients.R <net> <client> [<client> ...]
# The repaired file replaces res/networks/BBDD/<net>/<net>.<client>.csv.gz; the
# original stays in git history.

suppressMessages(library(bnlearn))

args <- commandArgs(trailingOnly = TRUE)
net <- args[1]
clients <- as.integer(args[-1])
nRows <- 5000

sanitise <- function(v) gsub("[^A-Za-z0-9_.-]", "_", v)
# bnlearn cannot parse a few exported BIFs (bnrep_forest); those networks are taken
# from bnRep itself, with node and state names sanitised as bnrep_export.R does
bn <- tryCatch(read.bif(file.path("res/networks/BIF", paste0(net, ".bif"))), error = function(e) {
  suppressMessages(library(bnRep))
  nm <- sub("^bnrep_", "", net)
  data(list = nm, package = "bnRep", envir = environment())
  b <- get(nm, envir = environment())
  probs <- lapply(nodes(b), function(nd) {
    p <- b[[nd]]$prob
    dn <- dimnames(p)
    nmn <- sanitise(names(dn))
    dimnames(p) <- lapply(dn, function(v) make.unique(gsub("[^A-Za-z0-9]", "_", v), sep = "_"))
    names(dimnames(p)) <- nmn
    p
  })
  names(probs) <- sanitise(nodes(b))
  d <- empty.graph(sanitise(nodes(b)))
  for (nd in names(probs)) for (q in names(dimnames(probs[[nd]]))[-1]) d <- set.arc(d, q, nd)
  custom.fit(d, probs)
})
levelsOf <- lapply(bn, function(nd) dimnames(nd$prob)[[1]])

for (i in clients) {
  path <- file.path("res/networks/BBDD", net, sprintf("%s.%d.csv.gz", net, i))
  con <- gzfile(path, "rt")
  lines <- readLines(con, warn = FALSE)
  close(con)
  header <- lines[1]
  cols <- strsplit(header, ",", fixed = TRUE)[[1]]
  body <- lines[-1]
  body <- body[nzchar(body)]
  parts <- strsplit(body, ",", fixed = TRUE)
  okLen <- vapply(parts, length, integer(1)) == length(cols)
  # valid states, compared without case: the bnlearn samples write "true", the BIF "True"
  lev <- lapply(cols, function(cn) tolower(levelsOf[[cn]]))
  okVal <- okLen
  okVal[okLen] <- vapply(parts[okLen], function(p) all(mapply(function(v, l) tolower(v) %in% l, p, lev)), logical(1))
  keep <- body[okVal]
  dropped <- length(body) - length(keep)

  added <- 0
  if (length(keep) > nRows) keep <- keep[seq_len(nRows)]
  if (length(keep) < nRows) {
    need <- nRows - length(keep)
    seed <- sum(utf8ToInt(net)) * 1000 + i
    set.seed(seed)
    s <- rbn(bn, n = need)
    # spell every state as the file already does (case), falling back to the BIF
    kept <- do.call(rbind, strsplit(keep, ",", fixed = TRUE))
    out <- matrix("", need, length(cols))
    for (k in seq_along(cols)) {
      v <- as.character(s[[cols[k]]])
      seen <- unique(kept[, k])
      m <- seen[match(tolower(v), tolower(seen))]
      out[, k] <- ifelse(is.na(m), v, m)
    }
    keep <- c(keep, apply(out, 1, paste, collapse = ","))
    added <- need
  }
  con <- gzfile(path, "wt")
  writeLines(c(header, keep), con)
  close(con)
  cat(sprintf("%s client %d: %d damaged rows dropped, %d drawn, %d rows\n", net, i, dropped, added, length(keep)))
}

# Engineering compatibility for the official TIGRESS code when exactly two TFs
# yield a single LARS step. The official stability-selection score is unchanged;
# only its dropped matrix dimension is restored before tigress() indexes it.
library(jsonlite)
job <- fromJSON('job.json')
stopifnot(identical(job$method, 'TIGRESS'))
.libPaths(c(job$library, .libPaths()))
set.seed(job$seed)
x <- as.matrix(read.csv(job$expression, row.names=1, check.names=FALSE))
tfs <- read.csv(job$tfs, colClasses='character')$tf
targets <- setdiff(colnames(x), tfs)
stopifnot(all(is.finite(x)), length(tfs) == 2, length(targets) > 0)
original_genes <- colnames(x)
variable <- apply(x, 2, sd) > 0
x <- x[, variable, drop=FALSE]
tf_used <- intersect(tfs, colnames(x))
target_used <- intersect(targets, colnames(x))
stopifnot(length(tf_used) == 2, length(target_used) > 0)
library(lars)
source(file.path(job$software, 'stabilityselection.R'))
source(file.path(job$software, 'tigress.R'))
official_stabilityselection <- stabilityselection
stabilityselection <- function(x, y, nsplit=100, nstepsLARS=20,
                               alpha=0.2, scoring='area') {
  score <- official_stabilityselection(x, y, nsplit=nsplit,
                                      nstepsLARS=nstepsLARS,
                                      alpha=alpha, scoring=scoring)
  if (is.null(dim(score))) {
    stopifnot(nstepsLARS == 1, length(score) == ncol(x))
    score <- matrix(score, nrow=1)
  }
  score
}
params <- job$params
score <- tigress(x, tflist=tf_used, targetlist=target_used,
                 alpha=params$alpha, nstepsLARS=params$nstepsLARS,
                 nsplit=params$nsplit, scoring='area',
                 allsteps=FALSE, usemulticore=FALSE)
stopifnot(all(is.finite(score)))
full <- matrix(0, length(tfs), length(targets), dimnames=list(tfs, targets))
full[tf_used, target_used] <- score
out <- expand.grid(tf=tfs, target=targets, stringsAsFactors=FALSE)
out$score <- as.vector(full)
write.csv(out, 'official_grn.csv', row.names=FALSE)
write_json(list(method=job$method, constant_genes=original_genes[!variable],
                projection='TF-to-nonTF candidate projection',
                effective_nstepsLARS=1,
                compatibility='restore dropped 1-row score matrix only',
                R=R.version.string), 'official_audit.json',
           auto_unbox=TRUE, pretty=TRUE)

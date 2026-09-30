# Read the authors' released objects without re-running any simulator.
root <- "C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/raw/publication_20260920"
source <- "C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/data/raw/formal_benchmark/manual_downloads"
dir.create(root, recursive=TRUE, showWarnings=FALSE)
describe <- function(x, depth=0) {
  result <- list(class=class(x), length=length(x), dimensions=dim(x))
  if (!is.null(dim(x))) {
    result$rownames_head <- head(rownames(x), 8)
    result$colnames_head <- head(colnames(x), 8)
  }
  if (is.list(x) && depth < 4) {
    n <- min(length(x), 30)
    result$names <- names(x)
    result$children <- lapply(seq_len(n), function(i) describe(x[[i]], depth+1))
    names(result$children) <- head(names(x), n)
  } else if (is.atomic(x)) result$head <- head(as.vector(x), 12)
  result
}
for (name in c("spatial_5000cells_500genes", "spatial_3000cells_2000genes")) {
  file <- file.path(root, paste0("sim_", name, ".rds"))
  if (!file.exists(file)) unzip(file.path(source, paste0(name, ".zip")),
    files=paste0("sim_", name, ".rds"), exdir=root)
  x <- readRDS(file)
  jsonlite::write_json(describe(x), file.path(root, paste0(name, "_schema.json")),
    pretty=TRUE, auto_unbox=TRUE, null="null")
  print(name); print(names(x)); print(object.size(x), units="MB")
  rm(x); gc()
}

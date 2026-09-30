# Conversion only. NO sim_true_counts/add_expr_noise call is permitted here.
library(jsonlite)
root <- getwd()  # Python sets this to StudyConfig.raw_root; no hidden environment request path.
jobs <- fromJSON(file.path(root, "export_plan.json"), simplifyVector=FALSE)
for (job in jobs) {
  out <- job$export_dir
  if (file.exists(file.path(out, "export_complete.json"))) next
  x <- readRDS(job$rds)
  dir.create(out, recursive=TRUE, showWarnings=FALSE)
  writeLines(capture.output(str(x, max.level=2)), file.path(out,"object_schema.txt"))
  counts <- x$counts_obs
  if (is.list(counts)) counts <- counts$counts
  scale <- "authors_published_observed_counts"
  if (is.null(counts) && !is.null(job$observed_csv)) {
    counts <- as.matrix(read.csv(job$observed_csv, row.names=1, check.names=FALSE))
    if (identical(dim(counts),rev(dim(x$counts)))) counts <- t(counts)
    stopifnot(identical(dim(counts),dim(x$counts)))
  }
  if (is.null(counts)) { counts <- x$counts; scale <- "authors_published_latent_true_counts" }
  stopifnot(is.matrix(counts), identical(dim(counts), dim(x$counts)))
  genes <- rownames(x$counts); cells <- colnames(x$counts)
  stopifnot(length(genes)==nrow(counts), length(cells)==ncol(counts))
  dimnames(counts) <- list(genes,cells)
  write.csv(t(counts), file.path(out,"expression.csv"))
  spatial <- !is.null(x$cci_locs)
  if (spatial) {
    stopifnot(identical(rownames(x$cci_locs), cells))
    coords <- data.frame(cell_id=cells,x=x$cci_locs[,1],y=x$cci_locs[,2])
  } else coords <- data.frame(cell_id=cells,x=0,y=0)
  write.csv(coords,file.path(out,"coordinates.csv"),row.names=FALSE)
  stopifnot(identical(rownames(x$cell_meta), cells))
  types <- if ('cell.type.idx' %in% names(x$cell_meta)) paste0('ct',x$cell_meta$cell.type.idx) else rep('unknown',length(cells))
  write.csv(data.frame(cell_id=cells,cell_type=types,region='global',sample_id=job$dataset_id,condition='published_simulation'),
    file.path(out,'metadata.csv'),row.names=FALSE)
  tf <- genes[x$.grn$regulators]
  write.csv(data.frame(tf=tf),file.path(out,'tf_list.csv'),row.names=FALSE)
  geff <- x$.grn$geff
  stopifnot(nrow(geff)==length(genes),ncol(geff)==length(tf))
  idx <- which(geff!=0,arr.ind=TRUE)
  write.csv(data.frame(tf=tf[idx[,2]],target=genes[idx[,1]],effect=geff[idx]),
    file.path(out,'grn_positive.csv'),row.names=FALSE)
  # Candidate LR identities come from author-declared simulation options, NOT active ct-pair labels.
  lr <- x$.options$cci$params
  if (!is.null(lr)) {
    stopifnot(ncol(lr)==3)
    names(lr) <- c('receptor','ligand','effect')
    # Old releases use gene101 internally but count rows may be numeric.
    convert <- function(ids) {
      ids <- as.character(ids)
      for (i in seq_along(ids)) {
        if (!(ids[i] %in% genes) && sub('^gene','',ids[i]) %in% genes) ids[i] <- sub('^gene','',ids[i])
        if (!(ids[i] %in% genes) && paste0('gene',ids[i]) %in% genes) ids[i] <- paste0('gene',ids[i])
      }; ids
    }
    lr$ligand <- convert(lr$ligand); lr$receptor <- convert(lr$receptor)
    stopifnot(all(lr$ligand %in% genes),all(lr$receptor %in% genes))
    write.csv(lr[,c('ligand','receptor')],file.path(out,'candidate_lr.csv'),row.names=FALSE)
  } else write.csv(data.frame(ligand=character(),receptor=character()),file.path(out,'candidate_lr.csv'),row.names=FALSE)
  ct <- x$cci_cell_type_param
  if (!is.null(ct)) write.csv(ct,file.path(out,'cci_author_table.csv'),row.names=FALSE)
  write_json(list(dataset_id=job$dataset_id, is_spatial=spatial, counts_source=scale,
    cells=length(cells),genes=length(genes),candidate_tfs=tf,
    rds_file=job$rds, simulator_executed=FALSE, source='https://zenodo.org/records/13119261'),
    file.path(out,'export_complete.json'),pretty=TRUE,auto_unbox=TRUE)
  cat('[converted]',job$dataset_id,scale,'\n')
  rm(x,counts);gc()
}

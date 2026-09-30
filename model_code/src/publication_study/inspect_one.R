root <- 'C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/raw/publication_20260920/benchmark/published_tree1_500_cells110_genes_sigma0.1_1'
x <- readRDS(file.path(root,'author_result.rds'))
print(names(x));print(dim(x$counts));print(dim(x$counts_obs));print(head(rownames(x$counts)));print(head(colnames(x$counts)))
z <- read.csv(file.path(root,'author_observed_counts.csv'),row.names=1,check.names=FALSE)
print(dim(z));print(head(rownames(z)));print(head(colnames(z)));print(head(x$cell_meta));print(names(x$.options$cci));print(x$.options$cci$params)

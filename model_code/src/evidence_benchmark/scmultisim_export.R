# Internal adapter called only by preprocess_jumpniche.py. User parameters live
# in configs/public_sim_data/scmultisim.py, not command-line flags.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
cfg <- jsonlite::fromJSON(args[[1]])
stopifnot(as.character(packageVersion("scMultiSim")) == cfg$expected_package_version)
suppressPackageStartupMessages(library(scMultiSim))
dir.create(cfg$destination, recursive = TRUE, showWarnings = FALSE)
write_table <- function(x, name) write.csv(x, file.path(cfg$destination, name), row.names = FALSE)
gene_id <- function(x) as.character(x)
data_env <- new.env()
data(list = cfg$grn_resource, package = "scMultiSim", envir = data_env)
grn <- get(cfg$grn_resource, envir = data_env)
stopifnot(is.data.frame(grn), ncol(grn) == 3)
names(grn) <- c("target", "regulator", "effect")
grn$target <- gene_id(grn$target)
grn$regulator <- gene_id(grn$regulator)
# Declared candidate catalog BEFORE the simulator selects active type-level edges.
lr <- data.frame(target = sprintf("SYN_R%02d", seq_len(cfg$lr_count)),
                 regulator = sprintf("SYN_L%02d", seq_len(cfg$lr_count)),
                 effect = rep(cfg$cci_effect, cfg$lr_count))
stopifnot(cfg$genes >= length(unique(c(grn$target, grn$regulator))) + 2 * cfg$lr_count)
write_table(data.frame(tf = sort(unique(grn$regulator))), "candidate_tfs.csv")
write_table(data.frame(ligand = lr$regulator, receptor = lr$target), "candidate_lr.csv")
set.seed(cfg$seed)
result <- sim_true_counts(list(
    rand.seed = as.integer(cfg$seed), GRN = grn, tree = Phyla3(),
    num.cells = as.integer(cfg$cells), num.genes = as.integer(cfg$genes),
    num.cifs = as.integer(cfg$num_cifs), discrete.cif = TRUE,
    discrete.min.pop.size = as.integer(cfg$minimum_cells_per_type),
    intrinsic.noise = cfg$intrinsic_noise, speed.up = TRUE,
    cci = list(params = lr, max.neighbors = 4, radius = 1,
               start.layer = as.integer(cfg$cells), grid.size = as.integer(cfg$grid_size),
               layout = cfg$layout, cell.type.interaction = "random",
               cell.type.lr.pairs = seq.int(cfg$active_lr_min, cfg$active_lr_max),
               del.lr.pair = FALSE)))
genes <- rownames(result$counts)
cells <- colnames(result$counts)
stopifnot(length(genes) == cfg$genes, length(cells) == cfg$cells)
set.seed(cfg$seed + 100000L)
add_expr_noise(result, randseed = cfg$seed + 100000L, protocol = "UMI",
               alpha_mean = cfg$capture_mean, alpha_sd = cfg$capture_sd,
               depth_mean = cfg$depth_mean, depth_sd = cfg$depth_sd)
observed <- result$counts_obs
if (is.list(observed)) observed <- observed$counts
stopifnot(identical(dim(observed), dim(result$counts)), all(is.finite(observed)), all(observed >= 0))
dimnames(observed) <- list(genes, cells)
write.csv(t(observed), file.path(cfg$destination, "expression.csv"))
locs <- result$cci_locs
stopifnot(identical(rownames(locs), cells))
write_table(data.frame(cell_id = cells, x = locs[,1], y = locs[,2]), "coordinates.csv")
meta <- result$cell_meta
stopifnot(nrow(meta) == length(cells), "cell.type.idx" %in% names(meta))
write_table(data.frame(cell_id = cells, cell_type = paste0("ct", meta$cell.type.idx),
                       region = "global", sample_id = cfg$dataset_id,
                       condition = "simulated"), "metadata.csv")
# Official internal GRN is gene-by-TF. Rows follow the count gene order;
# regulators are integer positions, NOT the output gene names.
geff <- result$.grn$geff
regulators <- result$.grn$regulators
stopifnot(nrow(geff) == length(genes), ncol(geff) == length(regulators))
idx <- which(geff != 0, arr.ind = TRUE)
write_table(data.frame(tf = genes[regulators[idx[,2]]], target = genes[idx[,1]],
                       effect = geff[idx]), "grn_positive_edges.csv")
ct <- result$cci_cell_type_param
stopifnot(is.data.frame(ct), all(c("ligand", "receptor", "ct1", "ct2", "effect") %in% names(ct)))
# Verified against scMultiSim 1.2.0 .simSpatial: current cell icell receives
# neighbor nb's ligand; ctype_factor = params[type(icell), type(nb), lr].
# Thus ct1 = receiver, ct2 = sender (not the reverse).
write_table(data.frame(sender_cell_type = paste0("ct", ct$ct2),
                       receiver_cell_type = paste0("ct", ct$ct1),
                       ligand = ct$ligand, receptor = ct$receptor, effect = ct$effect),
            "cci_positive_edges.csv")
saveRDS(result, file.path(cfg$destination, "official_result.rds"))
writeLines(capture.output(sessionInfo()), file.path(cfg$destination, "R_sessionInfo.txt"))
jsonlite::write_json(list(package = "scMultiSim", version = as.character(packageVersion("scMultiSim")),
    seed = cfg$seed, truth_unit = "cell_type_pair_by_candidate_LR",
    direction = "ct2_sender_ct1_receiver", genes = length(genes), cells = length(cells),
    grn_resource = cfg$grn_resource, network_topology = "fixed_public_GRN_across_seeds",
    synthetic_LR = TRUE), file.path(cfg$destination, "export_complete.json"), auto_unbox = TRUE, pretty = TRUE)

# Fixed single-gene subset of the downloaded CellChatDB.mouse object.
# Called by the registered Python preprocessing profile; never used as truth.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
loaded <- load(args[[1]])
stopifnot('CellChatDB.mouse' %in% loaded)
interactions <- CellChatDB.mouse$interaction
genes <- unique(as.character(CellChatDB.mouse$geneInfo$Symbol))
ligand <- as.character(interactions$ligand)
receptor <- as.character(interactions$receptor)
simple <- !grepl('[_|&;+]',ligand) & !grepl('[_|&;+]',receptor) &
          ligand %in% genes & receptor %in% genes & ligand!=receptor
out <- unique(data.frame(ligand=ligand[simple],receptor=receptor[simple],
                         stringsAsFactors=FALSE))
out <- out[order(out$ligand,out$receptor),,drop=FALSE]
write.csv(out,args[[2]],row.names=FALSE)

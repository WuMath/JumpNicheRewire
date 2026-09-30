# 从 reconstruct 目录运行 Rscript baselines/install_cellchat.R；只在用户手动运行时安装。
InstallConfig <- methods::setRefClass("InstallConfig", fields = list(
  cran = "character", cellchat_ref = "character"
), methods = list(initialize = function() {
  cran <<- "https://cloud.r-project.org"
  cellchat_ref <<- "jinworks/CellChat@a0d3b2d231d46c8787177fffeac908270c253747"
}))
cfg <- InstallConfig$new()
options(repos = c(CRAN = cfg$cran))
for (pkg in c("remotes","BiocManager","R6","data.table","jsonlite","digest","openxlsx","future","callr","NMF","circlize")) {
  if (!requireNamespace(pkg, quietly=TRUE)) install.packages(pkg)
}
BiocManager::install(c("ComplexHeatmap","BiocNeighbors","SingleCellExperiment"),ask=FALSE,update=FALSE)
remotes::install_github(cfg$cellchat_ref,upgrade="never")
print(packageVersion("CellChat"))
sessionInfo()

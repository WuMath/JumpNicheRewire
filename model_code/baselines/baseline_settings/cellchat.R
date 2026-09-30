# R 方法使用 R6 配置类；不读取 Python 主模型的配置。
CellChatConfig <- R6::R6Class("CellChatConfig", public = list(
  mode = "check", # check -> freeze -> run
  suite = "sccigen", # sccigen 或 visium；不使用合成 TF1/G0 名称
  registry = "C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/data/processed/evidence_benchmark/03_registries/evidence_registry.csv",
  dataset_ids = c("sccigen_human_breast_example1"),
  output_root = "C:/Users/Wuyan/Desktop/GRN/Futher/GRN2/reconstruct/data/result/independent_comparison_v1",
  species = "human", # Visium 改为 mouse，且改 dataset_ids/suite
  expression_scale = "raw_counts",
  target_sum = 10000,
  cell_type_column = "cell_type",
  seed = 20260906L,
  nboot = 100L,
  min_cells = 10L,
  average_type = "triMean",
  population_size = FALSE,
  workers = 1L,
  expected_version = "2.1.2",
  expected_commit = "a0d3b2d231d46c8787177fffeac908270c253747",
  preview_rows = 10000L,
  overwrite = FALSE
))
CELLCHAT_CONFIG <- CellChatConfig$new()

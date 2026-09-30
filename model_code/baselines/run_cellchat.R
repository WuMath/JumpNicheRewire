# 每份数据一个独立 R 子进程。仅交换输入/输出文件，不调用 JumpNiche。
# 请先 cd 到 reconstruct，再执行 Rscript baselines/run_cellchat.R。
if (!file.exists("baselines/baseline_settings/cellchat.R")) stop("请先进入 reconstruct 目录")
for (pkg in c("R6","data.table","jsonlite","digest","openxlsx","future","callr","CellChat")) {
  if (!requireNamespace(pkg, quietly=TRUE)) stop(paste("缺少",pkg,"；见 install_cellchat.R"))
}
source("baselines/baseline_settings/cellchat.R")
cfg <- as.list(CELLCHAT_CONFIG)
cfg <- cfg[!vapply(cfg,is.function,logical(1))]
cfg <- cfg[!names(cfg) %in% c(".__enclos_env__")]
if (as.character(packageVersion("CellChat")) != cfg$expected_version) stop("CellChat 版本不符合冻结配置")
if (!identical(packageDescription("CellChat")$RemoteSha,cfg$expected_commit)) stop("CellChat 提交号不匹配；请使用 install_cellchat.R 安装固定提交")
if (cfg$workers != 1L) stop("当前 R 入口显式串行，每数据独立进程；workers 保持 1")
registry <- data.table::fread(cfg$registry, colClasses="character",data.table=FALSE)
if (!length(cfg$dataset_ids) || anyDuplicated(registry$dataset_id)) stop("数据清单为空或存在重复编号")
rows <- registry[match(cfg$dataset_ids,registry$dataset_id),,drop=FALSE]
if (anyNA(rows$dataset_id) || any(grepl("merfish",rows$dataset_id,ignore.case=TRUE))) stop("无效数据选择")
if (any(rows$species != cfg$species)) stop("配置物种不匹配，不能简单将鼠基因改成大写当人类基因")
if (!cfg$suite %in% c("sccigen","visium")) stop("此原版 CellChat 入口仅用于真实基因符号的 sCCIgen/Visium")

lock_cfg <- cfg[!names(cfg) %in% c("mode","overwrite","dataset_ids","registry","output_root","preview_rows")]
lock_cfg$package_versions <- as.list(vapply(c("CellChat","R6","Matrix","future"),
  function(x) as.character(packageVersion(x)), character(1)))
lock_cfg$adapter_sha256 <- digest::digest(file="baselines/run_cellchat.R",algo="sha256")
signature <- digest::digest(lock_cfg,algo="sha256")
lock_dir <- file.path(cfg$output_root,cfg$suite,"00_protocols")
lock_path <- file.path(lock_dir,"CellChat.json")
if (cfg$mode == "check") {
  print(rows[,c("dataset_id","dataset_dir","species")])
  for (i in seq_len(nrow(rows))) {
    for (name in c("expression.csv","metadata.csv")) {
      if (!file.exists(file.path(rows$dataset_dir[i],name))) stop(paste("缺少",name))
    }
  }
  cat("检查通过：未读取真值，未运行模型。\n")
} else if (cfg$mode == "freeze") {
  if (file.exists(lock_path) &&
      jsonlite::fromJSON(lock_path)$signature != signature) stop("已有不同协议；请选择新的输出目录")
  dir.create(lock_dir,recursive=TRUE,showWarnings=FALSE)
  jsonlite::write_json(list(signature=signature,parameters=lock_cfg),lock_path,auto_unbox=TRUE,pretty=TRUE)
  cat("已冻结：",lock_path,"\n")
} else if (cfg$mode == "run") {
  if (!file.exists(lock_path) || jsonlite::fromJSON(lock_path)$signature != signature) stop("先冻结当前参数")
  run_dataset <- function(cfg,row,signature) {
    library(CellChat)
    # 子进程只知道这份数据与配置，不读取 benchmark truth。
    output <- file.path(cfg$output_root,cfg$suite,"02_methods","CellChat",row$dataset_id)
    dir.create(output,recursive=TRUE,showWarnings=FALSE)
    status_file <- file.path(output,"run_status.json")
    state <- list(method="CellChat",dataset_id=row$dataset_id,state="running",
                  truth_read_during_inference=FALSE,worker_pid=Sys.getpid(),protocol_signature=signature)
    write_status <- function(x) {
      temp <- paste0(status_file,".writing")
      jsonlite::write_json(x,temp,auto_unbox=TRUE,pretty=TRUE,null="null")
      if (file.exists(status_file)) unlink(status_file)
      if (!file.rename(temp,status_file)) stop("状态文件写入失败")
    }
    expression_file <- file.path(row$dataset_dir,"expression.csv")
    metadata_file <- file.path(row$dataset_dir,"metadata.csv")
    input_signature <- digest::digest(c(digest::digest(file=expression_file,algo="sha256"),
                                       digest::digest(file=metadata_file,algo="sha256")),algo="sha256")
    if (file.exists(status_file) && !cfg$overwrite) {
      old <- jsonlite::fromJSON(status_file)
      if (old$state=="completed") {
        if (old$protocol_signature==signature && old$input_signature==input_signature &&
            file.exists(file.path(output,"cci_edges.csv")) &&
            old$output_sha256==digest::digest(file=file.path(output,"cci_edges.csv"),algo="sha256")) return(old)
        stop("旧完成结果与当前配置不一致；请更换 output_root")
      }
    }
    state$input_signature <- input_signature
    write_status(state)
    started <- proc.time()[["elapsed"]]
    log_file <- file(file.path(output,"console.log"),open="wt")
    sink(log_file,split=TRUE)
    on.exit({sink();close(log_file)},add=TRUE)
    tryCatch({
      expr <- data.table::fread(expression_file,data.table=FALSE,check.names=FALSE)
      ids <- as.character(expr[[1]])
      genes <- colnames(expr)[-1]
      values <- as.matrix(expr[,-1,drop=FALSE]);storage.mode(values)<-"double"
      if (anyDuplicated(ids) || anyDuplicated(genes) || any(!is.finite(values)) || any(values<0)) stop("表达输入无效")
      meta <- data.table::fread(metadata_file,data.table=FALSE,colClasses="character")
      meta <- meta[match(ids,meta$cell_id),,drop=FALSE]
      if (anyNA(meta$cell_id) || !cfg$cell_type_column %in% colnames(meta)) stop("metadata 不匹配")
      rownames(meta)<-ids
      if (cfg$expression_scale=="raw_counts") {
        values <- log1p(values / pmax(rowSums(values),1) * cfg$target_sum)
      } else if (cfg$expression_scale!="log1p_normalized") stop("表达尺度配置错误")
      rownames(values)<-ids;colnames(values)<-genes
      set.seed(cfg$seed)
      future::plan("sequential")
      object <- createCellChat(object=t(values),meta=meta,group.by=cfg$cell_type_column)
      object@DB <- if (cfg$species=="mouse") CellChatDB.mouse else CellChatDB.human
      object <- subsetData(object)
      object <- identifyOverExpressedGenes(object)
      object <- identifyOverExpressedInteractions(object)
      object <- computeCommunProb(object,type=cfg$average_type,raw.use=TRUE,
                                  population.size=cfg$population_size,nboot=cfg$nboot,seed.use=cfg$seed)
      object <- filterCommunication(object,min.cells=cfg$min_cells)
      saveRDS(object,file.path(output,"official_cellchat.rds"))
      result <- as.data.frame.table(object@net$prob,stringsAsFactors=FALSE)
      colnames(result)<-c("sender_cell_type","receiver_cell_type","interaction_name","score")
      interactions <- object@LR$LRsig
      match_rows <- match(result$interaction_name,rownames(interactions))
      if (anyNA(match_rows)) stop("CellChat LR 标识未匹配，拒绝猜测")
      result$ligand <- interactions$ligand[match_rows]
      result$receptor <- interactions$receptor[match_rows]
      result$lr_id <- paste(result$ligand,result$receptor,sep="__")
      data.table::fwrite(result,file.path(output,"cci_edges.csv"))
      openxlsx::write.xlsx(list(run_info=data.frame(key="method",value="CellChat"),
        manifest=data.frame(table="cci_edges",rows=nrow(result),full_file="cci_edges.csv"),
        cci_edges=head(result,cfg$preview_rows)),file.path(output,"CellChat_results.xlsx"))
      state$state<-"completed";state$tasks<-list("cci_edges");state$cci_unit<-"cell_type"
      state$resource_mode<-"native_CellChatDB";state$convergence_passed<-NULL
      state$score_definition<-"official computeCommunProb probability, no p-value rank filtering"
      state$output_sha256<-digest::digest(file=file.path(output,"cci_edges.csv"),algo="sha256")
    },error=function(e) {state$state<<-"failed";state$error_message<<-conditionMessage(e)})
    state$runtime_seconds<-proc.time()[["elapsed"]]-started
    state$peak_rss_bytes<-NULL # R 入口未实现可靠峰值监测，不以0冒充
    write_status(state)
    capture.output(sessionInfo(),file=file.path(output,"sessionInfo.txt"))
    state
  }
  statuses<-list()
  for (i in seq_len(nrow(rows))) {
    # callr 每次启动新的 R：某份数据原生库崩溃也不终止后面的数据。
    statuses[[i]]<-tryCatch(callr::r(run_dataset,args=list(cfg=cfg,row=as.list(rows[i,,drop=FALSE]),
                                                        signature=signature),show=TRUE),
      error=function(e) {
        state<-list(method="CellChat",dataset_id=rows$dataset_id[i],state="failed",error_message=conditionMessage(e))
        output<-file.path(cfg$output_root,cfg$suite,"02_methods","CellChat",rows$dataset_id[i])
        dir.create(output,recursive=TRUE,showWarnings=FALSE)
        old_status<-file.path(output,"run_status.json")
        preserve<-file.exists(old_status) && identical(jsonlite::fromJSON(old_status)$state,"completed")
        if (!preserve) jsonlite::write_json(state,old_status,auto_unbox=TRUE,pretty=TRUE)
        state
      })
  }
  jsonlite::write_json(statuses,file.path(cfg$output_root,cfg$suite,"CellChat_run_inventory.json"),auto_unbox=TRUE,pretty=TRUE)
  if (any(vapply(statuses,function(x) x$state!="completed",logical(1)))) stop("部分数据失败；检查各数据目录中的状态和日志")
} else stop("mode 必须是 check/freeze/run")

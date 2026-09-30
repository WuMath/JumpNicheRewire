"""调用官方算法；不导入 JumpNicheRewire，不读取 truth。"""
from __future__ import annotations
import importlib
import os
from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
from .io import ann_data, simple_lr, require_types, json_write

def _budget(estimate, limit, label):
    if estimate > limit * 1024**3:
        raise MemoryError(f"{label} 估算至少 {estimate/1024**3:.2f} GiB，超过配置 {limit} GiB；"
                          "请使用超算或事先为所有方法统一确定较小面板。不要静默截断。")

def _grn_chunks(matrices, data, unsigned=True):
    genes = list(data["log"].columns)
    cells = list(data["log"].index)
    positions = [genes.index(tf) for tf in data["tfs"]]
    if not positions:
        raise ValueError("输入中没有可用 TF")
    for cell_index, matrix in enumerate(matrices):
        block = np.asarray(matrix)[positions,:]
        frame = pd.DataFrame(dict(cell_id=cells[cell_index],
                                  tf=np.repeat(data["tfs"],len(genes)),
                                  target=np.tile(genes,len(positions)),
                                  score=np.abs(block).ravel().astype(float),
                                  signed_score=np.full(block.size,np.nan)))
        yield frame[frame.tf.ne(frame.target)]

def tree(name,cfg,dc,row,data,out,writer):
    from arboreto.algo import genie3, grnboost2
    from distributed import Client, LocalCluster
    if not data["tfs"]:
        raise ValueError("No expressed TF candidates")
    with LocalCluster(n_workers=cfg.n_jobs,threads_per_worker=1,processes=False,
                      dashboard_address=None) as cluster, Client(cluster) as client:
        algorithm = genie3 if name == "genie3" else grnboost2
        result = algorithm(expression_data=data["log"],tf_names=data["tfs"],
                           client_or_address=client,seed=cfg.seed,verbose=True)
    result.to_csv(out / "official_grn.csv",index=False)
    result = result.rename(columns={"TF":"tf","importance":"score"})
    result["score"] = result.score.abs()
    result["signed_score"] = np.nan
    writer.write("grn_edges",[result[["tf","target","score","signed_score"]]])
    return dict(grn_unit="global",supports_signed_effect=False,
                score_definition="official arboreto feature importance")

def cespgrn(cfg,dc,row,data,out,writer):
    import torch
    from sklearn.decomposition import PCA
    sys.path.insert(0,str(cfg.repository / "src"))
    official = importlib.import_module("g_admm")
    kernel = importlib.import_module("kernel")
    values = data["log"].to_numpy()
    n,g = values.shape
    _budget(n*g*g*8*8 + n*n*8*8,cfg.memory_limit_gib,"CeSpGRN")
    if cfg.kernel_input == "expression_pca":
        latent = PCA(n_components=min(cfg.pca_components,n-1,g),random_state=cfg.seed).fit_transform(values)
    elif cfg.kernel_input == "spatial":
        if dc.suite == "sergio":
            raise ValueError("SERGIO 坐标只是适配布局；此 benchmark 使用 expression_pca")
        latent = data["coordinates"].to_numpy()
    else:
        raise ValueError("kernel_input 必须为 expression_pca/spatial")
    K,K_trun = kernel.calc_kernel_neigh(latent,k=min(cfg.kernel_k,n-1),
                       bandwidth=cfg.bandwidth,truncate=True,truncate_param=min(cfg.neighbors,n-1))
    covariance = official.est_cov(X=values,K_trun=K_trun,weighted_kt=True)
    model = official.G_admm_minibatch(X=values[:,None,:],K=K,pre_cov=covariance,
                                    batchsize=cfg.batch_size,seed=cfg.seed,device=torch.device(cfg.device))
    graphs = model.train(max_iters=cfg.max_iters,n_intervals=100,lamb=cfg.lamb,njobs=1)
    if np.shape(graphs) != (n,g,g) or not np.isfinite(graphs).all():
        raise ValueError("CeSpGRN 返回网络形状或数值无效")
    # 在转换表格前保存原数值结果，写表失败不丢失官方计算结果。
    np.save(out/"official_graphs.npy",graphs)
    writer.write("grn_edges",_grn_chunks(graphs,data))
    return dict(grn_unit="cell",supports_signed_effect=False,
                score_definition="absolute official partial-correlation graph; TF projection, not causal direction")

def spagrn(cfg,dc,row,data,out,writer):
    if row["species"] != cfg.reference_species or row["species"] == "synthetic":
        raise ValueError("SpaGRN 完整流程需要匹配物种的真实基因符号和 cisTarget；模拟 G0/TF1 不适用")
    if dc.expression_scale != "raw_counts" and cfg.hotspot_model == "danb":
        raise ValueError("SpaGRN danb 模型需要原始 count；不要传入已经 log 的矩阵")
    sys.path.insert(0,str(cfg.repository / "src"))
    from spagrn.regulatory_network import InferRegulatoryNetwork
    require_types(data,dc)
    obj = ann_data(data,dc,raw=True)
    obj.layers["raw_counts"] = obj.X.copy()
    tf_file=out/"expressed_tfs.txt"
    tf_file.write_text("\n".join(data["tfs"])+"\n",encoding="utf-8")
    model = InferRegulatoryNetwork(obj,project_name=row["dataset_id"])
    raw_dir=out/"official"
    raw_dir.mkdir(exist_ok=True)
    lr=simple_lr(data).rename(columns={"ligand":"from","receptor":"to"})
    # 1.1.4 源码 API；不用旧网页中已不适用的 method/prefix/c_threshold 参数。
    model.infer(databases=str(cfg.database),motif_anno_fn=str(cfg.motif_annotations),
                tfs_fn=str(tf_file),cluster_label=dc.cell_type_column,niche_df=lr,
                num_workers=cfg.n_jobs,output_dir=str(raw_dir),cache=False,save_tmp=True,
                layers="raw_counts",model=cfg.hotspot_model,latent_obsm_key="spatial",
                n_neighbors=cfg.neighbors)
    obj.write_h5ad(out/"official_spagrn.h5ad")
    frames=[]
    for regulon in model.regulons:
        weights=regulon.gene2weight
        frames.append(pd.DataFrame(dict(tf=regulon.transcription_factor,target=list(weights),
                                        score=np.abs(list(weights.values())),signed_score=np.nan)))
    if not frames:
        raise ValueError("SpaGRN 未产生 regulon；检查基因/TF/motif 覆盖，不伪造预测")
    writer.write("grn_edges",frames)
    return dict(grn_unit="global",supports_signed_effect=False,
                score_definition="motif-pruned regulon weights; native cisTarget prior",
                prior_overlap_requires_audit=True)

def liana(cfg,dc,row,data,out,writer):
    import liana as li
    require_types(data,dc)
    obj=ann_data(data,dc)
    kwargs={}
    if cfg.resource_mode == "matched":
        resource=simple_lr(data)[["ligand","receptor"]]
        if resource.empty:
            raise ValueError("没有可用的匹配 LR；SERGIO 的 NULL_LR 不能当通信参考")
        kwargs=dict(resource=resource,resource_name=None)
    elif cfg.resource_mode == "native":
        if row["species"] == "synthetic":
            raise ValueError("模拟基因没有 native LR 数据库映射")
        if row["species"] != cfg.resource_species:
            raise ValueError("LIANA 内置资源物种与数据不匹配；使用 matched 或匹配物种的资源")
        kwargs=dict(resource_name=cfg.resource_name)
    else:
        raise ValueError("resource_mode must be matched/native")
    li.mt.rank_aggregate(obj,groupby=dc.cell_type_column,use_raw=False,
                         n_perms=cfg.n_perms,min_cells=cfg.min_cells,
                         expr_prop=cfg.expr_prop,seed=cfg.seed,verbose=True,**kwargs)
    result=obj.uns["liana_res"].copy()
    result.to_csv(out/"official_liana.csv",index=False)
    result=result.rename(columns={"source":"sender_cell_type","target":"receiver_cell_type",
                                   "ligand_complex":"ligand","receptor_complex":"receptor"})
    result["score"]=1.0-pd.to_numeric(result["magnitude_rank"],errors="raise")
    result["lr_id"]=result.ligand+"__"+result.receptor
    writer.write("cci_edges",[result[["sender_cell_type","receiver_cell_type","ligand","receptor","lr_id","score"]]])
    return dict(cci_unit="cell_type",resource_mode=cfg.resource_mode,
                score_definition="1 - LIANA magnitude_rank; larger is stronger")

def cpdb_table(frame):
    """按 receptor_A/B 定方向；复合物保留原始结果，不能猜测拆分。"""
    identity=["gene_a","gene_b","receptor_a","receptor_b"]
    if not set(identity).issubset(frame):
        raise ValueError("CellPhoneDB 输出缺少方向注释列")
    truthy=lambda x: str(x).lower() in {"true","1","1.0"}
    result=[]
    for pair in [c for c in frame.columns if "|" in str(c)]:
        source,target=pair.split("|")
        f=frame.dropna(subset=["gene_a","gene_b"]).copy()
        a=f.receptor_a.map(truthy)
        b=f.receptor_b.map(truthy)
        # 严格简单 ligand->receptor 子集；A/B 都是 receptor 或都不是时不臆断。
        f=f[a ^ b].copy()
        a=f.receptor_a.map(truthy)
        result.append(pd.DataFrame(dict(
            sender_cell_type=np.where(a,target,source),
            receiver_cell_type=np.where(a,source,target),
            ligand=np.where(a,f.gene_b,f.gene_a),receptor=np.where(a,f.gene_a,f.gene_b),
            score=pd.to_numeric(f[pair],errors="raise").fillna(0).to_numpy())))
    if not result:
        raise ValueError("CellPhoneDB 没有可解释的单基因有方向 LR")
    result=pd.concat(result,ignore_index=True)
    result["lr_id"]=result.ligand+"__"+result.receptor
    return result

def cellphonedb(cfg,dc,row,data,out,writer):
    if row["species"] != "human":
        raise ValueError("CellPhoneDB 标准数据库需要人类基因。本适配器不把小鼠基因直接改大写，也不伪造同源映射。")
    require_types(data,dc)
    from cellphonedb.src.core.methods import cpdb_statistical_analysis_method
    obj=ann_data(data,dc)
    obj.write_h5ad(out/"counts.h5ad")
    metadata=pd.DataFrame(dict(Cell=obj.obs_names,cell_type=obj.obs[dc.cell_type_column].astype(str).values))
    if metadata.cell_type.str.contains(r"[|]").any():
        raise ValueError("细胞类型不能包含 |；先统一清理并保留映射")
    metadata.to_csv(out/"meta.tsv",sep="\t",index=False)
    result=cpdb_statistical_analysis_method.call(
        cpdb_file_path=str(cfg.database),meta_file_path=str(out/"meta.tsv"),
        counts_file_path=str(out/"counts.h5ad"),counts_data="hgnc_symbol",
        output_path=str(out/"official"),iterations=cfg.iterations,threads=cfg.threads,
        debug_seed=cfg.seed,threshold=cfg.threshold,result_precision=cfg.result_precision,
        score_interactions=True,output_suffix="fixed")
    result["interaction_scores"].to_csv(out/"official_scores.csv",index=False)
    writer.write("cci_edges",[cpdb_table(result["interaction_scores"])])
    return dict(cci_unit="cell_type",resource_mode="native_human",
                score_definition="official interaction_scores, simple directed pairs only",
                complex_policy="complex/ambiguous rows retained in official output, excluded from simple-LR export")

def imiracle(cfg,dc,row,data,out,writer):
    import torch
    require_types(data,dc)
    lr=simple_lr(data)
    if lr.empty:
        raise ValueError("iMIRACLE 需要有效 LR 候选，SERGIO NULL_LR 不适用")
    n,g=data["log"].shape
    _budget(n*n*len(lr)*4*4+n*g*len(lr)*4*3,cfg.memory_limit_gib,"iMIRACLE 多视图稠密矩阵")
    sys.path.insert(0,str(cfg.repository))
    from multiview_graph import MultiViewGraph
    from neural_networks import iMiracle
    from train_utils import iMiracleTrainer
    obj=ann_data(data,dc)
    interactions=lr[["ligand","receptor"]].copy()
    # ModuleDict 的 key 不能含 '.'，用内部编号并保存到实际 LR 的映射。
    interactions["interaction_name"]=[f"LR{i}" for i in range(len(lr))]
    mapping=dict(zip(interactions.interaction_name,lr.lr_id))
    json_write(out/"view_to_lr.json",mapping)
    initial=pd.get_dummies(obj.obs[dc.cell_type_column]).astype(float)
    graph=MultiViewGraph(interaction_db=interactions,complex_db=None,adata=obj,
                         expressed_prop=cfg.expressed_prop,precision="float",
                         n_neighbors=cfg.neighbors,quality_control=False)
    if not graph.multi_view_graph:
        raise ValueError("没有通过表达过滤的 LR 视图")
    model=iMiracle(mvG=graph,input_dim=initial.shape[1],hidden_dim=cfg.hidden_dim,
                   output_dim=g,device=cfg.device,gnn_type=cfg.gnn_type,
                   share_gnn=False,share_decoder=False,GRN=None,iterative=cfg.iterative)
    trainer=iMiracleTrainer(mode="ranking",model=model,adata=obj,initial_x=initial,
                            observed_x=obj.X,device=cfg.device,precision="float")
    trainer.load_parameter_for_ranking(epoch=cfg.epochs,learning_rate=cfg.learning_rate,
                                      patience=cfg.patience,threshold=cfg.threshold,
                                      omega=cfg.omega,spatial_loss_epoch=cfg.spatial_loss_epoch)
    trainer.train_model_for_ranking()
    torch.save(model.state_dict(),out/"official_model.pt")
    # 官方公开源码保存的是每个 LR 视图的表达贡献分量，不冒充 TF-target 或因果效应。
    def chunks():
        for name, values in model.saved_X_lr_GRN.items():
            np.save(out/f"official_component_{name}.npy",values)
            for start in range(0,n,16):
                block=values[start:start+16]
                yield pd.DataFrame(dict(receiver_id=np.repeat(obj.obs_names[start:start+len(block)],g),
                                         lr_id=mapping[name],target=np.tile(obj.var_names,len(block)),
                                         score=np.abs(block).ravel().astype(float)))
    writer.write("lr_gene",chunks())
    return dict(lr_gene_unit="receiver",score_definition="absolute saved_X_lr_GRN component",
                official_paper_regulatory_score_reproduced=False,
                interpretation="iMIRACLE-LRComponent adapter variant; not TF-target/CCI probability/causal effect")

def clarify(cfg,dc,row,data,out,writer):
    import torch
    if row["species"] != "mouse" or cfg.lr_database != 0:
        raise ValueError("核对的 CLARIFY 源码仅实现 Mouse=0 和其自带 scMultiSim=2；"
                         "文档声称的 Human=1 分支未实现。不要用它冒充当前 SERGIO/sCCIgen 适配。")
    require_types(data,dc)
    n=len(data["raw"])
    _budget((n*cfg.genes_per_cell)**2*4*6,cfg.memory_limit_gib,"CLARIFY gene-level dense decoder")
    src=cfg.repository/"src"
    old_cwd=Path.cwd()
    sys.path.insert(0,str(src))
    os.chdir(src)  # 官方子模块相对路径；只在本数据集子进程中生效。
    try:
        official=importlib.import_module("main")
        preprocessing=importlib.import_module("preprocessing")
        training=importlib.import_module("training")
        st=data["log"].copy().reset_index(drop=True)
        st.insert(0,"Cell_Type",data["metadata"][dc.cell_type_column].values)
        st.insert(0,"Y",data["coordinates"].iloc[:,1].values)
        st.insert(0,"X",data["coordinates"].iloc[:,0].values)
        st.insert(0,"Cell_ID",data["log"].index)
        chosen,lr_ids=preprocessing.select_LRgenes(st,cfg.genes_per_cell,cfg.lr_database)
        chosen.to_csv(out/"selected_st.csv",index=False)
        selected_genes=[x for x in chosen.columns if x not in {"Cell_ID","X","Y","Cell_Type"}]
        if len(selected_genes) != cfg.genes_per_cell:
            raise ValueError("CLARIFY selected panel does not match genes_per_cell")
        if not set(data["tfs"]).intersection(selected_genes):
            raise ValueError("CLARIFY 随机选中面板不含待评价 TF，停止；不能从 test 真值挑选面板")
        prep=out/"official_preprocess"
        prep.mkdir(exist_ok=True)
        cell_adj,gene_graph,_,_,grns,gene_features,_=official.preprocess(
            chosen,cfg.neighbors,lr_ids.values(),
            dict(bandwidth=cfg.initial_grn_bandwidth,n_neigh=cfg.initial_grn_neighbors,
                 lamb=cfg.initial_grn_lamb,max_iters=cfg.initial_grn_iterations))
        import networkx as nx
        arrays=dict(celllevel_adjacencylist=cell_adj,
                    celllevel_adjacencymatrix=preprocessing.convert_adjacencylist2adjacencymatrix(cell_adj),
                    celllevel_edgelist=preprocessing.convert_adjacencylist2edgelist(cell_adj),
                    celllevel_features=data["log"].to_numpy(),
                    genelevel_edgelist=nx.to_pandas_edgelist(gene_graph)[["source","target"]].to_numpy().T,
                    genelevel_features=gene_features,initial_grns=grns)
        for key,value in arrays.items():
            np.save(prep/f"{key}.npy",value)
        graph_data=training.create_pyg_data(str(prep),split=cfg.internal_edge_split)
        hyper=dict(num_genespercell=cfg.genes_per_cell,concat_hidden_dim=cfg.hidden_dim,
                   optimizer="adam",criterion=torch.nn.BCELoss(),num_epochs=cfg.epochs,
                   split=cfg.internal_edge_split)
        model=official.build_clarifyGAE_pytorch(graph_data,hyper)
        hyper["optimizer"]=(torch.optim.Adam(model.parameters(),lr=cfg.learning_rate),)
        model,history=training.train_gae(data=graph_data,model=model,hyperparameters=hyper)
        torch.save(model.state_dict(),out/"official_model.pt")
        history.to_csv(out/"internal_graph_reconstruction_history.csv",index=False)
        train_cells=graph_data[0][0]
        gene_data=graph_data[1]
        pos=train_cells.edge_label==1
        model.eval()
        with torch.no_grad():
            z,_,_,zg=model.encode(train_cells.x,gene_data.x,
                                   train_cells.edge_label_index[:,pos],gene_data.edge_index)
        np.save(out/"cell_embeddings.npy",z.cpu().numpy())
        np.save(out/"gene_embeddings.npy",zg.cpu().numpy())
        selected_data=dict(data)
        selected_data["log"]=data["log"][selected_genes]
        selected_data["tfs"]=[tf for tf in data["tfs"] if tf in selected_genes]
        def matrices():
            for index in range(n):
                block=zg[index*cfg.genes_per_cell:(index+1)*cfg.genes_per_cell]
                yield torch.sigmoid(block @ block.T).cpu().numpy()
        writer.write("grn_edges",_grn_chunks(matrices(),selected_data))
        def cci_chunks():
            for start in range(0,n,16):
                values=torch.sigmoid(z[start:start+16] @ z.T).cpu().numpy()
                frame=pd.DataFrame(dict(sender_id=np.repeat(data["log"].index[start:start+len(values)],n),
                                         receiver_id=np.tile(data["log"].index,len(values)),lr_id="__all__",
                                         score=values.ravel().astype(float)))
                yield frame[frame.sender_id.ne(frame.receiver_id)]
        writer.write("cci_edges",cci_chunks())
        return dict(grn_unit="cell",cci_unit="cell_total",supports_signed_effect=False,
                    score_definition="official GAE inner-product sigmoid",
                    internal_history_is_benchmark_truth=False,
                    selected_genes=selected_genes)
    finally:
        os.chdir(old_cwd)

def fit(name,cfg,dc,row,data,out,writer):
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if name in {"cespgrn","imiracle","clarify"}:
        import torch
        torch.manual_seed(cfg.seed)
    if name in {"genie3","grnboost2"}:
        return tree(name,cfg,dc,row,data,out,writer)
    return globals()[name](cfg,dc,row,data,out,writer)

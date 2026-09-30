"""All-spot real applications. Region labels are computational, not pathology diagnoses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from configs.real_data.settings.prepare_visium import VisiumConfig
from .resources import CASE_SOURCES,download,file_hash,write_json,storage_guard

def human_references(cfg):
    from preprocess.common import standardize_reference_tables
    from preprocess.prepare_curated_lr import curate_lr,_role_table
    from configs.real_data.settings.prepare_curated_lr import CuratedLRConfig
    out=cfg.processed_root/'references_human';out.mkdir(parents=True,exist_ok=True)
    raw=cfg.old_data_root/'raw/evidence_benchmark/01_raw_downloads/public_simulator'
    tfs,prior,_,audit=standardize_reference_tables(raw/'Human-TF-Target.tsv',raw/'Human-ligand-receptor.tsv')
    support=cfg.old_data_root/'raw/ModelData/omnipath_support'
    lr,detailed,curation,rejected=curate_lr(pd.read_csv(raw/'Human-ligand-receptor.tsv',sep='\t'),
        _role_table(support/'human_intercell_ligand.tsv','ligand'),
        _role_table(support/'human_intercell_receptor.tsv','receptor'),CuratedLRConfig())
    for name,frame in (('tf_list',tfs),('prior_tf_gene',prior),('lr_pairs',lr),
                       ('lr_pairs_detailed',detailed),('curation_audit',curation),('rejected_lr',rejected)):
        frame.to_csv(out/(name+'.csv'),index=False)
    # No human motif database is present; disclose the absent evidence, do not invent it.
    write_json(out/'source_audit.json',dict(tf_reference=str(raw/'Human-TF-Target.tsv'),
        lr_reference=str(raw/'Human-ligand-receptor.tsv'),motif_evidence='not_available',
        complex_policy='complexes excluded by strict simple-symbol curation',filter_audit=audit))
    return out

def prepare_cases(cfg):
    from preprocess.prepare_visium import main as convert_visium
    from preprocess.prepare_lr_tf import prepare_signaling_edges,build_lr_tf_prior
    from configs.real_data.settings.prepare_lr_tf import LRTFConfig
    from evidence_benchmark.common import finalize_dataset
    rows=[];human=human_references(cfg)
    for species in ('Mouse','Human'):
        name='WikiPathways_2024_'+species
        download(cfg,'https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName='+name,
            cfg.raw_root/'pathways'/(name+'.gmt'),'https://maayanlab.cloud/Enrichr/#libraries')
    human_signal=download(cfg,
        'https://omnipathdb.org/interactions?format=tsv&datasets=omnipath&organisms=9606&genesymbols=1&fields=sources,references,curation_effort',
        cfg.raw_root/'references/human_signaling.tsv','https://r.omnipathdb.org/reference/omnipath_query.html')
    for case in cfg.cases:
        storage_guard(cfg)
        name,slug,species=CASE_SOURCES[case]
        files=json.loads((cfg.raw_root/'cases'/case/'source_files.json').read_text(encoding='utf-8'))
        out=cfg.processed_root/('case_'+case)
        refs=human if species=='human' else cfg.old_data_root/'processed/references'
        params=VisiumConfig(matrix_h5=Path(files['filtered_feature_bc_matrix.h5']),
            spatial_source=Path(files['spatial.tar.gz']),reference_dir=refs,output_dir=out,
            maximum_spots=None,maximum_genes=cfg.case_maximum_genes,maximum_tfs=cfg.case_maximum_tfs,
            maximum_lr_pairs=cfg.case_maximum_lr_pairs,random_seed=cfg.seed,sample_id=name,
            condition_label='tumor' if species=='human' else 'healthy')
        convert_visium(params)
        expression=pd.read_csv(out/'expression.csv',index_col=0)
        normalized=np.log1p(expression.div(expression.sum(axis=1).replace(0,1),axis=0)*10000)
        pcs=PCA(n_components=10,random_state=cfg.seed).fit_transform(normalized)
        clusters=KMeans(n_clusters=6,n_init=10,random_state=cfg.seed).fit_predict(pcs)
        metadata=pd.read_csv(out/'metadata.csv')
        metadata['region']=['expression_cluster_'+str(i+1) for i in clusters]
        metadata['cell_type']='NA'
        metadata['annotation_source']='unsupervised_expression_cluster_not_celltype_or_histology'
        metadata.to_csv(out/'metadata.csv',index=False)
        if species=='human':
            pc=LRTFConfig(reference_dir=out,maximum_frontier_nodes=1000,maximum_path_length=4)
            signal=prepare_signaling_edges(pd.read_csv(human_signal,sep='\t'),pc)
            prior,paths,audit=build_lr_tf_prior(signal,pd.read_csv(out/'lr_pairs.csv'),pd.read_csv(out/'tf_list.csv'),pc)
            prior.to_csv(out/'prior_lr_tf.csv',index=False)
            paths.to_parquet(out/'reference_receptor_tf_paths.parquet',index=False)
            audit.to_csv(out/'lr_tf_audit.csv',index=False)
        path=finalize_dataset(out,dataset_id='case_'+case,evidence_track='application',
            source='https://www.10xgenomics.com/datasets/'+slug,evaluation_scope='pathway_application',
            truth_kind='no_complete_network_truth',species=species,tissue=case,truth_files=(),
            notes='All tissue spots; single section; expression clusters not biological labels; association only.')
        row=json.loads(path.read_text(encoding='utf-8'));row.pop('files',None)
        row.update(split='case',is_spatial=True,counts_source='real_10x_counts',stratum=case,
                   independence_group=case,raw_genes=None,selected_targets=cfg.case_maximum_targets,
                   communication_available=True)
        write_json(out/'annotation_audit.json',dict(method='PCA10_KMeans6',seed=cfg.seed,
            labels_are_cell_types=False,labels_are_validated_anatomical_regions=False,
            all_tissue_spots_retained=True,biological_samples=1,independent_activity=False,perturbation=False))
        rows.append(row)
        print('[case prepared]',case,len(expression),'spots',flush=True)
    return rows

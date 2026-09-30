"""Published archives and real-case data -> explicit, label-isolated inputs."""
from dataclasses import asdict
from pathlib import Path
import io
import json
import subprocess
import zipfile
import numpy as np
import pandas as pd
from .resources import CASE_SOURCES, PUBLIC_MD5, download, file_hash, storage_guard, write_json

def extract_member(archive, name, destination, cfg):
    info = archive.getinfo(name)
    storage_guard(cfg, info.file_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with archive.open(info) as source, destination.open('wb') as target:
            import shutil
            shutil.copyfileobj(source, target, 1024*1024)

def export_archives(cfg):
    records, jobs = [], []
    for name, md5 in PUBLIC_MD5.items():
        path = cfg.public_archive_root / name
        actual = file_hash(path, 'md5')
        if actual != md5:
            raise ValueError(f'Public checksum mismatch: {path}')
        records.append(dict(path=str(path),md5=actual,source='https://zenodo.org/records/13119261',bytes=path.stat().st_size))
    write_json(cfg.result_root/'01_data/public_archive_checksums.json', records)
    for tree in cfg.trees:
        with zipfile.ZipFile(cfg.public_archive_root/f'main_tree{tree}.zip') as outer:
            for sigma in cfg.noise_levels:
                for replicate in cfg.validation_replicates + cfg.test_replicates:
                    name = f'tree{tree}_500_cells110_genes_sigma{sigma}_{replicate}'
                    dataset_id = 'published_' + name
                    folder = cfg.raw_root / 'benchmark' / dataset_id
                    with zipfile.ZipFile(io.BytesIO(outer.read(name+'.zip'))) as inner:
                        extract_member(inner, name+'/res.rds', folder/'author_result.rds', cfg)
                        observed = name+'/integ/counts_obs.csv'
                        if observed not in inner.namelist():
                            raise ValueError(f'Missing published observed counts: {name}')
                        extract_member(inner, observed, folder/'author_observed_counts.csv', cfg)
                    jobs.append(dict(dataset_id=dataset_id,rds=str(folder/'author_result.rds'),
                        observed_csv=str(folder/'author_observed_counts.csv'),export_dir=str(folder/'export'),
                        split='validation' if replicate in cfg.validation_replicates else 'test',
                        stratum=f'tree{tree}_sigma{sigma}',independence_group=f'tree{tree}_sigma{sigma}'))
    for name in cfg.published_spatial:
        folder=cfg.raw_root/'benchmark'/name
        with zipfile.ZipFile(cfg.public_archive_root/(name+'.zip')) as archive:
            extract_member(archive,'sim_'+name+'.rds',folder/'author_result.rds',cfg)
        jobs.append(dict(dataset_id='published_'+name,rds=str(folder/'author_result.rds'),
                         export_dir=str(folder/'export'),split='test',stratum='large_spatial_latent_counts',
                         independence_group=name))
    write_json(cfg.raw_root/'export_plan.json',jobs)
    subprocess.run([str(cfg.rscript),'--vanilla',str(Path(__file__).with_name('export_public.R'))],cwd=cfg.raw_root,check=True)
    return jobs

def prepare_benchmark(cfg,jobs):
    from evidence_benchmark.common import write_empty_priors, finalize_dataset
    rows=[]
    for job in jobs:
        root=Path(job['export_dir']); out=cfg.processed_root/job['dataset_id']
        out.mkdir(parents=True,exist_ok=True)
        info=json.loads((root/'export_complete.json').read_text(encoding='utf-8'))
        expr=pd.read_csv(root/'expression.csv',index_col=0)
        expr.index=expr.index.astype(str); expr.columns=expr.columns.astype(str)
        tfs=pd.read_csv(root/'tf_list.csv',dtype=str).tf.tolist()
        lr=pd.read_csv(root/'candidate_lr.csv',dtype=str)
        targets=[g for g in expr if g not in tfs]
        if len(targets)>cfg.benchmark_maximum_targets:
            normalized=np.log1p(expr.div(expr.sum(axis=1).replace(0,1),axis=0)*10000)
            variance=normalized[targets].var()
            targets=sorted(targets,key=lambda g:(-variance[g],g))[:cfg.benchmark_maximum_targets]
        # LR identity is a declared candidate catalog. No active type-pair labels used here.
        genes=list(dict.fromkeys(tfs+targets+lr.ligand.tolist()+lr.receptor.tolist()))
        expr[genes].to_csv(out/'expression.csv',index_label='cell_id')
        for name in ('coordinates.csv','metadata.csv','tf_list.csv'):
            pd.read_csv(root/name,dtype={'tf':str}).to_csv(out/name,index=False)
        lr=lr.assign(lr_id=lr.ligand+'__'+lr.receptor,signal_type='secreted',rho=1.0,confidence=1.0)
        lr.to_csv(out/'lr_pairs.csv',index=False)
        write_empty_priors(out)
        pd.DataFrame({'target':targets}).to_csv(out/'evaluation_targets.csv',index=False)
        # Only this label-conversion step reads edge truth; no derived edge enters model files.
        truth=out/'truth';truth.mkdir(exist_ok=True)
        positives=pd.read_csv(root/'grn_positive.csv',dtype={'tf':str,'target':str})
        universe=pd.MultiIndex.from_product([tfs,targets],names=['tf','target']).to_frame(index=False)
        universe=universe.merge(positives,on=['tf','target'],how='left',validate='one_to_one')
        universe['label']=universe.effect.fillna(0).ne(0).astype(int)
        universe.to_csv(truth/'truth_grn.csv',index=False)
        path=finalize_dataset(out,dataset_id=job['dataset_id'],evidence_track='public_simulator',
            source='https://doi.org/10.5281/zenodo.13119261',evaluation_scope='published_GRN_global',
            truth_kind='exact_simulation',species='synthetic',tissue='published_simulation',
            truth_files=('truth_grn.csv',),notes='All cells retained; fixed expression-selected target panel; published data, not resimulated.')
        row=json.loads(path.read_text(encoding='utf-8')); row.update({k:job[k] for k in ('split','stratum','independence_group')})
        row.update(is_spatial=info['is_spatial'],counts_source=info['counts_source'],raw_genes=info['genes'],
                   selected_targets=len(targets),communication_available=not lr.empty)
        rows.append(row)
    return rows

def acquire_cases(cfg):
    records=[]
    for case in cfg.cases:
        name,slug,species=CASE_SOURCES[case]
        page='https://www.10xgenomics.com/datasets/'+slug
        files={}
        for suffix in ('filtered_feature_bc_matrix.h5','spatial.tar.gz'):
            url=f'https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/{name}/{name}_{suffix}'
            if case=='mouse_brain':
                path=cfg.old_data_root/'raw/real_data/visium_ffpe_mouse_brain'/f'{name}_{suffix}'
                if not path.is_file(): raise FileNotFoundError(path)
            else:
                path=download(cfg,url,cfg.raw_root/'cases'/case/f'{name}_{suffix}',page)
            files[suffix]=str(path)
            records.append(dict(dataset_id=case,species=species,source=page,url=url,
                local_path=str(path),sha256=file_hash(path),bytes=path.stat().st_size))
        write_json(cfg.raw_root/'cases'/case/'source_files.json',files)
    write_json(cfg.result_root/'01_data/real_sources.json',records)

def prepare(cfg,action='prepare'):
    if action=='reuse':
        return rebase_prepared(cfg)
    if action=='check':
        return {'new_bytes':storage_guard(cfg),'settings':asdict(cfg)}
    if action not in ('prepare','download'):
        raise ValueError('Publication preprocessing supports check/prepare/download')
    cfg.result_root.mkdir(parents=True,exist_ok=True)
    storage_guard(cfg)
    write_json(cfg.result_root/'study_settings.json',asdict(cfg))
    if action=='download':
        acquire_cases(cfg);return cfg.raw_root
    jobs=export_archives(cfg)
    rows=prepare_benchmark(cfg,jobs)
    pd.DataFrame(rows).drop(columns=['files'],errors='ignore').to_csv(cfg.processed_root/'benchmark_registry.csv',index=False)
    acquire_cases(cfg)
    from .real_cases import prepare_cases
    cases=prepare_cases(cfg)
    pd.DataFrame(cases).to_csv(cfg.processed_root/'case_registry.csv',index=False)
    return cfg.processed_root


def rebase_prepared(cfg):
    """Relocate verified prepared inputs on another machine; no regeneration."""
    from .resources import verify_model_inputs
    prepared=[]
    for kind in ('benchmark','case'):
        path=cfg.processed_root/f'{kind}_registry.csv'
        table=pd.read_csv(path)
        for index,row in table.iterrows():
            folder=cfg.processed_root/str(row.dataset_id)
            table.loc[index,'dataset_dir']=str(folder)
            table.loc[index,'truth_dir']=str(folder/'truth')
            verify_model_inputs(table.loc[index].to_dict())
            manifest_path=folder/'evidence_manifest.json'
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['dataset_dir']=str(folder);manifest['truth_dir']=str(folder/'truth')
            # Preserve content hashes and signature; only storage locations change.
            write_json(manifest_path,manifest)
        table.to_csv(path,index=False);prepared.append(str(path))
    return {'state':'verified_reused','registries':prepared}

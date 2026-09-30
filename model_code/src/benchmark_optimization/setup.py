"""Prepare a development registry and install pinned public implementations locally."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import urllib.request
import pandas as pd
from publication_study.resources import download, file_hash, write_json, verify_model_inputs

def guard(cfg, reserve=0):
    import os, shutil
    used=0
    for root in cfg.budget_roots:
        for directory, _, names in os.walk(root):
            for name in names:
                try: used += (Path(directory)/name).stat().st_size
                except FileNotFoundError: pass
    if used+reserve > cfg.maximum_new_gib*1024**3:
        raise RuntimeError('resource_limit: combined publication+optimization storage budget')
    if shutil.disk_usage(cfg.result_root.anchor).free-reserve < cfg.minimum_free_disk_gib*1024**3:
        raise RuntimeError('resource_limit: free disk')
    return used

def prepare(cfg):
    guard(cfg)
    table=pd.read_csv(cfg.source_registry)
    selected=table.loc[table.split.eq(cfg.development_split)].copy()
    if cfg.development_split != 'validation' or len(selected)!=6:
        raise ValueError('Round 01 is restricted to the six original validation datasets')
    for row in selected.to_dict('records'): verify_model_inputs(row)
    selected['original_split']=selected['split']; selected['split']='development'
    cfg.processed_root.mkdir(parents=True,exist_ok=True)
    # References, not copies: do not touch the immutable original prepared files.
    selected.to_csv(cfg.processed_root/'development_registry.csv',index=False)
    write_json(cfg.result_root/'01_plan/data_roles.json',dict(
        source_registry=str(cfg.source_registry), development_dataset_ids=selected.dataset_id.tolist(),
        original_test_status='previously unblinded; NOT independent confirmation',
        confirmation_registry=None, cases_enabled=False))
    return cfg.processed_root/'development_registry.csv'

def install(cfg):
    guard(cfg, reserve=50*1024**2)
    source=cfg.raw_root/'software'; source.mkdir(parents=True,exist_ok=True)
    library=cfg.raw_root/'R_library'; library.mkdir(parents=True,exist_ok=True)
    lock=source/'tigress_commit.json'
    if lock.exists(): commit=json.loads(lock.read_text(encoding='utf-8'))['commit']
    else:
        request=urllib.request.Request('https://api.github.com/repos/jpvert/tigress/commits/master',
                                      headers={'User-Agent':'JumpNicheRewire-public-research'})
        with urllib.request.urlopen(request,timeout=40) as response: commit=json.load(response)['sha']
        write_json(lock,dict(repository='https://github.com/jpvert/tigress',commit=commit))
    # Official R source is retained byte-for-byte; no local substitute algorithm.
    for name in ('tigress.R','stabilityselection.R'):
        download(cfg,f'https://raw.githubusercontent.com/jpvert/tigress/{commit}/R/{name}',
                 source/name,'https://github.com/jpvert/tigress')
    download(cfg,f'https://raw.githubusercontent.com/jpvert/tigress/{commit}/DESCRIPTION',
             source/'TIGRESS_DESCRIPTION','https://github.com/jpvert/tigress')
    write_json(source/'install_settings.json',dict(library=str(library)))
    subprocess.run([str(cfg.rscript),'--vanilla',str(Path(__file__).with_name('install.R'))],
                   cwd=source,check=True,timeout=900)
    guard(cfg)
    return source/'installed_versions.json'

def run(cfg, action):
    if action in ('prepare','reuse'): return prepare(cfg)
    if action=='install': return install(cfg)
    if action=='check': return {'combined_bytes':guard(cfg),'settings':asdict(cfg)}
    raise ValueError('Optimization preprocessing: prepare / reuse / install / check')

"""Sequential, resumable experiment scheduler with disk/RAM guardrails."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import time
import traceback
import psutil
import pandas as pd
from .resources import storage_guard,file_hash,write_json,verify_model_inputs

def implementation_hash():
    import hashlib
    root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    # Scientific functions/configuration, not report layout or status timestamps.
    files=list((root/'jump_niche_rewire').glob('*.py'))
    files+=[Path(__file__).with_name('inference.py'),root.parent/'configs/public_sim_data/publication_study.py']
    for p in sorted(files): digest.update(p.name.encode());digest.update(p.read_bytes())
    return digest.hexdigest()

def execute(cfg,dataset_id,method):
    registry_path=cfg.processed_root/('case_registry.csv' if dataset_id.startswith('case_') else 'benchmark_registry.csv')
    current=pd.read_csv(registry_path)
    selected=current.loc[current.dataset_id.eq(dataset_id)]
    if len(selected)!=1:raise ValueError('Dataset registry must contain exactly one selected row')
    verify_model_inputs(selected.iloc[0].to_dict())
    root=cfg.result_root/'02_runs'/dataset_id/method
    root.mkdir(parents=True,exist_ok=True)
    status=root/'run_status.json'
    signature=implementation_hash()
    if status.exists():
        state=json.loads(status.read_text(encoding='utf-8'))
        if state['state']=='running' and psutil.pid_exists(state.get('pid',-1)):
            raise RuntimeError(f'This dataset still has a live inference process: {state.get("pid")}; adopt or await it before restarting')
        if state['state']=='completed':
            if len(selected)!=1 or selected.iloc[0]['signature']!=state.get('input_signature'):
                raise ValueError(f'Input registry changed for {dataset_id}; refusing stale cache')
            if state.get('implementation_hash')!=signature:
                raise ValueError(f'Code/config changed for cached run {root}; use a new study version')
            if state.get('prediction_sha256')!=file_hash(root/'global_grn.parquet'):
                raise ValueError('Prediction cache checksum mismatch')
            print('[reuse]',dataset_id,method,flush=True);return state
        # Keep every failed/interrupted attempt, including the full traceback/log.
        archive=root/'attempt_history'/str(time.time_ns())
        archive.mkdir(parents=True,exist_ok=True)
        import shutil
        shutil.copy2(status,archive/'run_status.json')
        if (root/'run.log').exists():shutil.copy2(root/'run.log',archive/'run.log')
    storage_guard(cfg)
    if psutil.virtual_memory().available < cfg.minimum_available_ram_gib*1024**3:
        raise RuntimeError('resource_limit: not enough available memory to start')
    python=cfg.model_python if method=='JumpNicheRewire' else cfg.tree_python
    project=Path(__file__).resolve().parents[2]
    env=os.environ.copy()
    env['PYTHONPATH']=str(project)+os.pathsep+str(project/'src')
    prefix=python.parent
    env['PATH']=os.pathsep.join([str(prefix),str(prefix/'Library/bin'),str(prefix/'Scripts'),env.get('PATH','')])
    env['CONDA_PREFIX']=str(prefix)
    env['PYTHONUTF8']='1'
    env['MKL_THREADING_LAYER']='SEQUENTIAL'
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    code=f"from publication_study.inference import run_inference; s=run_inference({dataset_id!r},{method!r}); raise SystemExit(0 if s['state']=='completed' else 1)"
    print('[start]',dataset_id,method,flush=True)
    peak=0;limit_error=None
    with (root/'run.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([str(python),'-u','-c',code],cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        while process.poll() is None:
            try:
                parent=psutil.Process(process.pid)
                rss=sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True) if p.is_running())
                peak=max(peak,rss)
                storage_guard(cfg)
                if rss>cfg.maximum_worker_rss_gib*1024**3 or psutil.virtual_memory().available<cfg.minimum_available_ram_gib*1024**3:
                    raise RuntimeError('resource_limit: worker RSS or available RAM threshold reached')
            except psutil.NoSuchProcess:pass
            except Exception as error:
                limit_error=str(error)
                # Only this scheduler's child tree, never unrelated user processes.
                try:
                    children=psutil.Process(process.pid).children(recursive=True)
                    for child in children:child.terminate()
                except psutil.NoSuchProcess:pass
                process.terminate();process.wait(timeout=15);break
            time.sleep(2)
    state=json.loads(status.read_text(encoding='utf-8')) if status.exists() else dict(dataset_id=dataset_id,method=method,state='crashed')
    state.update(exit_code=process.returncode,peak_rss_gib=peak/1024**3,implementation_hash=signature)
    if limit_error:state.update(state='resource_limit',error=limit_error)
    elif state['state']=='running':state['state']='crashed'
    write_json(status,state)
    print('[finished]',dataset_id,method,state['state'],flush=True)
    write_json(cfg.result_root/'latest_progress.json',dict(dataset_id=dataset_id,method=method,
        state=state['state'],updated_epoch=time.time(),new_bytes=storage_guard(cfg)))
    return state

def run(cfg,action='run'):
    if action=='status':
        return [json.loads(p.read_text(encoding='utf-8')) for p in (cfg.result_root/'02_runs').rglob('run_status.json') if 'attempt_history' not in p.parts]
    if action=='check': return dict(new_bytes=storage_guard(cfg),configuration=asdict(cfg))
    if action=='evaluate':
        from .analysis import analyze
        return analyze(cfg)
    if action not in ('run','validation'):raise ValueError('Actions: check/status/run/validation/evaluate')
    registry=pd.read_csv(cfg.processed_root/'benchmark_registry.csv')
    expected=len(cfg.trees)*len(cfg.noise_levels)*(len(cfg.validation_replicates)+len(cfg.test_replicates))+len(cfg.published_spatial)
    if len(registry)!=expected:raise ValueError(f'Incomplete benchmark registry: {len(registry)}/{expected}')
    lock=cfg.result_root/'study_process.json'
    if lock.exists():
        previous=json.loads(lock.read_text(encoding='utf-8'))
        pid=previous.get('pid',-1)
        if psutil.pid_exists(pid) and abs(psutil.Process(pid).create_time()-previous.get('process_started',0))<1:
            raise RuntimeError('This study already has an active scheduler')
    write_json(lock,dict(pid=os.getpid(),process_started=psutil.Process().create_time(),state='running'))
    results=[]
    try:
        # Validation is a numerical acceptance stage, not a test-driven hyperparameter search.
        for row in registry.loc[registry.split.eq('validation')].itertuples():
            for method in cfg.methods: results.append(execute(cfg,row.dataset_id,method))
        write_json(cfg.result_root/'02_runs/validation_audit.json',results)
        if any(s['state']!='completed' for s in results):
            raise RuntimeError('Validation incomplete or failed; do not unlock test')
        frozen=dict(selection='prespecified existing parameters; no test optimization',settings=asdict(cfg),
            implementation_hash=implementation_hash(),validation_runs=len(results),test_truth_read_for_selection=False)
        project=Path(__file__).resolve().parents[2]
        frozen['default_config_sha256']={str(p.relative_to(project)):file_hash(p) for p in (project/'configs/common').glob('*.py')}
        path=cfg.result_root/'02_runs/frozen_parameters.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))['implementation_hash']!=frozen['implementation_hash']:
            raise ValueError('Frozen implementation changed; new study version required')
        if path.exists():
            saved=json.loads(path.read_text(encoding='utf-8'))
            old_defaults=saved.get('default_config_sha256')
            if old_defaults is None:
                test_ids=registry.loc[registry.split.eq('test'),'dataset_id']
                if any((cfg.result_root/'02_runs'/key).exists() for key in test_ids):
                    raise ValueError('Cannot extend freeze provenance after test started')
                # Extend audit metadata before the first test fit, without changing any parameter.
                frozen['provenance_extension']='Default-file hashes added before first test; no scientific setting changed.'
            elif old_defaults!=frozen['default_config_sha256']:
                raise ValueError('Frozen default configuration changed; new study version required')
        write_json(path,frozen)
        if action=='validation':
            write_json(lock,dict(pid=os.getpid(),state='validation_complete'))
            return path
        for row in registry.loc[registry.split.eq('test')].itertuples():
            for method in cfg.methods:results.append(execute(cfg,row.dataset_id,method))
        cases=pd.read_csv(cfg.processed_root/'case_registry.csv')
        if set(cases.dataset_id)!={'case_'+c for c in cfg.cases}:raise ValueError('Incomplete case registry')
        for row in cases.itertuples():results.append(execute(cfg,row.dataset_id,'JumpNicheRewire'))
        write_json(cfg.result_root/'02_runs/all_run_audit.json',results)
        from .analysis import analyze
        result=analyze(cfg)
        write_json(lock,dict(pid=os.getpid(),state='finished_computation',all_runs_completed=all(s['state']=='completed' for s in results)))
        return result
    except Exception as error:
        write_json(lock,dict(pid=os.getpid(),state='stopped',error=str(error),traceback=traceback.format_exc()))
        raise

def continue_after_validation():
    """Internal dependency queue, not a second user-facing entry program."""
    from configs.public_sim_data.publication_study import StudyConfig
    cfg=StudyConfig()
    pending=cfg.result_root/'continuation_status.json'
    write_json(pending,dict(pid=os.getpid(),state='waiting_for_existing_validation',started_epoch=time.time()))
    # Keep this authorized overnight batch awake without changing system power settings.
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        lock=cfg.result_root/'study_process.json'
        previous=json.loads(lock.read_text(encoding='utf-8')) if lock.exists() else {}
        pid=previous.get('pid',-1);created=previous.get('process_started',0)
        while psutil.pid_exists(pid):
            try:
                if abs(psutil.Process(pid).create_time()-created)>1:break
            except psutil.NoSuchProcess:break
            time.sleep(20)
        validation=cfg.result_root/'02_runs/validation_audit.json'
        if not validation.exists() or any(x.get('state')!='completed' for x in json.loads(validation.read_text(encoding='utf-8'))):
            raise RuntimeError('Validation did not finish successfully; review before continuing')
        write_json(pending,dict(pid=os.getpid(),state='running_full_study',started_epoch=time.time()))
        result=run(cfg,'run')
        write_json(pending,dict(pid=os.getpid(),state='finished',result=result,finished_epoch=time.time()))
    except Exception as error:
        write_json(pending,dict(pid=os.getpid(),state='stopped',error=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

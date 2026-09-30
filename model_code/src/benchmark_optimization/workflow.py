"""Finite, resumable development search. No automatic confirmation or case analysis."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import pandas as pd
import psutil
from publication_study.resources import file_hash,write_json,verify_model_inputs
from .setup import guard

def fingerprint(cfg):
    project=Path(__file__).resolve().parents[2]
    files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.R'))
    files+=list((project/'src/jump_niche_rewire').glob('*.py'))
    files+=list((project/'configs/common').glob('*.py'))
    files+=[project/'configs/public_sim_data/benchmark_optimization.py',
            project/'src/publication_study/inference.py',project/'configs/public_sim_data/publication_study.py',
            project/'src/independent_baselines/comparison.py']
    files+=list((cfg.raw_root/'software').glob('*.R'))+list((cfg.raw_root/'software').glob('*versions.json'))
    h=hashlib.sha256()
    for path in sorted(files): h.update(str(path.relative_to(project)).encode());h.update(path.read_bytes())
    return h.hexdigest()

def rows_for_development(cfg):
    rows=pd.read_csv(cfg.processed_root/'development_registry.csv')
    if len(rows)!=6 or rows.dataset_id.duplicated().any() or not rows.split.eq('development').all():
        raise ValueError('Expected exactly 6 uniquely identified development datasets')
    old=pd.read_csv(cfg.source_registry)
    allowed=set(old.loc[old.split.eq('validation'),'dataset_id'])
    if set(rows.dataset_id)!=allowed: raise ValueError('Unblinded test cannot enter round-01 registry')
    for row in rows.to_dict('records'): verify_model_inputs(row)
    return rows

def lock_plan(cfg, rows):
    plan=dict(settings=asdict(cfg),candidates=cfg.candidates(),
              methods=list(cfg.methods),selection_metric='AP',selection='equal-condition mean AP; candidate_id breaks exact ties',
              implementation_hash=fingerprint(cfg),registry_sha256=file_hash(cfg.processed_root/'development_registry.csv'),
              expected_tasks=len(rows)*sum(len(v) for v in cfg.candidates().values()),
              inference_truth_access=False,confirmation_enabled=False,cases_enabled=False)
    plan=json.loads(json.dumps(plan,default=str))
    path=cfg.result_root/'01_plan/search_plan.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=plan:
        raise ValueError('Search plan/code changed. Preserve this round; use a new result_root for another round.')
    if not path.exists(): write_json(path,plan)
    return plan

def live(record):
    try: return abs(psutil.Process(record['pid']).create_time()-record.get('process_started',-1))<1
    except (KeyError,psutil.Error): return False

def execute(cfg,row,method,candidate,plan):
    root=cfg.result_root/'02_runs'/method/candidate/row['dataset_id'];root.mkdir(parents=True,exist_ok=True)
    status=root/'run_status.json'
    if status.exists():
        old=json.loads(status.read_text(encoding='utf-8'))
        if old.get('state')=='running' and live(old): raise RuntimeError('Verified live worker exists; do not duplicate')
        if old.get('state')=='completed':
            verify_model_inputs(row)
            if (old.get('implementation_hash')!=plan['implementation_hash'] or
                old.get('input_signature')!=row['signature'] or
                old.get('prediction_sha256')!=file_hash(root/'global_grn.parquet')):
                raise ValueError('Changed input/implementation/prediction cache')
            print('[reuse]',method,candidate,row['dataset_id'],flush=True);return old
        # Explicit resume keeps all attempts; the original failure is never erased.
        archive=root/'attempt_history'/str(time.time_ns());archive.mkdir(parents=True)
        shutil.copy2(status,archive/'run_status.json')
        if (root/'run.log').exists(): shutil.copy2(root/'run.log',archive/'run.log')
    guard(cfg)
    if psutil.virtual_memory().available<cfg.minimum_available_ram_gib*1024**3:
        raise RuntimeError('resource_limit: available RAM before launching')
    python=cfg.tree_python if method in ('GENIE3','GRNBoost2') else cfg.model_python
    project=Path(__file__).resolve().parents[2]
    env=os.environ.copy();env['PYTHONPATH']=str(project)+os.pathsep+str(project/'src')
    env['PYTHONUTF8']='1';env['PYTHONDONTWRITEBYTECODE']='1';env['CONDA_PREFIX']=str(python.parent)
    env['PATH']=os.pathsep.join([str(python.parent),str(python.parent/'Library/bin'),str(python.parent/'Scripts'),env.get('PATH','')])
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[key]='1'
    env['MKL_THREADING_LAYER']='SEQUENTIAL'
    code=f"from benchmark_optimization.worker import run_worker; s=run_worker({row['dataset_id']!r},{method!r},{candidate!r}); raise SystemExit(0 if s['state']=='completed' else 1)"
    start=time.time();peak=0;error=None;birth=None
    print('[start]',method,candidate,row['dataset_id'],flush=True)
    with (root/'run.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([str(python),'-u','-c',code],cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try: birth=psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess: pass
        while process.poll() is None:
            try:
                parent=psutil.Process(process.pid)
                rss=sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True) if p.is_running())
                peak=max(peak,rss);guard(cfg)
                if rss>cfg.maximum_worker_rss_gib*1024**3 or psutil.virtual_memory().available<cfg.minimum_available_ram_gib*1024**3:
                    raise RuntimeError('resource_limit: worker RSS/available RAM')
                if time.time()-start>cfg.maximum_task_hours*3600: raise RuntimeError('time_limit: per-task budget')
            except psutil.NoSuchProcess: pass
            except Exception as exc:
                error=str(exc)
                # Only terminate this process tree, whose Popen handle is owned here.
                try:
                    for child in psutil.Process(process.pid).children(recursive=True): child.terminate()
                except psutil.NoSuchProcess: pass
                process.terminate();process.wait(timeout=15);break
            time.sleep(2)
    state=json.loads(status.read_text(encoding='utf-8')) if status.exists() else dict(
        dataset_id=row['dataset_id'],method=method,candidate=candidate,state='crashed',input_signature=row['signature'])
    state.update(implementation_hash=plan['implementation_hash'],exit_code=process.returncode,
                 peak_rss_gib=peak/1024**3,pid=process.pid,process_started=birth)
    if error: state.update(state='resource_limit' if error.startswith('resource_limit') else 'time_limit',error=error)
    elif process.returncode!=0 and state.get('state')=='completed': state.update(state='crashed',error='nonzero exit after write')
    elif state.get('state')=='running': state['state']='crashed'
    write_json(status,state)
    print('[finished]',method,candidate,state['state'],flush=True)
    write_json(cfg.result_root/'latest_progress.json',dict(dataset_id=row['dataset_id'],method=method,
               candidate=candidate,state=state['state'],updated_epoch=time.time(),combined_bytes=guard(cfg)))
    return state

def run(cfg,action='run'):
    if action in ('test','freeze','cases'):
        raise PermissionError('This round is development-only. New untouched public confirmation data and a locked protocol are required; no cases yet.')
    if action=='status': return [json.loads(p.read_text(encoding='utf-8')) for p in (cfg.result_root/'02_runs').rglob('run_status.json') if 'attempt_history' not in p.parts]
    rows=rows_for_development(cfg)
    if action=='check': return dict(datasets=len(rows),methods=cfg.methods,expected_tasks=len(rows)*sum(map(len,cfg.candidates().values())),combined_bytes=guard(cfg),cases_enabled=False)
    if action=='evaluate':
        from .evaluate import evaluate
        return evaluate(cfg)
    if action not in ('run','smoke'): raise ValueError('Optimization main: check / run / smoke / status / evaluate')
    if not (cfg.raw_root/'software/installed_versions.json').exists(): raise RuntimeError('Run preprocessing action install first')
    for status in (cfg.result_root/'02_runs').rglob('run_status.json'):
        if 'attempt_history' in status.parts: continue
        previous=json.loads(status.read_text(encoding='utf-8'))
        if previous.get('state')=='running' and live(previous):
            raise RuntimeError('An orphan/live worker still exists; audit it before starting any new queue')
    lock=cfg.result_root/'study_process.json'
    # Exclusive create prevents two schedulers racing on an absent lock.
    lock.parent.mkdir(parents=True,exist_ok=True)
    if lock.exists():
        previous=json.loads(lock.read_text(encoding='utf-8'))
        if live(previous): raise RuntimeError('An optimization scheduler is already alive')
        archive=cfg.result_root/'scheduler_history'/f'{time.time_ns()}.json'
        archive.parent.mkdir(exist_ok=True);lock.replace(archive)
    process_state=dict(state='running',pid=os.getpid(),process_started=psutil.Process().create_time(),action=action)
    with lock.open('x',encoding='utf-8') as stream: json.dump(process_state,stream)
    if os.name=='nt': __import__('ctypes').windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        plan=lock_plan(cfg,rows);results=[]
        # Install/integration checks first; all methods remain on the fixed roster even if they fail.
        for method in ('PPCOR','CLR','TIGRESS','GRNBoost2','GENIE3','JumpNicheRewire'):
            if action=='smoke' and method not in ('PPCOR','CLR','TIGRESS'): continue
            candidates=list(cfg.candidates()[method])
            for candidate in candidates[:1] if action=='smoke' else candidates:
                for row in (rows.iloc[:1] if action=='smoke' else rows).to_dict('records'):
                    if fingerprint(cfg)!=plan['implementation_hash']: raise RuntimeError('Code changed while search is running; stop for provenance audit')
                    state=execute(cfg,row,method,candidate,plan);results.append(state)
                    if state['state']=='resource_limit': raise RuntimeError(state['error'])
        from .evaluate import evaluate
        evaluation=evaluate(cfg)
        process_state.update(state='smoke_complete' if action=='smoke' else 'development_complete',
                             fits_completed=sum(s['state']=='completed' for s in results),
                             fits_attempted=len(results),evaluation=evaluation,
                             confirmation_complete=False,cases_enabled=False)
        write_json(lock,process_state);return evaluation
    except BaseException as error:
        process_state.update(state='stopped',error=str(error),traceback=traceback.format_exc())
        write_json(lock,process_state);raise
    finally:
        if os.name=='nt': __import__('ctypes').windll.kernel32.SetThreadExecutionState(0x80000000)

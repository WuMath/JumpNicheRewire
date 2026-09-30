"""Recover the one known orphan fit after fixing the checkpoint-monitor race.

Internal one-time recovery, not a user-facing entry. It waits for the live fit
without starting a duplicate, finalizes its resource audit, then resumes main.
"""
import json
import os
import time
import psutil
from configs.public_sim_data.publication_study import StudyConfig
from .resources import storage_guard,write_json,file_hash,verify_model_inputs
from .workflow import implementation_hash,run

def recover():
    cfg=StudyConfig()
    import ctypes
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    pending=cfg.result_root/'continuation_status.json'
    write_json(pending,dict(pid=os.getpid(),state='recovering_live_fit',started_epoch=time.time()))
    try:
        # This exact worker finished between controller shutdown and recovery.
        # Its prediction and input are verified; no full-run RSS was retained.
        completed=cfg.result_root/'02_runs/published_tree3_500_cells110_genes_sigma0.5_1/GENIE3/run_status.json'
        if completed.exists():
            saved=json.loads(completed.read_text(encoding='utf-8'))
            if saved.get('state')=='completed' and not saved.get('implementation_hash'):
                import pandas as pd
                expected='3493b442e09f32f4dad809f778d82ec17692fa5692fc453596d75a9d70aaecce'
                assert implementation_hash()==expected, 'Scientific code changed; cannot recover provenance'
                assert saved.get('pid')==21832, 'Unexpected orphan identity'
                assert saved['prediction_sha256']==file_hash(completed.parent/'global_grn.parquet')
                registry=pd.read_csv(cfg.processed_root/'benchmark_registry.csv')
                row=registry.loc[registry.dataset_id.eq(saved['dataset_id'])].iloc[0].to_dict()
                verify_model_inputs(row)
                assert saved['input_signature']==row['signature']
                write_json(completed.parent/'orphan_status_before_recovery.json',saved)
                saved.update(implementation_hash=expected,
                    resource_audit_note='Known worker 21832 completed after controller shutdown. Input/prediction checksums verified; full-run peak RSS unavailable, not estimated.')
                write_json(completed,saved)
        for status in (cfg.result_root/'02_runs').rglob('run_status.json'):
            if 'attempt_history' in status.parts:continue
            state=json.loads(status.read_text(encoding='utf-8'))
            if state.get('state')!='running':continue
            pid=state.get('pid',-1);peak=0
            try:
                p=psutil.Process(pid)
                if abs(p.create_time()-state.get('started_epoch',0))>30:
                    raise RuntimeError('Unexpected PID identity during recovery')
                while p.is_running():
                    storage_guard(cfg)
                    memory=p.memory_info().rss;peak=max(peak,memory)
                    if memory>cfg.maximum_worker_rss_gib*1024**3 or psutil.virtual_memory().available<cfg.minimum_available_ram_gib*1024**3:
                        p.terminate()
                        raise RuntimeError('resource_limit during recovery')
                    time.sleep(2)
            except psutil.NoSuchProcess:pass
            state=json.loads(status.read_text(encoding='utf-8'))
            if state.get('state')=='running':state['state']='interrupted'
            state.update(implementation_hash=implementation_hash(),peak_rss_gib=peak/1024**3,
                resource_audit_note='Scheduler repaired while inference continued; RSS is maximum observed during recovery, not guaranteed full-run peak.')
            write_json(status,state)
        write_json(pending,dict(pid=os.getpid(),state='running_full_study',started_epoch=time.time()))
        result=run(cfg,'run')
        write_json(pending,dict(pid=os.getpid(),state='finished',result=result,finished_epoch=time.time()))
    except Exception as error:
        import traceback
        write_json(pending,dict(pid=os.getpid(),state='stopped',error=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

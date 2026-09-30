"""Serial, spawn-isolated task supervision; no CLI/request-file routing.

Sampling reacts to resource excess (not an OS allocation hard cap). All signals
are limited to a child started here or descendants with verified birth times.
"""
from dataclasses import dataclass
import contextlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
import time
import traceback
import uuid
import psutil
from .execution import atomic_json
from .persistence import ResourceGuard


@dataclass(frozen=True)
class SupervisorConfig:
    poll_seconds: float = .25
    termination_grace_seconds: float = 5.
    python_executable: str | None = None


def process_record(pid):
    process = psutil.Process(pid)
    return dict(pid=pid, process_started=process.create_time(), command=process.cmdline())


def verified_live(record):
    try:
        process = psutil.Process(record['pid'])
        return (abs(process.create_time()-record['process_started']) < .001 and
                process.cmdline() == record['command'] and process.is_running())
    except (psutil.Error, KeyError, TypeError):
        return False


def _pipe_ready(connection):
    try:
        return connection.poll()
    except (EOFError, BrokenPipeError):
        return False


def _child_entry(task, args, log_path, connection):
    # These are numerical library thread settings, never experiment paths.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    with Path(log_path).open('a', encoding='utf8', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            connection.send(dict(kind='started', **process_record(os.getpid())))
            try:
                value = task(*args)
                connection.send(dict(kind='result', result=value))
            except BaseException as error:
                traceback.print_exc()
                connection.send(dict(kind='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
                raise SystemExit(1)
            finally:
                connection.close()


def _stop_owned(process, descendants, options):
    for record in reversed(list(descendants.values())):
        if verified_live(record):
            with contextlib.suppress(psutil.Error): psutil.Process(record['pid']).terminate()
    if process.is_alive(): process.terminate()
    process.join(timeout=options.termination_grace_seconds)
    if process.is_alive(): process.kill(); process.join(timeout=options.termination_grace_seconds)
    for record in descendants.values():
        if verified_live(record):
            with contextlib.suppress(psutil.Error): psutil.Process(record['pid']).kill()


def run_isolated(task, args, output_root, resource_config, options=None):
    options = options or SupervisorConfig()
    if options.poll_seconds <= 0 or options.termination_grace_seconds <= 0:
        raise ValueError('Positive supervisor intervals required')
    root = Path(output_root); root.mkdir(parents=True, exist_ok=True)
    status_path, lock_path = root/'run_status.json', root/'supervisor.lock'
    old = json.loads(status_path.read_text(encoding='utf8')) if status_path.exists() else None
    if old and old.get('state') == 'running' and verified_live(old):
        raise RuntimeError('Verified live worker exists; refusing duplicate')
    # A stale controller lock is archived only after worker and controller checks.
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding='utf8'))
        if verified_live(lock): raise RuntimeError('Verified live controller exists; refusing duplicate')
        archived = root/'attempt_history'/uuid.uuid4().hex
        archived.mkdir(parents=True)
        lock_path.replace(archived/'stale_supervisor.lock')
    token = uuid.uuid4().hex
    owner = dict(**process_record(os.getpid()), token=token)
    with lock_path.open('x', encoding='utf8') as stream: json.dump(owner, stream)
    process = None
    descendants = {}
    try:
        if old:
            archived = root/'attempt_history'/uuid.uuid4().hex
            archived.mkdir(parents=True)
            atomic_json(archived/'run_status.json', old)
            if (root/'run.log').exists(): (root/'run.log').replace(archived/'run.log')
            if old.get('state') == 'running': atomic_json(archived/'interruption_audit.json',
                dict(state='interrupted', reason='no matching live worker', checked_epoch=time.time()))
        guard = ResourceGuard(resource_config); guard()
        context = mp.get_context('spawn')
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_child_entry, args=(task, args, root/'run.log', sender))
        # Serial launcher may select the installed official baseline interpreter.
        # Pass task/config via spawn, never through environment request paths.
        from multiprocessing.spawn import get_executable
        previous_executable = get_executable()
        previous_path = os.environ.get('PATH', '')
        previous_sys_path = list(sys.path)
        try:
            if options.python_executable:
                executable = Path(options.python_executable).resolve(strict=True)
                mp.set_executable(str(executable))
                os.environ['PATH'] = os.pathsep.join((str(executable.parent),
                    str(executable.parent/'Library/bin'), str(executable.parent/'Scripts'), previous_path))
                # spawn serializes the parent's sys.path. Remap interpreter-owned
                # paths too, otherwise NumPy from the model env enters tree Python.
                remapped = []
                for entry in previous_sys_path:
                    try:
                        relative = Path(entry).resolve().relative_to(Path(sys.prefix).resolve())
                        entry = str(executable.parent/relative)
                    except ValueError:
                        pass
                    remapped.append(entry)
                sys.path[:] = remapped
            process.start()
        finally:
            mp.set_executable(previous_executable)
            os.environ['PATH'] = previous_path
            sys.path[:] = previous_sys_path
        sender.close()
        started = time.monotonic()
        state = dict(state='running', controller=owner, started_epoch=time.time(), attempt_token=token)
        # Capture the actual spawned handle's PID; no historical PID is signaled.
        with contextlib.suppress(psutil.Error): state.update(process_record(process.pid))
        atomic_json(status_path, state)
        peak, final_message, limit_error = 0, None, None
        while process.is_alive() or _pipe_ready(receiver):
            try:
                while _pipe_ready(receiver):
                    message = receiver.recv()
                    if message['kind'] == 'started':
                        state.update({k: message[k] for k in ('pid', 'process_started', 'command')})
                        atomic_json(status_path, state)
                    else: final_message = message
            except (EOFError, BrokenPipeError):
                pass
            if not process.is_alive(): break
            try:
                parent = psutil.Process(process.pid)
                if 'process_started' in state and abs(parent.create_time()-state['process_started']) >= .001:
                    raise RuntimeError('Worker process identity changed')
                total = parent.memory_info().rss
                for child in parent.children(recursive=True):
                    try:
                        record = process_record(child.pid)
                        descendants[(record['pid'], record['process_started'])] = record
                        total += child.memory_info().rss
                    except psutil.Error: pass
                peak = max(peak, total)
                if total > resource_config.maximum_worker_rss_gib*1024**3:
                    raise RuntimeError('resource_limit: worker-tree RSS')
                if time.monotonic()-started > resource_config.maximum_task_hours*3600:
                    raise RuntimeError('resource_limit: worker time')
                guard()
            except psutil.NoSuchProcess:
                pass
            except RuntimeError as error:
                limit_error = str(error)
                _stop_owned(process, descendants, options)
                break
            time.sleep(options.poll_seconds)
        process.join(timeout=options.termination_grace_seconds)
        # Catch a final message sent between the previous poll and child exit.
        try:
            while _pipe_ready(receiver):
                message = receiver.recv()
                if message['kind'] != 'started': final_message = message
        except (EOFError, BrokenPipeError):
            pass
        receiver.close()
        if process.is_alive():
            limit_error = 'worker_exit_timeout'; _stop_owned(process, descendants, options)
        if any(verified_live(record) for record in descendants.values()):
            _stop_owned(process, descendants, options)
            limit_error = limit_error or 'worker_left_live_descendants'
        if limit_error:
            state.update(state='resource_limit' if limit_error.startswith('resource_limit') else 'failed', error=limit_error)
        elif process.exitcode == 0 and final_message and final_message['kind'] == 'result':
            result = final_message['result']
            child_state = result.get('state', 'completed') if isinstance(result, dict) else 'completed'
            state.update(state=child_state, result=result)
        elif final_message and final_message['kind'] == 'failed':
            state.update(state='failed', **{k: v for k, v in final_message.items() if k != 'kind'})
            if state.get('error', '').startswith('resource_limit'):
                state['state'] = 'resource_limit'
        else:
            state.update(state='crashed', error='Worker exited without a successful result')
        state.update(exit_code=process.exitcode, peak_sampled_worker_tree_rss_bytes=peak or None,
            observed_descendants=list(descendants.values()),
            memory_measurement='sampled RSS, not an OS allocation cap or guaranteed instantaneous peak',
            runtime_seconds=time.monotonic()-started, finished_epoch=time.time())
        atomic_json(status_path, state)
        return state
    except BaseException:
        if process is not None and process.is_alive(): _stop_owned(process, descendants, options)
        atomic_json(status_path, dict(state='interrupted_or_controller_failure', traceback=traceback.format_exc(), controller=owner))
        raise
    finally:
        if lock_path.exists() and json.loads(lock_path.read_text(encoding='utf8')).get('token') == token:
            lock_path.unlink()


def run_serial_jobs(jobs, resource_config, options=None):
    results = []
    for task, args, output_root in jobs:
        state = run_isolated(task, args, output_root, resource_config, options)
        results.append(state)
        # An engineering/model failure doesn't cancel later data. A resource
        # violation stops safely instead of repeatedly stressing the machine.
        if state['state'] == 'resource_limit': break
    return results

"""Scheduling."""
import subprocess
import sys
from typing import Dict, List
import yaml


def _log_cpe(exc):
    print(f"Error running scheduler subprocess, return code: {exc.returncode}", file=sys.stderr)
    stdout = exc.stdout.decode().strip()
    if stdout:
        print("stdout:", file=sys.stderr)
        print(stdout, file=sys.stderr)
    stderr = exc.stderr.decode().strip()
    if stderr:
        print("stderr:", file=sys.stderr)
        print(stderr, file=sys.stderr)


def sched_pipeline(model_name, buffers_in, buffers_out, batch_size, dtype='torch.float32',
                   models_file=None, dev_types_file=None, dev_file=None,
                   app_paths=None) -> List[Dict[str, List[int]]]:
    """Schedule the pipeline."""
    if app_paths is None:
        app_paths = []
    assert isinstance(app_paths, (list, tuple))
    args = ['-i', str(buffers_in), '-o', str(buffers_out),
            '-b', str(batch_size),
            '-d', dtype,
            '-m', model_name]
    if models_file:
        args += ['-M', models_file]
    if dev_types_file:
        args += ['-T', dev_types_file]
    if dev_file:
        args += ['-D', dev_file]

    proc = None
    for app_path in app_paths:
        try:
            proc = subprocess.run([app_path] + args, capture_output=True, check=True)
            break
        except FileNotFoundError:
            pass
        except subprocess.CalledProcessError as e:
            _log_cpe(e)
            raise
    if proc is None:
        # try to find app on the system path
        try:
            proc = subprocess.run(['sched-pipeline'] + args, capture_output=True, check=True)
        except FileNotFoundError:
            print('Could not locate sched-pipeline application - is it on your PATH?')
            raise
        except subprocess.CalledProcessError as e:
            _log_cpe(e)
            raise

    stderr = proc.stderr.decode().strip()
    if stderr:
        print(stderr, file=sys.stderr)
    stdout = proc.stdout.decode()
    sched = yaml.safe_load(stdout)
    assert isinstance(sched, list)
    return sched

"""Scheduling."""
import subprocess
import sys
from typing import Dict, List
import yaml

def sched_pipeline(model_name, buffers_in, buffers_out, batch_size, dtype='torch.float32',
                   models_file=None, dev_types_file=None, dev_file=None,
                   app='sched-pipeline') -> List[Dict[str, List[int]]]:
    """Schedule the pipeline."""
    args = [app,
            '-i', str(buffers_in), '-o', str(buffers_out),
            '-b', str(batch_size),
            '-d', dtype,
            '-m', model_name]
    if models_file:
        args += ['-M', models_file]
    if dev_types_file:
        args += ['-T', dev_types_file]
    if dev_file:
        args += ['-D', dev_file]
    try:
        proc = subprocess.run(args, capture_output=True, check=True)
    except FileNotFoundError:
        print(f'Could not locate scheduling application - is it on your PATH?')
        raise
    stderr = proc.stderr.decode()
    if stderr:
        print(stderr, file=sys.stderr)
    stdout = proc.stdout.decode()
    sched = yaml.safe_load(stdout)
    assert isinstance(sched, list)
    return sched

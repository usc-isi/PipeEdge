"""Scheduling."""
import logging
import os
import subprocess
from typing import Dict, List, Optional
import yaml


logger = logging.getLogger(__name__)


def _log_cpe(exc):
    logger.info("Error running scheduler subprocess, return code: %d", {exc.returncode})
    stdout = exc.stdout.decode().strip()
    if stdout:
        logger.info("stdout:")
        logger.info(stdout)
    stderr = exc.stderr.decode().strip()
    if stderr:
        logger.error("stderr:")
        logger.error(stderr)


def sched_pipeline(model_name: str, buffers_in: int, buffers_out: int, batch_size: int,
                   dtype: str='torch.float32', models_file: Optional[str]=None,
                   dev_types_file: Optional[str]=None, dev_file: Optional[str]=None,
                   app_paths: Optional[List[str]]=None) -> List[Dict[str, List[int]]]:
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
        app_path = 'sched-pipeline'
        if os.name == 'nt':
            app_path += '.exe'
        try:
            proc = subprocess.run([app_path] + args, capture_output=True, check=True)
        except FileNotFoundError:
            logger.error('Could not locate sched-pipeline application - is it on your PATH?')
            raise
        except subprocess.CalledProcessError as e:
            _log_cpe(e)
            raise

    stderr = proc.stderr.decode().strip()
    if stderr:
        logger.warning(stderr)
    stdout = proc.stdout.decode()
    sched = yaml.safe_load(stdout)
    assert isinstance(sched, list)
    return sched

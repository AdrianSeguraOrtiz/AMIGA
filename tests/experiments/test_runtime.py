"""Subprocess cleanup preserves the runtime lifecycle contract."""
import subprocess
import sys
from scripts.experiments.amiga_exp.runtime import terminate_process_group


def test_cleanup_stops_separate_worker_and_tolerates_already_exited():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                               start_new_session=True)
    try:
        terminate_process_group(process)
        assert process.poll() is not None
        terminate_process_group(process)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

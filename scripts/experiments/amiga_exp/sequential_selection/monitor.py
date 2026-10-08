"""Read persisted run status and flag stale or dead supervisors."""
import json
import os
from pathlib import Path
import time


def status(run):
    run = Path(run).resolve()
    state = json.loads((run / 'state.json').read_text())
    pid = state.get('supervisor_pid')
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            pass
        except PermissionError:
            alive = True
    age = time.time() - state.get('accounted_at_unix', (run / 'state.json').stat().st_mtime)
    result = dict(state, supervisor_alive=alive, heartbeat_age_seconds=round(age, 1),
                  run=str(run), supervisor_log=str(run / 'supervisor.log'))
    if state['status'] == 'running_selection' and state.get('run') and Path(state['run']).resolve() != run:
        child = Path(state['run']) / 'state.json'
        if child.is_file():
            result['selection'] = status(child.parent)
            result['health'] = result['selection']['health']
            return result
    if state['status'] in ('running', 'running_selection', 'waiting_for_pilot', 'summarizing') and (not alive or age > 120):
        result['health'] = 'stale_or_interrupted; inspect supervisor.log before resuming'
    else:
        result['health'] = state['status']
    return result

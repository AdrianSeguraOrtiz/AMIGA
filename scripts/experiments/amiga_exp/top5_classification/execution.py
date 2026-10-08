"""Dispatch the same selection and held-out fitting steps with the new target."""
from . import selection_execution, outer_execution


def completed_job(run, job):
    module = selection_execution if job['stage'] in ('phase2', 'phase3') else outer_execution
    return module.completed_job(run, job)


def execute_job(root, contract, job, plan, run, destination, threads):
    if plan.get(job['id']) != job:
        raise ValueError('Worker job differs from the frozen plan')
    module = selection_execution if contract['mode'] == 'selection' else outer_execution
    return module.execute_job(root, contract, job, plan, run, destination, threads)

#!/usr/bin/env python3
from subprocess import check_output
from os.path import basename, abspath
from time import time
from pprint import pprint
import click
import boto3

job_def = {'gpu': 'ahassanAWSBatch'}
job_queue = {'gpu': 'ahassanRasterVisionGpu'}


@click.command()
@click.argument('local_dir')
@click.argument('s3_dir')
@click.argument('out_dir')
@click.argument('cmd')
@click.option('--name', type=str, default=None)
@click.option(
    '--sync_args',
    type=str,
    default=('--exclude ".ipynb_checkpoints/*" '
             '--exclude "__pycache__/*" '
             '--exclude "data/*" '
             '--exclude "batch_submit.py" '
             '--exclude "*.ipynb" '))
@click.option('--debug', is_flag=True)
def batch_submit(local_dir: str, s3_dir: str, out_dir: str, cmd: str,
                 name: str, sync_args: str, debug: bool):
    client = boto3.client('batch')
    timestamp = int(time())
    if name is not None:
        job_name = f'ssl-{name}-{timestamp}'
    else:
        job_name = f'ssl-{timestamp}'

    local_dir = abspath(local_dir)
    dir_name = basename(local_dir)

    sync_in_cmd = (f'aws s3 sync {s3_dir} {dir_name} {sync_args} && '
                   f'cd {dir_name}')
    sync_out_cmd = f'aws s3 sync . {out_dir} {sync_args}'

    if debug:
        print(cmd)
    else:
        print(f'Syncing from {local_dir} to {s3_dir}')
        print(
            check_output(
                f'aws s3 sync {local_dir} {s3_dir} {sync_args}', shell=True)
            .decode())

    cmd = f'{sync_in_cmd} && {cmd} && {sync_out_cmd}'
    cmd_list = ['/bin/bash', '-c', cmd]
    cmd_list = [s for s in cmd_list if len(s.strip()) > 0]
    job_id = _submit(client, cmd_list, job_name, debug=debug)
    print(f'<<< Submitted >>>')
    print(f'job_id: {job_id}\n')


def _submit(client, cmd_list, job_name, attempts=1, device='gpu', debug=False):
    kwargs = {
        'jobName': job_name,
        'jobQueue': job_queue[device],
        'jobDefinition': job_def[device],
        'containerOverrides': {
            'command': cmd_list
        },
        'retryStrategy': {
            'attempts': attempts
        }
    }
    pprint(kwargs)

    if debug:
        return 'example-job-id'

    job_id = client.submit_job(**kwargs)['jobId']
    return job_id


if __name__ == '__main__':
    batch_submit()

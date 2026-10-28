"""Finite, recoverable study supervisor. Launch detached from the chat terminal."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import copy
from datetime import datetime,timezone

import yaml
from src.training.provenance import provenance

ROOT=Path(__file__).resolve().parents[1]


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def status_report(output,matrix):
    rows=['# Council GPU study',f"\nSupervisor status: {matrix['status']}; heartbeat: {matrix.get('heartbeat_at')}.",
        '\nFresh exploratory assisted QMIX/IQL comparison. C5 is trained only if prerequisite stages pass.',
        '\n| Condition | Algorithm | Seed | Process | Stage | Saved RL steps | Latest gate |',
        '|---|---|---:|---|---|---:|---|']
    for job in matrix['jobs']:
        directory=Path(job.get('directory',output/'not_started'))
        try:report=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):report={}
        try:pointer=json.loads((directory/'latest_checkpoint.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):pointer={}
        current=report.get('current',{})
        rows.append(f"| {job.get('condition','main')} | {job['algorithm']} | {job['seed']} | {job['status']} | {current.get('stage','pending')} | {pointer.get('environment_steps',0):,} | {current.get('gate_passed','pending')} |")
    rows.extend(['\nA heartbeat confirms supervisor liveness. Increasing saved steps confirms recoverable training progress; expert preparation can precede the first checkpoint.',
        '\nFinished processes can have failed development gates or execution errors. Check each job outcome; completion is not a competence claim.',
        '\nMaximum budget: 2.73 million joint RL steps per algorithm/seed, with strict stopping. Expert transitions and optimization are additional, explicitly recorded costs.',
        '\n[Council decisions](../../docs/21_council_adjudication_2026-10-26.md)'])
    temporary=output/'status.md.tmp';temporary.write_text('\n'.join(rows)+'\n',encoding='utf-8')
    os.replace(temporary,output/'status.md')


def process_identity(pid):
    """PID plus birth token prevents accepting a later reused Windows PID."""
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return None
        try:
            code=wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value!=259:return None
            values=[wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle,*[ctypes.byref(v) for v in values]):return None
            return str((values[0].dwHighDateTime<<32)|values[0].dwLowDateTime)
        finally:kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
        return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except (OSError,IndexError):return None


def effective_status(matrix,now=None):
    if matrix.get('status')!='running':return matrix.get('status','unknown')
    if (now or time.time())-matrix.get('heartbeat_unix',0)>90:return 'stale_heartbeat'
    identity=process_identity(matrix.get('supervisor_pid',0))
    if identity is None or identity!=matrix.get('supervisor_birth'):return 'supervisor_not_alive'
    return 'running'


def study_jobs(config):
    conditions=config.get('conditions',{'main':dict(seeds=config['seeds'],overrides={})})
    return [dict(condition=name,algorithm=algorithm,seed=seed,status='queued')
        for name,spec in conditions.items() for seed in spec.get('seeds',config['seeds'])
        for algorithm in config['algorithms']]


def job_config(config,job):
    child=copy.deepcopy(config)
    child.update(config.get('conditions',{}).get(job.get('condition','main'),{}).get('overrides',{}))
    child['study']=config['study']+'_'+job.get('condition','main')
    return child


def run(config_path,output,parallel=2,resume=False):
    config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=True)
    state_path=output/'matrix.json';source=provenance(ROOT)
    if state_path.exists():
        if not resume:raise ValueError('Study already exists; choose a new directory or explicitly resume')
        matrix=json.loads(state_path.read_text(encoding='utf-8'))
        identity=process_identity(matrix.get('supervisor_pid',0))
        if identity is not None and identity==matrix.get('supervisor_birth'):
            raise ValueError('Study supervisor is already alive, even if its heartbeat is stale')
        if matrix['config']!=config:
            raise ValueError('Resume config differs; preserve this study and use a new directory')
        for job in matrix['jobs']:
            identity=process_identity(job.get('pid',0))
            if identity is not None and identity==job.get('process_birth'):
                raise ValueError('A previous child is still alive; do not duplicate its training')
    else:
        if resume:raise ValueError('No study exists to resume')
        matrix=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),source=source,config=config,
            jobs=study_jobs(config))
        (output/'protocol.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for folder,pattern in [('src','*.py'),('scripts','*.py'),('configs','*.yaml'),('tests','*.py')]:
                for path in sorted((ROOT/folder).rglob(pattern)):archive.write(path,path.relative_to(ROOT))
            for path in ROOT.glob('requirements*.txt'):archive.write(path,path.relative_to(ROOT))
    matrix.update(status='running',supervisor_pid=os.getpid(),supervisor_birth=process_identity(os.getpid()))
    pending=[job for job in matrix['jobs'] if job.get('status')!='finished']
    active=[]
    def save():
        matrix.update(heartbeat_unix=time.time(),heartbeat_at=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path,matrix)
        status_report(output,matrix)
    save()
    while pending or active:
        while pending and len(active)<parallel:
            job=pending.pop(0)
            directory=output/f"{job.get('condition','main')}_{job['algorithm']}_seed{job['seed']}"
            continuation=(directory/'latest_checkpoint.json').exists()
            if directory.exists() and not continuation:
                # Never overwrite partial expert preparation without a recoverable pointer.
                directory.rename(output/f"{directory.name}_interrupted_{time.time_ns()}")
            directory.mkdir(exist_ok=True)
            child_config=directory/'protocol.yaml'
            if not continuation:child_config.write_text(yaml.safe_dump(job_config(config,job),sort_keys=False),encoding='utf-8')
            command=[sys.executable,'-u','-m','scripts.train_research_protocol','--config',str(child_config),
                '--algorithm',job['algorithm'],'--seed',str(job['seed']),'--output',str(directory)]
            if continuation:command.append('--resume-protocol')
            log=(directory/'training.log').open('a',encoding='utf-8')
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            job.update(status='running',directory=str(directory),pid=child.pid,process_birth=process_identity(child.pid),resumed=continuation)
            active.append((job,child,log));save()
        for entry in list(active):
            job,child,log=entry;code=child.poll()
            if code is None:continue
            log.close();active.remove(entry)
            job.update(status='finished' if code==0 else 'execution_error',exit_code=code)
            result=Path(job['directory'])/'result.json'
            if result.exists():job['outcome']=json.loads(result.read_text(encoding='utf-8')).get('status')
            save()
        if time.time()-matrix['heartbeat_unix']>=10:save()
        time.sleep(.5)
    matrix.update(status='finished',finished_at=datetime.now(timezone.utc).isoformat(),
        source_unchanged=provenance(ROOT)['source_sha256']==source['source_sha256'])
    save()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parallel',type=int,default=2)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.status:
        matrix=json.loads((args.output/'matrix.json').read_text(encoding='utf-8'))
        print(json.dumps(dict(effective_status=effective_status(matrix),matrix=matrix),indent=2));return
    if args.parallel<1:raise ValueError('Parallel jobs must be positive')
    run(args.config.resolve(),args.output.resolve(),args.parallel,args.resume)


if __name__=='__main__':main()

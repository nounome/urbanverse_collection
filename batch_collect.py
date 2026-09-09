"""Sequential fixed-scene collection queue with bounded retries and resume."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import collect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes',nargs='+',default=list(collect.load_scenes()))
    p.add_argument('--count',type=int,default=1);p.add_argument('--start-seed',type=int,default=1000)
    p.add_argument('--profile',choices=['smoke','preview','formal'],default='smoke')
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--attempts',type=int,default=3)
    p.add_argument('--wall-timeout',type=float,default=3600)
    p.add_argument('--journal',type=Path,required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.count<1 or not 1<=a.attempts<=3: p.error('count >=1 and attempts in 1..3 required')
    if not set(a.scenes)<=set(collect.load_scenes()):p.error('Unknown scene')
    spec=dict(scenes=a.scenes,count=a.count,start_seed=a.start_seed,profile=a.profile,attempts=a.attempts)
    journal=json.loads(a.journal.read_text()) if a.journal.exists() else dict(spec=spec,tasks={})
    if journal['spec']!=spec:raise ValueError('Journal configuration differs; use a new journal')
    failures=0
    for scene in a.scenes:
        for index in range(a.count):
            key=f'{scene}/{index}'
            if journal['tasks'].get(key,{}).get('status')=='passed':continue
            task=journal['tasks'].setdefault(key,dict(status='pending',attempts=[]))
            for attempt in range(len(task['attempts']),a.attempts):
                seed=a.start_seed+index*a.attempts+attempt
                cmd=[sys.executable,str(collect.ROOT/'collect.py'),'run','--scene',scene,'--seed',str(seed),
                    '--profile',a.profile,'--gpu',str(a.gpu),'--wall-timeout',str(a.wall_timeout)]
                if a.dry_run:print(' '.join(cmd));break
                # Persist 'running' before launch: interrupted attempts are not
                # silently reused as success when the queue resumes.
                event=dict(seed=seed,status='running',started_at=datetime.now().isoformat())
                task['attempts'].append(event);a.journal.parent.mkdir(parents=True,exist_ok=True)
                a.journal.write_text(json.dumps(journal,indent=2))
                code=subprocess.run(cmd,cwd=collect.ROOT).returncode
                event.update(exit_code=code,status='passed' if code==0 else 'failed')
                task['status']=event['status'];a.journal.write_text(json.dumps(journal,indent=2))
                if code==0:break
            if task['status']!='passed':failures+=1
    return 0 if a.dry_run else int(bool(failures))


if __name__=='__main__':raise SystemExit(main())

"""Build calibrated vehicle USD cache once; explicit GPU operation, no simulation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import collect


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--wall-timeout',type=float,default=900);a=p.parse_args()
    root=collect.ROOT;catalog=root/'configs/dynamic_agents/catalogs/scene10_portable_15_vehicle_catalog.json'
    doc=json.loads(catalog.read_text());cache=root/doc['converted_usd_cache_dir'];run=cache.parent
    if all((cache/i/'vehicle.usd').is_file() for i in doc['default_asset_ids']):
        print('Vehicle cache exists; no hash check performed.');return 0
    if run.exists():raise RuntimeError('Incomplete existing cache; inspect before replacing: '+str(run))
    def query(cmd):return subprocess.check_output(cmd,text=True,timeout=10)
    inventory=dict(nvml=query(['nvidia-smi','-L']),pci=query(['lspci']),nodes=[str(p) for p in Path('/dev').glob('nvidia[0-9]*')],
        state=query(['nvidia-smi']),git=query(['git','-C',str(root),'rev-parse','HEAD']))
    n=sum(line.startswith('GPU ') for line in inventory['nvml'].splitlines())
    pci=sum('NVIDIA' in line and ('VGA compatible' in line or '3D controller' in line) for line in inventory['pci'].splitlines())
    if n!=pci or n!=len(inventory['nodes']):raise RuntimeError('GPU inventory mismatch')
    def free():return int(query(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits']).strip())
    if free()<7500:raise RuntimeError('Need 7500 MiB current free VRAM for this initial conversion budget')
    run.mkdir(parents=True);(run/'metadata').mkdir();(run/'metadata/prelaunch.json').write_text(json.dumps(inventory,indent=2))
    cmd=[sys.executable,str(root/'tools/urbanverse/dynamic_agents/assets/convert_vehicle_catalog.py'),
         '--catalog',str(catalog),'--run-dir',str(run),'--gpu',str(a.gpu),'--asset-count',str(len(doc['default_asset_ids']))]
    reason=None;samples=[];started=time.monotonic()
    with (run/'run_log.txt').open('x') as log:
        child=subprocess.Popen(cmd,cwd=root,env={**os.environ,'OMNI_KIT_ACCEPT_EULA':'YES'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                available=free();samples.append(available)
                if available<1536 or time.monotonic()-started>a.wall_timeout:
                    reason='memory reserve or wall timeout';break
                time.sleep(2)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
            code=child.wait()
            (run/'metadata/process_exit.json').write_text(json.dumps(dict(exit_code=code,stop_reason=reason,free_mib_samples=samples)))
    summary=json.loads((run/'metadata/summary.json').read_text()) if (run/'metadata/summary.json').exists() else {}
    return int(code!=0 or reason is not None or summary.get('status')!='success')


if __name__=='__main__':raise SystemExit(main())

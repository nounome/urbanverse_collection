"""Standalone entry points. No simulation starts without the explicit run command."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))


def load_scenes():return json.loads((ROOT/'scenes.json').read_text())


def doctor(assets=False):
    from urbanverse.dynamic_agents.config import TrafficSceneConfig
    from urbanverse.dynamic_agents.navigation.mesh_loop_workflow import MeshMap
    from urbanverse.dynamic_agents.pedestrians.walkable_regions import WalkableRegionConfig
    errors=[]
    for name,row in load_scenes().items():
        try:
            # Config loader deliberately requires source/assets: config-only check
            # below validates just durable inputs, then optional full asset check.
            for key in ('traffic_config','mixed_config','reference_route','inventory'):
                if not (ROOT/row[key]).is_file():raise FileNotFoundError(row[key])
            MeshMap(ROOT/row['inventory'])
            mixed_path=ROOT/row['mixed_config']
            mixed=json.loads(mixed_path.read_text())
            WalkableRegionConfig.load((mixed_path.parent/mixed['walkable_regions']).resolve())
            if assets:TrafficSceneConfig.load(ROOT/row['traffic_config'])
            print(name,'config OK',row['qualification'])
        except Exception as exc:errors.append(f'{name}: {exc}')
    if assets:
        from urbanverse.dynamic_agents.navigation.policy_selection import resolve_policy
        try:resolve_policy()
        except Exception as exc:errors.append(str(exc))
        policy_manifest=ROOT/'data/locomotion_policies/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39/manifest.json'
        if not policy_manifest.is_file():errors.append('Missing policy source manifest; run download_assets.py policy')
        catalog=json.loads((ROOT/'configs/dynamic_agents/catalogs/scene10_portable_15_vehicle_catalog.json').read_text())
        cache=ROOT/catalog['converted_usd_cache_dir']
        missing=[asset for asset in catalog['default_asset_ids'] if not (cache/asset/'vehicle.usd').is_file()]
        if missing:errors.append(f'Missing {len(missing)} converted vehicle assets; run prepare_vehicle_cache.py')
        for p in ('repos/IsaacLab/apps','data/isaacsim_assets_4_5/Isaac/People',
                  'data/isaacsim_assets_4_5/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd'):
            if not (ROOT/p).exists():errors.append('Missing: '+p)
    for error in errors:print(error,file=sys.stderr)
    print('No RTX or new-machine runtime acceptance is implied by doctor.')
    return int(bool(errors))


def plan(scene,seed):
    from urbanverse.dynamic_agents.navigation.mesh_go2_route import generate
    row=load_scenes()[scene]
    reference=json.loads((ROOT/row['reference_route']).read_text())
    spec=dict(row['go2_spec'],seed=seed)
    folder=ROOT/'configs/generated'/f'{scene}_{seed}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    folder.mkdir(parents=True,exist_ok=False)
    settings=dict(go2_route=spec,removed_instance_roots=reference.get('generation',{}).get('removed_instance_roots',[]))
    (folder/'settings.json').write_text(json.dumps(settings,indent=2))
    generate(ROOT/row['inventory'],folder/'settings.json',folder/'route.json')
    return folder/'route.json'


def command(scene,profile,gpu,route,duration):
    row=load_scenes()[scene]
    env=dict(os.environ,PYTHONPATH=str(ROOT/'tools'),OMNI_KIT_ACCEPT_EULA='YES',
             ALLOW_SHARED_GPU='1',POLICY_KIND='robot_lab',ROAMING_HEADLESS='1',
             ROAD_SWEEP_VIDEO='0',ROAMING_GHOST_TWO_PANEL_VIDEO='0',GO2_FRONT_PINHOLE='0',
             MIXED_ROAMING_CONFIG=str(ROOT/row['mixed_config']),REFERENCE_ROUTE=str(route),
             TRAFFIC_INITIAL_FILL='1',URBANVERSE_CAPTURE_ONLY_RENDER='1',
             LOOKAHEAD_DISTANCE='0.4',MINIMUM_TRACKING_SPEED='0.16',MAX_FORWARD_SPEED='0.35',
             MAX_TRACKING_YAW_RATE='0.8',CURVATURE_SPEED_GAIN='1',MAXIMUM_CROSS_TRACK_ERROR='0.6')
    env.update(row['runtime_environment'])
    camera=profile!='smoke';formal=profile=='formal'
    env.update(GO2_THREE_CAMERA=str(int(camera)),OVERVIEW_VIDEO=str(int(camera)),
        THREE_CAMERA_WIDTH='1280' if formal else '480',THREE_CAMERA_HEIGHT='720' if formal else '384',
        OVERVIEW_FPS='10' if formal else '5',STOP_AT_ROUTE_GOAL=str(int(formal)),
        DURATION_S=str(duration or (600 if formal else 10 if camera else 3)))
    label=f'{scene}_{profile}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    cmd=['bash',str(ROOT/'tools/dynamic_agents/runners/run_go2_traffic_probe.sh'),'route',str(gpu),label,str(ROOT/row['traffic_config'])]
    return cmd,env,label


def run_one(args,route):
    cmd,env,label=command(args.scene,args.profile,args.gpu,route,args.duration)
    if args.dry_run:
        print(json.dumps(dict(command=cmd,environment={k:v for k,v in env.items() if k not in os.environ or os.environ[k]!=v}),indent=2));return 0
    # The underlying runner records all-GPU inventories and rejects inconsistency.
    reserve=args.reserve_mib;estimated=5600 if args.profile=='smoke' else 12000
    def free():
        return int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
    if free()<estimated+reserve:raise RuntimeError(f'Insufficient current free VRAM for estimate {estimated}+reserve {reserve} MiB; no process launched')
    logdir=ROOT/'outputs/collection_batches'/label;logdir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();samples=[];reason=None
    with (logdir/'launch.log').open('x') as log:
        child=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                try:available=free();samples.append(dict(elapsed_s=time.monotonic()-started,free_mib=available))
                except Exception:reason='GPU monitor failed';break
                if available<reserve:reason='GPU reserve crossed';break
                if time.monotonic()-started>args.wall_timeout:reason='wall timeout';break
                time.sleep(2)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
            code=child.wait()
    text=(logdir/'launch.log').read_text(errors='replace')
    match=re.search(r'RUN_DIR=(\S+)',text);run=Path(match.group(1)) if match else None
    summary_path=run/'metadata/summary.json' if run else None
    summary=json.loads(summary_path.read_text()) if summary_path and summary_path.exists() else {}
    passed=code==0 and not reason and summary.get('status')=='passed'
    if args.profile=='formal':passed=passed and summary.get('goal_reached') is True
    contract=None
    if passed and args.profile!='smoke':
        from urbanverse.dynamic_agents.admission.capture_contract import audit_capture
        contract=audit_capture(run);passed=contract['status']!='failed'
    report=dict(status='passed' if passed else 'failed',run_dir=str(run) if run else None,
                process_exit_code=code,stop_reason=reason,profile=args.profile,route=str(route),
                gpu_samples=samples,capture_contract=contract,
                scope='Recorded runtime/file gates only; no pixel-level or cross-server acceptance inferred')
    (logdir/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='gpu_samples'},indent=2))
    return int(not passed)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('list');d=sub.add_parser('doctor');d.add_argument('--assets',action='store_true')
    for action in ('plan','run'):
        a=sub.add_parser(action);a.add_argument('--scene',choices=load_scenes(),required=True)
        a.add_argument('--seed',type=int,required=action=='plan')
        if action=='run':
            a.add_argument('--profile',choices=['smoke','preview','formal'],default='smoke')
            a.add_argument('--gpu',type=int,default=0);a.add_argument('--duration',type=float)
            a.add_argument('--wall-timeout',type=float,default=3600);a.add_argument('--reserve-mib',type=int,default=1536)
            a.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.action=='run' and (a.reserve_mib<=0 or a.wall_timeout<=0 or (a.duration is not None and a.duration<=0)):
        p.error('GPU reserve, timeout and duration must be positive')
    if a.action=='list':
        for k,v in load_scenes().items():print(k,v['population'],v['qualification'])
        return 0
    if a.action=='doctor':return doctor(a.assets)
    if a.action=='plan':print(plan(a.scene,a.seed));return 0
    route=plan(a.scene,a.seed) if a.seed is not None else ROOT/load_scenes()[a.scene]['reference_route']
    return run_one(a,route)


if __name__=='__main__':raise SystemExit(main())

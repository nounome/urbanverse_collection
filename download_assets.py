"""Explicit asset downloads; no application-level hash verification."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import json

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))


def craftbench():
    from huggingface_hub import hf_hub_download
    scenes=json.loads((ROOT/'scenes.json').read_text())
    for name,row in scenes.items():
        config=ROOT/row['traffic_config'];doc=json.loads(config.read_text())
        source=(config.parent/doc['source_usd']).resolve()
        archive=(config.parent/doc['source_tar']).resolve()
        if source.is_file():print(name,'already extracted');continue
        target=source.parent.parent
        if target.exists():raise RuntimeError(f'Incomplete existing extraction: {target}; inspect before replacing')
        key=str(archive.relative_to(ROOT/'data/urbanverse_craftbench/raw'))
        downloaded=Path(hf_hub_download('Oatmealliu/UrbanVerse-CraftBench',key,repo_type='dataset',local_dir=ROOT/'data/urbanverse_craftbench/raw'))
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='extract_',dir=target.parent) as tmp:
            with tarfile.open(downloaded) as tar:
                for member in tar.getmembers():
                    path=Path(member.name)
                    if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                        raise ValueError('Unsupported archive member: '+member.name)
                tar.extractall(tmp)
            expected=Path(tmp)/'Collected_export_version/export_version.usd'
            if not expected.is_file():raise RuntimeError('Unexpected archive layout: '+str(downloaded))
            target.mkdir()
            for child in Path(tmp).iterdir():shutil.move(str(child),str(target/child.name))
        print(name,'extracted')


def policy():
    commit='376d42c9b128f963ab08579762d5a216a976ce39'
    target=ROOT/'data/locomotion_policies/rl_sar'/commit
    for relative in ('policy/go2/base.yaml','policy/go2/robot_lab/config.yaml','policy/go2/robot_lab/policy.pt','LICENSE'):
        p=target/relative
        if p.is_file() and p.stat().st_size:continue
        p.parent.mkdir(parents=True,exist_ok=True)
        partial=p.with_suffix(p.suffix+'.part')
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/fan-ziqi/rl_sar/{commit}/{relative}',timeout=120) as response,partial.open('wb') as output:
            shutil.copyfileobj(response,output)
        if not partial.stat().st_size:raise RuntimeError('Empty download: '+relative)
        partial.replace(p)
    manifest=target/'manifest.json'
    if not manifest.exists():
        manifest.write_text(json.dumps(dict(repository='https://github.com/fan-ziqi/rl_sar',commit=commit,
            selected_policy='robot_lab',hash_verification=False),indent=2))


def go2():
    from urbanverse.dynamic_agents.assets.download_isaac_people_assets import list_prefix,download
    prefix='Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/Go2/'
    rows=list_prefix(prefix)
    if not rows:raise RuntimeError('Go2 asset listing empty')
    destination=ROOT/'data/isaacsim_assets_4_5/Isaac/IsaacLab/Robots/Unitree/Go2'
    for row in rows:print(download(row,destination,prefix,3))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['craftbench','policy','people','go2','all']);a=p.parse_args()
    kinds=['craftbench','policy','people','go2'] if a.kind=='all' else [a.kind]
    for kind in kinds:
        if kind=='people':
            subprocess.run([sys.executable,'-m','urbanverse.dynamic_agents.assets.download_isaac_people_assets',
                '--destination',str(ROOT/'data/isaacsim_assets_4_5/Isaac/People'),
                '--run-dir',str(ROOT/'outputs/asset_downloads'/f'people_{datetime.now():%Y%m%d_%H%M%S}')],
                cwd=ROOT,env={**__import__('os').environ,'PYTHONPATH':str(ROOT/'tools')},check=True)
        else:globals()[kind]()


if __name__=='__main__':main()

import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
import collect
import batch_collect


def test_twelve_scenes_and_durable_maps():
    assert set(collect.load_scenes())=={f'scene{i:02d}' for i in range(1,13)}
    assert collect.doctor()==0


def test_profiles_keep_frozen_controls_and_camera_contract():
    scene=collect.load_scenes()['scene02']
    for profile,size,fps,cameras in [('smoke',('480','384'),'5','0'),('preview',('480','384'),'5','1'),('formal',('1280','720'),'10','1')]:
        cmd,env,_=collect.command('scene02',profile,0,ROOT/scene['reference_route'],None)
        assert (env['THREE_CAMERA_WIDTH'],env['THREE_CAMERA_HEIGHT'])==size
        assert env['OVERVIEW_FPS']==fps and env['GO2_THREE_CAMERA']==cameras
        assert env['POLICY_KIND']=='robot_lab'
        for k,v in scene['runtime_environment'].items():assert env[k]==v
        assert 'urbanverse_imagegoal' not in ' '.join(cmd)


def test_owner_requested_no_policy_hash_verification(tmp_path):
    from urbanverse.dynamic_agents.navigation.policy_selection import checked
    p=tmp_path/'weight.pt';p.write_bytes(b'fixture')
    assert checked(p,'deliberately_different_digest')==p


def test_batch_retries_and_resumes_without_gpu(monkeypatch,tmp_path):
    journal=tmp_path/'journal.json';calls=[]
    def fake_run(*args,**kwargs):
        calls.append(args);return SimpleNamespace(returncode=1 if len(calls)==1 else 0)
    monkeypatch.setattr(batch_collect.subprocess,'run',fake_run)
    monkeypatch.setattr(sys,'argv',['batch_collect.py','--scenes','scene02','--journal',str(journal)])
    assert batch_collect.main()==0 and len(calls)==2
    assert batch_collect.main()==0 and len(calls)==2
    data=json.loads(journal.read_text());assert data['tasks']['scene02/0']['status']=='passed'


def test_batch_dry_run_does_not_write_or_launch(monkeypatch,tmp_path):
    journal=tmp_path/'journal.json'
    monkeypatch.setattr(batch_collect.subprocess,'run',lambda *a,**k: (_ for _ in ()).throw(AssertionError('launched')))
    monkeypatch.setattr(sys,'argv',['batch_collect.py','--scenes','scene02','--journal',str(journal),'--dry-run'])
    assert batch_collect.main()==0
    assert not journal.exists()

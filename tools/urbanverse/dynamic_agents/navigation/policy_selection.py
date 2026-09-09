"""Resolve frozen policy artifacts without depending on disposable run outputs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_POLICY_KIND = 'robot_lab'
EXTERNAL_KINDS = ('robot_lab', 'himloco')
EXTERNAL_SOURCE = ROOT / 'data/locomotion_policies/rl_sar/376d42c9b128f963ab08579762d5a216a976ce39'


def checked(path, digest=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Frozen policy artifact is missing: {path}')
    # Standalone delivery: existence checks only, as requested by the owner.
    return path


def resolve_policy(kind=DEFAULT_POLICY_KIND, profile='flat', checkpoint=None, policy=None, source=EXTERNAL_SOURCE):
    if kind in EXTERNAL_KINDS:
        if profile != 'flat':
            raise ValueError('robot_lab/HIMLoco use the flat Go2 task; select torchscript for the legacy rough policy')
        source = Path(source).resolve()
        manifest = json.loads((ROOT/'configs/dynamic_agents/catalogs/go2_external_policy_sources.json').read_text())
        weight = f'policy/go2/{kind}/' + ('policy.pt' if kind == 'robot_lab' else 'himloco.pt')
        for relative in ['policy/go2/base.yaml', f'policy/go2/{kind}/config.yaml', weight]:
            checked(source/relative, manifest['files'][relative]['sha256'])
        return checked(source/weight), checked(source/weight)
    if kind == 'go2z1_v2_checkpoint':
        if not checkpoint: raise ValueError('go2z1_v2_checkpoint requires GO2Z1_CHECKPOINT')
        return checked(checkpoint), checked(checkpoint)
    if kind != 'torchscript': raise ValueError(f'Unknown policy kind: {kind}')
    if checkpoint or policy:
        if not checkpoint or not policy: raise ValueError('Set both GO2_CHECKPOINT and GO2_POLICY for custom legacy artifacts')
        return checked(checkpoint), checked(policy)
    if profile != 'flat':
        raise ValueError('Legacy rough policy requires explicit GO2_CHECKPOINT and GO2_POLICY paths outside outputs/')
    directory = ROOT/'configs/dynamic_agents/catalogs/go2_flat_frozen_policy'
    manifest = json.loads((directory/'manifest.json').read_text())
    return (checked(ROOT/manifest['checkpoint_cache_path'], manifest['official_checkpoint_sha256']),
            checked(directory/manifest['policy_path'], manifest['policy_sha256']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', default=DEFAULT_POLICY_KIND)
    p.add_argument('--profile', default='flat', choices=('flat','rough'))
    p.add_argument('--source', type=Path, default=EXTERNAL_SOURCE)
    p.add_argument('--checkpoint');p.add_argument('--policy')
    a=p.parse_args()
    checkpoint,policy=resolve_policy(a.kind,a.profile,a.checkpoint or None,a.policy or None,a.source)
    print(checkpoint);print(policy)


if __name__ == '__main__': main()

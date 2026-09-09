"""Relocate exported source/config paths after cloning to a different directory."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def configure():
    manifest=json.loads((ROOT/'export_manifest.json').read_text())
    marker=ROOT/'.collection_root'
    old=marker.read_text().strip() if marker.exists() else manifest['configured_root']
    if old!=str(ROOT):
        for name in manifest['files']:
            p=ROOT/name
            if p.suffix in ('.py','.sh','.json','.kit','.in','.yaml') and p.is_file():
                text=p.read_text();updated=text.replace(old,str(ROOT))
                if updated!=text:p.write_text(updated)
    marker.write_text(str(ROOT)+'\n')
    print('Configured collection root:',ROOT)


if __name__=='__main__':configure()

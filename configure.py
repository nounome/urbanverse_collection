"""Generate local Kit experiences without modifying versioned configuration."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def configure():
    for template in sorted((ROOT/'configs').glob('*.kit.in')):
        escaped_root = json.dumps(str(ROOT), ensure_ascii=False)[1:-1]
        content = template.read_text().replace('@COLLECTION_ROOT@', escaped_root)
        target = template.with_suffix('')
        if not target.exists() or target.read_text() != content:
            target.write_text(content)


if __name__=='__main__':configure()

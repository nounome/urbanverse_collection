"""Record actual process exit after Kit exits (including uncatchable aborts)."""
import argparse
import json
from pathlib import Path


def finalize(run_dir, exit_code):
    metadata = Path(run_dir) / 'metadata'
    summary_path = metadata / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    report = dict(process_exit_code=int(exit_code), reported_status=summary.get('status'),
                  process_status='completed' if exit_code == 0 else 'failed')
    with (metadata / 'process_exit.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    if exit_code != 0:
        # Only finalize the newly launched run, never a historical result.
        summary['status_before_process_exit'] = summary.get('status')
        summary['status'] = 'failed'
        summary['process_exit_code'] = int(exit_code)
        summary['process_failure_reason'] = 'nonzero exit after/in capture; see run_log.txt'
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('exit_code', type=int)
    args = parser.parse_args()
    finalize(args.run_dir, args.exit_code)
    summary=json.loads((args.run_dir/'metadata/summary.json').read_text())
    raise SystemExit(args.exit_code or (0 if summary.get('status') in ('passed','success','diagnostic_completed') else 2))

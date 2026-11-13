"""Execute pytest and store real stdout, exit code, JUnit and environment lock."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

project = Path(__file__).resolve().parents[1]
workspace = project.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',default='verification_001')
    args = parser.parse_args()
    docs = project/'docs'
    log,json_path,junit = [docs/f'{args.name}.{ext}' for ext in ('log','json','xml')]
    if any(p.exists() for p in (log,json_path,junit)):
        raise FileExistsError('Verification outputs are write-once; choose another --name')
    cmd = [sys.executable,'-m','pytest',str(project/'tests'),'-q',f'--junitxml={junit}']
    start = time.perf_counter()
    result = subprocess.run(cmd,cwd=workspace,capture_output=True,text=True,encoding='utf-8',errors='replace')
    log.write_text(result.stdout+result.stderr,encoding='utf-8')
    counts = {}
    if junit.exists():
        suite = ET.parse(junit).getroot()
        for attr in ('tests','failures','errors','skipped'):
            counts[attr] = sum(int(s.attrib.get(attr,0)) for s in suite.findall('testsuite'))
    packages = {d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']!='bayesian-ring-ssr'}
    report = dict(command=cmd,cwd=str(workspace),exitcode=result.returncode,elapsed_seconds=time.perf_counter()-start,
                  **counts,python=sys.version,packages=packages)
    json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lock = project/'requirements-lock-windows-py311.txt'
    if not lock.exists():
        lock.write_text('# Actual validated Windows/Python3.11 environment; package itself installed separately with -e.\n'
                        +'\n'.join(f'{k}=={v}' for k,v in sorted(packages.items(),key=lambda x:x[0].lower()))+'\n',encoding='utf-8')
    print(result.stdout+result.stderr)
    print(json.dumps(dict(exitcode=result.returncode,**counts,report=str(json_path)),indent=2))
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())

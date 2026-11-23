"""Fetch two immutable Menagerie model folders, retaining licenses and file hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REPO='google-deepmind/mujoco_menagerie'
FOLDERS=('agilex_piper','franka_emika_panda')


def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'EASE-research-model-fetch'})
    with urllib.request.urlopen(request,timeout=90) as response:return response.read()


def run(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    commit=json.loads(get(f'https://api.github.com/repos/{REPO}/commits/main'))['sha']
    (root/'FETCH.json').write_text(json.dumps(dict(repository=REPO,commit=commit,folders=FOLDERS),indent=2))
    tree=json.loads(get(f'https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1'))
    assert not tree.get('truncated')
    entries=[e for e in tree['tree'] if e['type']=='blob' and any(e['path'].startswith(f+'/') for f in FOLDERS)
        and (not e['path'].lower().endswith('.png') or '/assets/' in e['path'])]
    def fetch(entry):
        path=entry['path'];assert '..' not in Path(path).parts
        data=get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path}')
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert blob==entry['sha'],path
        dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        return dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),git_blob=blob)
    with ThreadPoolExecutor(max_workers=8) as pool:files=list(pool.map(fetch,entries))
    manifest=dict(repository=f'https://github.com/{REPO}',commit=commit,files=files,
        scope='Unmodified upstream robot models/assets/licenses; source PNG previews omitted. No upstream code executed.')
    (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(commit=commit,files=len(files),bytes=sum(e['bytes'] for e in files)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(**vars(p.parse_args()))

"""Package the observed study, without bundling private runtimes or old results."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import platform

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'data/result/publication_20260920'
DEST=RESULT/'07_delivery'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()


def pack(name,files):
    path=DEST/name;manifest=[]
    files=sorted(set(p for p in files if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'))
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in files:
            relative=file.relative_to(ROOT).as_posix()
            archive.write(file,relative)
            manifest.append(dict(path=relative,bytes=file.stat().st_size,sha256=digest(file)))
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(path) as archive:assert archive.testzip() is None
    return dict(file=name,bytes=path.stat().st_size,files=len(files),sha256=digest(path),crc_check='passed')


def build():
    DEST.mkdir(exist_ok=True);env=DEST/'environment';env.mkdir(exist_ok=True)
    for name,python in [('jumpniche','C:/Anaconda/envs/jumpniche/python.exe'),('jnr-trees-clean','C:/Anaconda/envs/jnr-trees-clean/python.exe')]:
        value=subprocess.check_output([python,'-m','pip','freeze'],text=True,encoding='utf-8')
        (env/(name+'-windows-pip-freeze.txt')).write_text(value,encoding='utf-8')
        values=json.loads(subprocess.check_output([python,'-c',"import sys,platform,importlib.metadata,json; print(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{x.metadata['Name']:x.version for x in importlib.metadata.distributions()}}))"],text=True,encoding='utf-8'))
        (env/(name+'-versions.json')).write_text(json.dumps(values,indent=2),encoding='utf-8')
        packages={k.lower():v for k,v in values['packages'].items()}
        required=['numpy','pandas','scipy','pyarrow','psutil','openpyxl','scikit-learn']
        required+=['arboreto','dask','distributed'] if name=='jnr-trees-clean' else ['h5py','anndata','matplotlib','threadpoolctl','pytest']
        (env/(name+'-portable-requirements.txt')).write_text('\n'.join(f'{p}=={packages[p]}' for p in required)+'\n',encoding='utf-8')
    source=[]
    for folder in ('src','configs','docs','tests'):
        source.extend(p for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.R','.md','.txt','.yml','.yaml','.json'))
    source.extend(ROOT/p for p in ('main_jumpniche.py','preprocess_jumpniche.py','requirements.txt','README.md'))
    source.extend(env.glob('*'))
    # Include verified prepared values for personal cross-machine replication.
    inputs=list((ROOT/'data/processed/publication_20260920').rglob('*'))
    inputs.extend((ROOT/'data/raw/publication_20260920/pathways').glob('*'))
    results=[]
    for folder in ('01_data','02_runs','03_evaluation','04_cases'):
        results.extend(p for p in (RESULT/folder).rglob('*') if p.is_file() and not any('checkpoint' in x for x in p.parts))
    results.extend(RESULT.glob('*.json'))
    results.extend((RESULT/'06_audit').glob('*.json'))
    results.extend(p for p in (ROOT/'data/raw/publication_20260920').rglob('*.source.json') if 'render_dependencies' not in p.parts)
    results.append(ROOT/'docs/PUBLICATION_DELIVERY.md')
    records=[pack('source_code.zip',source),pack('prepared_inputs.zip',inputs),pack('study_results.zip',results)]
    (DEST/'package_checksums.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))
    return records


if __name__=='__main__':build()

"""Resource limits, immutable downloads and publication provenance."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import urllib.request
from datetime import datetime, timezone

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)

def file_hash(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()

def storage_guard(cfg, reserve=0):
    roots = (cfg.raw_root, cfg.processed_root, cfg.result_root)
    used = 0
    for root in roots:
        if not root.exists():continue
        for directory,_,names in __import__('os').walk(root):
            for name in names:
                # Checkpoints are atomically renamed while fitting. A vanished
                # file is not evidence of exhausted disk or an invalid model.
                try:used += (Path(directory)/name).stat().st_size
                except FileNotFoundError:continue
    if used + reserve > cfg.maximum_new_gib * 1024**3:
        raise RuntimeError('resource_limit: study storage budget exceeded')
    if shutil.disk_usage(cfg.raw_root.anchor).free - reserve < cfg.minimum_free_disk_gib * 1024**3:
        raise RuntimeError('resource_limit: insufficient free disk')
    return used

def download(cfg, url, destination, source_page):
    destination = Path(destination)
    marker = destination.with_suffix(destination.suffix + '.source.json')
    if destination.exists() and marker.exists():
        saved = json.loads(marker.read_text(encoding='utf-8'))
        if saved['url'] != url or saved['sha256'] != file_hash(destination):
            raise ValueError(f'Changed download cache: {destination}')
        return destination
    if destination.exists():
        raise FileExistsError(f'Unverified file exists; preserve and inspect: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent':'JumpNicheRewire-public-research/1.0'})
            with urllib.request.urlopen(request, timeout=45) as response:
                size = int(response.headers.get('Content-Length', 0))
                storage_guard(cfg, size)
                total = 0
                with temporary.open('wb') as output:
                    while block := response.read(1024*1024):
                        output.write(block); total += len(block)
                        if total % (8*1024*1024) == 0:
                            storage_guard(cfg)
                if size and total != size:
                    raise IOError(f'Download length mismatch {total}/{size}')
            temporary.replace(destination)
            write_json(marker, dict(url=url, source_page=source_page, bytes=total,
                sha256=file_hash(destination), retrieved_utc=datetime.now(timezone.utc).isoformat()))
            print(f'[download verified] {destination.name}: {total} bytes', flush=True)
            return destination
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

PUBLIC_MD5 = {
 'main_tree1.zip':'f7bc45378f3c3d54b3e818b8fc8edaa3',
 'main_tree3.zip':'319780bc7e8e428199e5d149443016fd',
 'main_tree5.zip':'3065b58dadb97b21ecdd953c716baf9b',
 'spatial_5000cells_500genes.zip':'5ee5acd69545c1a30fbf560643166d02',
 'spatial_3000cells_2000genes.zip':'1430e905e8309e58a5a76ed3923bd4de',
}

CASE_SOURCES = {
 'mouse_brain':('Visium_FFPE_Mouse_Brain','adult-mouse-brain-ffpe-1-standard-1-3-0','mouse'),
 'mouse_kidney':('Visium_FFPE_Mouse_Kidney','adult-mouse-kidney-ffpe-1-standard-1-3-0','mouse'),
 'human_breast_cancer':('Visium_FFPE_Human_Breast_Cancer','human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0','human'),
}

def verify_model_inputs(row):
    """Verify only model-input files; never open a truth label in inference."""
    root=Path(row['dataset_dir'])
    manifest=json.loads((root/'evidence_manifest.json').read_text(encoding='utf-8'))
    for name in ('expression.csv','coordinates.csv','metadata.csv','tf_list.csv','lr_pairs.csv','prior_tf_gene.csv','prior_lr_tf.csv'):
        if file_hash(root/name)!=manifest['files'][name]:
            raise ValueError(f'Changed prepared model input: {root/name}')

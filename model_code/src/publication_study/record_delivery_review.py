"""Record the completed 2026-09-21 review; not an automatic visual QA engine.

The page list below documents the actual human-visible tool inspection in this
task. Future edited manuscripts require a new inspection, not blind reuse.
"""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'data/result/publication_20260920'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def record():
    manuscript=RESULT/'05_manuscript/JumpNicheRewire_public_data_study.docx'
    render=RESULT/'06_audit/word_render_final'
    pages=[render/f'page-{i}.png' for i in range(1,16)]
    assert all(p.is_file() for p in pages)
    numerical=json.loads((RESULT/'06_audit/independent_numerical_audit.json').read_text(encoding='utf-8'))
    assert numerical['status']=='passed' and numerical['formal_runs']==81
    with ZipFile(manuscript) as doc:
        assert b'w:pBdr' not in doc.read('word/styles.xml')
    qa=dict(reviewed_utc=datetime.now(timezone.utc).isoformat(),manuscript_sha256=sha(manuscript),
        renderer='packaged render_docx.py with task-local SHA256-verified LibreOffice 26.2.6.3',
        pages=[dict(page=i+1,sha256=sha(p),visual_inspection='passed') for i,p in enumerate(pages)],
        corrected=['Removed inherited blue title border','Prevented split table rows and orphan table headers','Wrapped complete pathway names'],
        scientific_figures_reviewed=15,
        review_scope='Model-visible inspection of every rendered page and 12 metric plus 3 case figures; not external peer review',
        document_claims_reviewed=['No all-metric superiority','No causal validation','One distinct significant kidney pathway, not three','TF coverage 2/3/2','One missing full-run GENIE3 peak RSS'],
        regression_tests_passed=7)
    (RESULT/'06_audit/delivery_review.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    audit_path=RESULT/'scientific_audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    audit.update(numerical_review='passed',word_render_review='15 pages inspected, passed',core_delivery_ready=True,
        ready_for_publication=False,all_user_scientific_goals_achieved=False,
        remaining_scientific_limits=['all-metric superiority not established','no independent perturbation evaluation',
            'no spatial baseline comparison or ablation','limited real-case TF panels','no independent causal pathway validation'],
        publication_gate='Core-scope factual report delivered after numerical and visual audit. Strong method claims require additional independent evidence and external peer review.')
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':record()

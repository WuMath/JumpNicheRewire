"""Locked evaluation and descriptive pathway analyses. No fitting/tuning here."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from scipy.stats import hypergeom,wilcoxon
from .resources import file_hash,write_json

def bh(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if not len(p):return p
    order=np.argsort(p);ordered=p[order]
    q=np.minimum.accumulate((ordered*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty_like(q);result[order]=np.minimum(q,1)
    return result

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if not len(p):return p
    order=np.argsort(p)
    values=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    result=np.empty_like(values);result[order]=np.minimum(values,1)
    return result

def accepted_prediction(root,method):
    path=root/'run_status.json'
    if not path.exists():return None,'missing'
    state=json.loads(path.read_text(encoding='utf-8'))
    if state.get('state')!='completed':return None,state.get('state','invalid')
    if method=='JumpNicheRewire' and state.get('convergence_passed') is not True:
        return None,'convergence_failed'
    prediction=root/'global_grn.parquet'
    if not prediction.exists() or file_hash(prediction)!=state.get('prediction_sha256'):
        return None,'prediction_checksum_failed'
    frame=pd.read_parquet(prediction)
    frame['tf']=frame.tf.astype(str);frame['target']=frame.target.astype(str)
    if frame.duplicated(['tf','target']).any() or not np.isfinite(frame.score).all() or (frame.score<0).any():
        return None,'invalid_prediction'
    return frame,'accepted'

def benchmark(cfg):
    from independent_baselines.comparison import metrics
    from independent_baselines.metric_references import catalog,bibliography
    out=cfg.result_root/'03_evaluation';out.mkdir(parents=True,exist_ok=True)
    registry=pd.read_csv(cfg.processed_root/'benchmark_registry.csv')
    frozen=cfg.result_root/'02_runs/frozen_parameters.json'
    # Test metrics can only be unblinded after the frozen file exists.
    eligible=registry if frozen.exists() else registry.loc[registry.split.eq('validation')]
    audit=[];values=[]
    for row in eligible.itertuples():
        manifest=json.loads((Path(row.dataset_dir)/'evidence_manifest.json').read_text(encoding='utf-8'))
        if file_hash(Path(row.truth_dir)/'truth_grn.csv')!=manifest['files']['truth/truth_grn.csv']:
            raise ValueError(f'Changed evaluation labels: {row.dataset_id}')
        truth=pd.read_csv(Path(row.truth_dir)/'truth_grn.csv',dtype={'tf':str,'target':str})
        for method in cfg.methods:
            root=cfg.result_root/'02_runs'/row.dataset_id/method
            frame,state=accepted_prediction(root,method)
            record=dict(dataset_id=row.dataset_id,split=row.split,method=method,state=state,
                        n_candidates=len(truth),n_positive=int(truth.label.sum()))
            if frame is None:audit.append(record);continue
            joined=truth.merge(frame[['tf','target','score']],on=['tf','target'],how='left',validate='one_to_one')
            record.update(n_predicted_candidates=int(joined.score.notna().sum()),n_tfs=int(frame.tf.nunique()))
            scores=joined.score.fillna(0).to_numpy()
            audit.append(record)
            for name,value in metrics(joined.label.to_numpy(),scores,cfg.classification_top_k).items():
                values.append(dict(dataset_id=row.dataset_id,split=row.split,stratum=row.stratum,
                    independence_group=row.independence_group,method=method,metric=name,value=value))
    pd.DataFrame(audit).to_csv(out/'run_audit.csv',index=False)
    data=pd.DataFrame(values)
    data.to_csv(out/'dataset_metrics.csv',index=False)
    if data.empty:return dict(benchmark_complete=False,reason='no accepted runs')
    summary=data.groupby(['split','stratum','method','metric']).value.agg(['mean','std','count']).reset_index()
    summary.to_csv(out/'summary_by_stratum.csv',index=False)
    definitions=catalog(data.metric.unique(),SimpleNamespace(classification_top_k=100))
    definitions.to_csv(out/'metric_definitions.csv',index=False)
    bibliography().to_csv(out/'metric_references.csv',index=False)
    tables=out/'per_metric';tables.mkdir(exist_ok=True)
    for metric,group in data.groupby('metric'):
        group.pivot(index=['split','stratum','dataset_id'],columns='method',values='value').reset_index().to_csv(tables/(metric+'.csv'),index=False)
    # Report noisy observed-count test sets separately from latent-count large spatial examples.
    test=data.loc[data.split.eq('test') & ~data.stratum.eq('large_spatial_latent_counts')]
    paired=[];random=np.random.default_rng(cfg.seed)
    performance=definitions.loc[definitions.kind.eq('performance'),'metric'].tolist()
    for metric in performance:
        current=test.loc[test.metric.eq(metric)]
        wide=current.pivot(index=['independence_group','dataset_id'],columns='method',values='value')
        for other in cfg.methods[1:]:
            if 'JumpNicheRewire' not in wide or other not in wide:continue
            both=wide[['JumpNicheRewire',other]].dropna()
            # Avoid treating correlated cells or shared-condition replicates as independent samples.
            differences=(both.JumpNicheRewire-both[other]).groupby(level='independence_group').mean().to_numpy()
            n=len(differences)
            if n<2:continue
            sample=random.choice(differences,size=(cfg.bootstrap_repetitions,n),replace=True).mean(axis=1)
            p=1.0 if np.all(differences==0) else float(wilcoxon(differences,alternative='two-sided',method='auto').pvalue)
            paired.append(dict(metric=metric,comparator=other,n_condition_groups=n,n_datasets=len(both),
                mean_paired_difference=float(differences.mean()),ci_low=float(np.quantile(sample,.025)),
                ci_high=float(np.quantile(sample,.975)),wilcoxon_p=p,
                mean_strictly_better=bool(differences.mean()>0)))
    paired=pd.DataFrame(paired)
    if not paired.empty:paired['holm_p']=holm(paired.wilcoxon_p)
    paired.to_csv(out/'paired_statistics.csv',index=False)
    all_test=registry.loc[registry.split.eq('test')]
    expected=len(all_test)*len(cfg.methods)
    a=pd.DataFrame(audit)
    accepted=int((a.split.eq('test')&a.state.eq('accepted')).sum())
    enough=not paired.empty and len(paired)==len(performance)*(len(cfg.methods)-1) and paired.n_datasets.eq(18).all()
    conclusions=dict(benchmark_complete=accepted==expected,expected_test_runs=expected,accepted_test_runs=accepted,
        all_metrics_strictly_better=bool(paired.mean_strictly_better.all()) if enough else None,
        all_metrics_ci_above_zero=bool(paired.ci_low.gt(0).all()) if enough else None,
        claims_apply_to='prespecified noisy observed-count 500-cell 110-gene published subset',
        larger_latent_count_datasets='reported separately; no measurement noise was added',
        causal_claim_supported=False)
    write_json(out/'claim_audit.json',conclusions)
    plot_metrics(cfg,data,performance)
    return conclusions

def plot_metrics(cfg,data,names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=cfg.result_root/'03_evaluation/figures';out.mkdir(parents=True,exist_ok=True)
    test=data.loc[data.split.eq('test') & ~data.stratum.eq('large_spatial_latent_counts')]
    if test.empty:return
    for metric in names:
        fig,ax=plt.subplots(figsize=(6,4))
        current=test[test.metric.eq(metric)]
        positions={m:i for i,m in enumerate(cfg.methods)}
        for _,group in current.groupby('dataset_id'):
            group=group.set_index('method').reindex(cfg.methods)
            ax.plot(range(len(cfg.methods)),group.value,color='#c0c0c0',alpha=.5,lw=.7)
        for m,i in positions.items():
            vals=current.loc[current.method.eq(m),'value']
            ax.scatter(np.full(len(vals),i),vals,s=18,label=m,zorder=3)
            if len(vals):ax.scatter([i],[vals.mean()],marker='_',s=450,color='black',zorder=4)
        ax.set_xticks(range(len(cfg.methods)),cfg.methods)
        ax.set_ylabel(metric);ax.set_ylim(0,1);ax.set_title('Published scMultiSim held-out datasets')
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout();fig.savefig(out/(metric+'.png'),dpi=180);plt.close(fig)

def read_gmt(path):
    result={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        parts=line.split('\t')
        if len(parts)>2:result[parts[0]]=set(g.split(',')[0] for g in parts[2:] if g)
    if not result:raise ValueError('No pathway terms in downloaded library')
    return result

def enrich(gene_sets,background,library,cfg):
    background=set(background);records=[]
    # Enrichr's species-specific mouse library uses uppercase symbols. This is
    # case normalization within the same species, NOT mouse-to-human orthology.
    original={g.upper():g for g in background}
    if len(original)!=len(background):raise ValueError('Ambiguous case-normalized gene symbols')
    background=set(original)
    for group,genes in gene_sets.items():
        query={g.upper() for g in genes}&background
        if not query:continue
        for term,full in library.items():
            annotated={g.upper() for g in full} & background
            if not cfg.minimum_pathway_size<=len(annotated)<=cfg.maximum_pathway_size:continue
            hit=query&annotated
            # Include zero-overlap tests in the FDR family.
            p=float(hypergeom.sf(len(hit)-1,len(background),len(annotated),len(query)))
            records.append(dict(gene_set=group,pathway=term,overlap=len(hit),query_size=len(query),
                pathway_in_background=len(annotated),background_size=len(background),p_value=p,
                overlap_genes=';'.join(sorted(original[g] for g in hit)),query_genes=';'.join(sorted(original[g] for g in query))))
    result=pd.DataFrame(records,columns=['gene_set','pathway','overlap','query_size','pathway_in_background',
        'background_size','p_value','overlap_genes','query_genes'])
    result['q_value']=bh(result.p_value.to_numpy())
    result['significant']=result.q_value.le(cfg.pathway_fdr)
    return result.sort_values(['q_value','p_value','gene_set','pathway'])

def cases(cfg):
    out=cfg.result_root/'04_cases';out.mkdir(exist_ok=True,parents=True)
    records=[]
    registry=pd.read_csv(cfg.processed_root/'case_registry.csv')
    for row in registry.itertuples():
        run=cfg.result_root/'02_runs'/row.dataset_id/'JumpNicheRewire'
        grn,state=accepted_prediction(run,'JumpNicheRewire')
        if grn is None:
            records.append(dict(dataset_id=row.dataset_id,state=state));continue
        dest=out/row.dataset_id;dest.mkdir(exist_ok=True)
        background=sorted(grn.target.unique())
        ranked=grn.groupby('target').score.max().sort_values(ascending=False)
        # Explicit deterministic gene-name secondary sort for pathway candidate lists.
        def top(frame,score='score'):
            ordered=frame.sort_values([score,'target'],ascending=[False,True])
            return ordered.loc[ordered[score].gt(0),'target'].drop_duplicates().head(cfg.pathway_top_targets).tolist()
        gene_sets={'global_GRN':top(ranked.reset_index())}
        for tf,frame in grn.groupby('tf'):gene_sets['TF_'+tf]=top(frame)
        lr_gene=run/'parquet/lr_gene.parquet'
        if lr_gene.exists():
            frame=pd.read_parquet(lr_gene)
            if not frame.empty:
                for lr,group in frame.groupby('lr_id'):
                    # LR-gene output is a documented top-k mechanism exploration, not full benchmark scores.
                    group=group.groupby('target',as_index=False).score.max()
                    gene_sets['LR_'+str(lr)]=top(group)
        species='Human' if row.species=='human' else 'Mouse'
        library_path=cfg.raw_root/'pathways'/('WikiPathways_2024_'+species+'.gmt')
        enrichment=enrich(gene_sets,background,read_gmt(library_path),cfg)
        enrichment.to_csv(dest/'pathway_enrichment_all_tests.csv',index=False)
        enrichment[enrichment.significant].to_csv(dest/'pathway_enrichment_significant.csv',index=False)
        pd.DataFrame({'gene':background}).to_csv(dest/'enrichment_background.csv',index=False)
        grn.sort_values(['score','tf','target'],ascending=[False,True,True]).to_csv(dest/'global_GRN_ranked.csv',index=False)
        paths=run/'parquet/receptor_paths.parquet'
        if paths.exists():pd.read_parquet(paths).to_csv(dest/'mechanism_paths_preview.csv',index=False)
        write_json(dest/'analysis_audit.json',dict(pathway_library=str(library_path),library_sha256=file_hash(library_path),
            background='all actually fitted target genes',tests=len(enrichment),fdr_family='all gene sets and pathways within this case',
            top_k=cfg.pathway_top_targets,biological_replicates=1,causal_inference=False,
            interpretation='prior-informed exploratory association, not independently validated pathway activation'))
        case_figure(cfg,row,run,dest,enrichment)
        records.append(dict(dataset_id=row.dataset_id,state='analyzed',spots=row.cells,
            fitted_tfs=grn.tf.nunique(),fitted_targets=len(background),pathway_tests=len(enrichment),
            significant_tests=int(enrichment.significant.sum()),top_pathway=enrichment.pathway.iloc[0] if len(enrichment) else None,
            best_q=float(enrichment.q_value.min()) if len(enrichment) else None))
    pd.DataFrame(records).to_csv(out/'case_summary.csv',index=False)
    return records

def case_figure(cfg,row,run,dest,enrichment):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    coords=pd.read_csv(Path(row.dataset_dir)/'coordinates.csv',index_col=0)
    activity=pd.read_parquet(run/'parquet/regulon_activity.parquet')
    if activity.empty:return
    tf=activity.groupby('tf').activity.var().sort_values(ascending=False).index[0]
    current=activity[activity.tf.eq(tf)].set_index('cell_id').reindex(coords.index)
    import textwrap
    fig,axes=plt.subplots(2,1,figsize=(7,8.2),gridspec_kw={'height_ratios':[1.15,1]},layout='constrained')
    dots=axes[0].scatter(coords.x,coords.y,c=current.activity,s=6,cmap='viridis')
    axes[0].invert_yaxis();axes[0].set_aspect('equal');axes[0].set_title(f'{row.tissue.replace("_"," ")}\nInferred {tf} regulon activity',fontsize=11)
    axes[0].set_xlabel('image x');axes[0].set_ylabel('image y');fig.colorbar(dots,ax=axes[0],shrink=.7)
    terms=enrichment[enrichment.gene_set.eq('global_GRN')].head(8).iloc[::-1]
    axes[1].barh(range(len(terms)),-np.log10(terms.q_value.clip(lower=1e-300)),color='#4e79a7')
    labels=[textwrap.fill(str(x),44) for x in terms.pathway]
    axes[1].set_yticks(range(len(terms)),labels,fontsize=8)
    axes[1].axvline(-np.log10(.05),color='red',ls='--',lw=1)
    axes[1].set_xlabel('-log10 BH q');axes[1].set_title('Global GRN pathway enrichment',fontsize=11)
    if not terms.q_value.le(.05).any():axes[1].text(.5,.97,'No pathway passes BH q <= 0.05',transform=axes[1].transAxes,ha='center',va='top',fontsize=8)
    fig.savefig(dest/'case_overview.png',dpi=180);plt.close(fig)

def analyze(cfg):
    result=benchmark(cfg)
    case_results=cases(cfg)
    result['cases_analyzed']=sum(r['state']=='analyzed' for r in case_results)
    result['complete_experiments']=bool(result.get('benchmark_complete') and result['cases_analyzed']==len(cfg.cases))
    result['ready_for_publication']=False
    result['publication_gate']='Requires independent review of methods, numerical audits, all plots, source citations and Word rendering; no automatic superiority claim.'
    write_json(cfg.result_root/'scientific_audit.json',result)
    return result

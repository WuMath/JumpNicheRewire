"""Fail-closed task capabilities. A missing or unsupported task is not zero."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodCapability:
    method: str
    native_tasks: tuple[str, ...]
    explicit_reference_grn: bool
    caveat: str


METHODS = {
    'CeSpGRN': MethodCapability('CeSpGRN', ('tf_target',), False,
        'Cell-specific association graph; TF projection is not causal direction. No native LR-on/off pair.'),
    'CLARIFY': MethodCapability('CLARIFY', ('tf_target', 'cell_total_cci'), False,
        'Refined cell/gene graphs; initializer vs refinement is not an LR-only ablation.'),
    'SpaGRN': MethodCapability('SpaGRN', ('tf_target', 'receptor_tf_target'), False,
        'Full pipeline requires species-matched regulatory priors; receptor paths are not additive GRN coefficients.'),
    'iMIRACLE': MethodCapability('iMIRACLE', ('ligand_target', 'expression'), False,
        'Basic+LR expression components do not constitute basic+LR TF-target GRNs. Paper score reproduction pending.'),
    'JNR-v2': MethodCapability('JNR-v2', ('tf_target',), True,
        'Reference must be declared; expression-derived LR exposure is not independent activity or causal identification.'),
}


def require_task(method, task, *, truth_task, baseline_available=True):
    if method not in METHODS:
        raise ValueError('Unregistered method')
    if task not in METHODS[method].native_tasks:
        raise ValueError('unsupported_task: never substitute a zero score')
    if truth_task != task:
        raise ValueError('truth_task_mismatch')
    if not baseline_available:
        raise ValueError('missing_output: never substitute a zero score')
    return True


def clarify_dense_lower_bounds(cells, genes):
    if type(cells) is not int or type(genes) is not int or min(cells, genes) < 1:
        raise ValueError('Positive integer dimensions required')
    entries = (cells*genes)**2
    return {'nodes': cells*genes, 'single_float32_matrix_bytes': entries*4,
            'float64_block_diag_bytes': entries*8,
            'boolean_mask_bytes': entries,
            'is_peak_estimate': False}

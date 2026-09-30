"""Memory bounded equivalent of CLARIFY's intracellular MSE term.

The pinned CLARIFY training code uses a block diagonal mask over
sigmoid(z_g @ z_g.T), then PyTorch's default mean MSE against flattened
initial GRNs. This computes exactly those selected entries per cell.
It is an adapter primitive, not a complete CLARIFY implementation.
"""

def intracellular_mse(z_g, initial_grns, genes_per_cell, *, cell_block=8):
    import torch

    if z_g.ndim != 2 or initial_grns.ndim != 1:
        raise ValueError("Expected 2D embeddings and flattened 1D initial GRNs")
    if genes_per_cell < 1 or cell_block < 1 or z_g.shape[0] % genes_per_cell:
        raise ValueError("Invalid cell/gene dimensions")
    cells = z_g.shape[0] // genes_per_cell
    if initial_grns.numel() != cells * genes_per_cell * genes_per_cell:
        raise ValueError("Initial GRN shape differs from CLARIFY mask order")
    if initial_grns.device != z_g.device or initial_grns.dtype != z_g.dtype:
        raise ValueError("Initial GRN device/dtype differs from embedding")
    z = z_g.reshape(cells, genes_per_cell, z_g.shape[1])
    target = initial_grns.reshape(cells, genes_per_cell, genes_per_cell)
    total = z_g.new_zeros(())
    for first in range(0, cells, cell_block):
        last = min(first + cell_block, cells)
        block = z[first:last]
        reconstruction = torch.sigmoid(torch.bmm(block, block.transpose(1, 2)))
        total = total + torch.square(reconstruction - target[first:last]).sum()
    return total / initial_grns.numel()

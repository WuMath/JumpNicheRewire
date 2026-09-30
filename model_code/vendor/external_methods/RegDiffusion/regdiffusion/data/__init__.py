"""
This package provides some access to some sample data to run. Right now, you 
can use it to quickly access the BEELINE single cell benchmark and the two
microglia datasets we used in the RegDiffusion paper. 
"""

from .beeline import load_beeline, download_beeline, load_beeline_ground_truth
from .microglia import load_atlas_microglia, load_hammond_microglia



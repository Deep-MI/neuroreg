"""Command-line entry points for neuroreg tools.

Available commands
------------------
robreg       Robust image-to-image registration (analogous to mri_robust_register).
multireg     Multi-timepoint robust registration with initial mean-space construction and iterative refinement.
coreg        Image-based cross-modal registration (analogous to mri_coreg).
bbreg        Boundary-based registration using cortical surfaces (analogous to bbregister).
segreg       Segmentation-centroid registration that writes LTAs.
segcentroids Write centroid target JSON files from segmentations.
lta          LTA transform utilities (diff, invert, concat).
vol2vol      Apply linear transforms to images, reslice to reference geometry, or map headers.
"""

CLI reference
=============

The project exposes a small set of user-facing commands. The homepage README
contains the full usage walkthroughs; this page gives a quick command index and
points to the backing Python modules documented in :doc:`../api/cli`.

robreg
------

IRLS-backed robust image-to-image registration for same-contrast or
closely-related contrast pairs.

Example::

   robreg --mov moving.nii.gz --ref fixed.nii.gz --out moving_to_fixed.lta

coreg
-----

Image-based cross-modal registration when only images are available.

Example::

   coreg --mov moving.nii.gz --ref fixed.nii.gz --out moving_to_fixed.lta --dof 6

bbreg
-----

Boundary-based registration using cortical surfaces or a segmentation-derived
surface pair.

Example::

   bbreg --mov bold.nii.gz --subject_dir /path/to/subject --out bold_to_t1.lta

lta
---

Utilities for comparing, concatenating, and inverting FreeSurfer LTA files.

Examples::

   lta diff reg1.lta reg2.lta
   lta invert in.lta out.lta
   lta concat a_to_b.lta b_to_c.lta a_to_c.lta

neuroreg-sys_info
-----------------

Print package, dependency, and runtime information useful for debugging
installations and CI environments.

Example::

   neuroreg-sys_info --developer

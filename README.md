[![PyPI version](https://badge.fury.io/py/robreg.svg)](https://pypi.org/project/robreg/)
# robreg

Robreg is a tool for the robust and symmetric registration of 3D images, e.g. MRI. 
It makes use of Pytorch's build in gradient backpropagation and therefore can run
on a GPU very efficiently. 

This project is a work-in-progress in an early development stage. Updated
documentation, testing and functionality will follow. 

The current version can register two 3D images via a Gaussian pyramid 
with a specified degree-of-freedom: 

-  3 : Translation only
-  6 : Rigid (translation and rotation)
-  9 : Rigid with scaling
- 12 : Fully affine

## API Documentation

The API Documentation can be found at https://deep-mi.org/robreg .

## References:

If you use this software for a publication please cite both these papers:

- Reuter, Rosas, Fischl (2010). 
  Highly accurate inverse consistent registration: a robust approach. 
  NeuroImage 53(4):1181-1196
  [https://doi.org/10.1016/j.neuroimage.2012.02.084](https://doi.org/10.1016/j.neuroimage.2012.02.084)
- Reuter, Schmansky, Rosas, Fischl
  Within-subject template estimation for unbiased longitudinal image analysis.
  NeuroImage 61(4):1402-1418
  [https://doi.org/10.1016/j.neuroimage.2012.02.084](https://doi.org/10.1016/j.neuroimage.2012.02.084)
- Reuter, Fischl (2011).
  Avoiding asymmetry-induced bias in longitudinal image processing.
  NeuroImage 57(1):19-21
  [https://doi.org/10.1016/j.neuroimage.2011.02.076](https://doi.org/10.1016/j.neuroimage.2011.02.076)


We invite you to check out our lab webpage at https://deep-mi.org

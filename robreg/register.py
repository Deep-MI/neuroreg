import time
from typing import Optional, Union

import nibabel as nib
import torch
from torch import Tensor

from .image.pyramid import build_gaussian_pyramid
from .nn.reg_model import RegModel
from .nn.training import training_loop
from .transforms.initialize import get_ixform_centroids
from .transforms.lta import write_lta


def register(
        simg: Tensor,
        timg: Tensor,
        dof: int = 6,
        v2v_init: Optional[Tensor] = None,
        centroid_init: bool = True,
        n: int = 30,
        verbose: bool = False,
        device: str = 'cpu'
) -> tuple[Tensor, list[float], "RegModel"]:
    """
    Register (align) two images using a transformation optimization model (RegModel).

    This function performs an alignment of two images by optimizing a transformation
    model with a specified degree of freedom (dof). The alignment process can be initialized
    either by directly providing a voxel-to-voxel transformation matrix (`v2v_init`) or by
    automatically computing an initial translation based on centroid alignment
    (`centroid_init`). The optimization iterates through a training loop, minimizing the
    difference between the source image (`simg`) and target image (`timg`).

    Parameters
    ----------
    simg : Tensor
        The source image represented as a PyTorch tensor.
    timg : Tensor
        The target image represented as a PyTorch tensor.
    dof : int, optional
        The degree of freedom for the transformation model (default is 6).
        e.g., 6 represents rigid body alignment (translation + rotation).
    v2v_init : Tensor, optional
        Initial voxel-to-voxel transformation matrix, if provided.
        Overrides centroid initialization if both are supplied.
    centroid_init : bool, optional
        Whether to initialize the transformation based on centroid alignment
        (default is True). Ignored if `v2v_init` is provided.
    n : int, optional
        The number of optimization iterations in the training loop (default is 30).
    verbose : bool, optional
        If True, prints progress information including the loss during optimization.
    device : str, optional
        The device to run the optimization on, such as 'cpu' or 'cuda' (default is 'cpu').

    Returns
    -------
    tuple[Tensor, list[float], RegModel]
        A tuple containing:
        - v2v (Tensor): The resulting voxel-to-voxel transformation matrix after optimization.
        - losses (list): A list of loss values observed during the optimization process.
        - m (RegModel): The trained registration model instance containing the optimized parameters.

    Notes
    -----
    - If both `v2v_init` and `centroid_init` are specified, `v2v_init` takes precedence
      and a warning is printed.
    - This function utilizes a PyTorch optimization loop with RMSProp as the optimizer.
    """
    if v2v_init is not None and centroid_init:
        # cannot do both
        print("WARNING: register: cannot pass v2v_init and centroid_init=True, will use v2v_init")
        centroid_init = False
    if centroid_init:
        v2v_init = get_ixform_centroids(simg, timg)
        print("v2v_init from centroid alignment:", v2v_init)
    m = RegModel(dof=dof, v2v_init=v2v_init, source_shape=simg.shape, target_shape=timg.shape, device=device)
    opt = torch.optim.RMSprop(m.parameters(), lr=0.001)
    # when choosing these, update also parameter to stop early in training loop
    # opt = torch.optim.Adam(m.parameters(), lr=0.001)
    # opt = torch.optim.SGD(m.parameters(), lr=0.001) # does not seem to work
    # opt = torch.optim.LBFGS(m.parameters(), lr=0.001) # neither this one
    losses = training_loop(m, opt, simg.to(device), timg.to(device), n=n, verbose=verbose)
    v2v = m.get_v2v_from_weights(simg.shape, timg.shape)
    return v2v, losses, m


def register_pyramid(
    src: Union[str, nib.Nifti1Image],
    trg: Union[str, nib.Nifti1Image],
    lta_name: Optional[str] = None,
    mapped_name: Optional[str] = None,
    device: str = 'cpu'
) -> Tensor:
    """
    Perform multi-resolution image registration using an iterative coarse-to-fine alignment.

    This function aligns two 3D images (`src` and `trg`) by building Gaussian pyramids for the
    images and progressively registering lower-resolution versions of the images first. The
    resulting transformation is then refined for higher-resolution levels in the pyramid to
    compute a final voxel-to-voxel transformation matrix that aligns the source image to the
    target image.

    Parameters
    ----------
    src : Union[str, nibabel.Nifti1Image]
        The source image, provided either as a file path to a `.mgz` file (string) or as a nibabel image object.
        The image will be moved to align with the target.
    trg : Union[str, nibabel.Nifti1Image]
        The target image, provided either as a file path to a `.mgz` file (string) or as a nibabel image object.
        The source image will be aligned to this image.
    lta_name : Optional[str], optional
        The output filename for saving the final transformation matrix in LTA format (default is None).
    mapped_name : Optional[str], optional
        The output filename for saving the mapped version of the source image after registration (default is None).
    device : str, optional
        The device to use for computation, such as 'cpu' or 'cuda' (default is 'cpu').

    Returns
    -------
    torch.Tensor
        The final rigid-body transformation matrix (`Mr2r`) between the reference frames of the source
        and target images, representing their alignment in 3D space (RAS-to-RAS).

    Notes
    -----
    - If `src` and `trg` are provided as strings, they are loaded using `nibabel`.
    - The images must have the same voxel size for accurate registration.
    - The function prints periodic log messages indicating the progress at each resolution level within the pyramid.
    - This function assumes that the input images are preloaded or provided as filenames referencing valid `.mgz` files.

    Examples
    --------
    >>> register_pyramid("src.mgz", "trg.mgz", lta_name="alignment.lta", mapped_name="mapped_image.mgz", device="cuda")
    >>> src_image = nib.load("src.mgz")
    >>> trg_image = nib.load("trg.mgz")
    >>> register_pyramid(src_image, trg_image, device="cpu")
    """
    start = time.perf_counter()
    if isinstance(src, str):
        src = nib.load(src)
    if isinstance(trg, str):
        trg = nib.load(trg)
    sdata = torch.from_numpy(src.get_fdata()).float()
    tdata = torch.from_numpy(trg.get_fdata()).float()
    # voxel size should be the same !!!
    simgs, saffines = build_gaussian_pyramid(sdata, src.affine)
    timgs, taffines = build_gaussian_pyramid(tdata, trg.affine)

    Mr2r = torch.eye(4, 4, dtype=saffines[0].dtype)
    count = 0
    debug = False
    n = 10
    torch.set_printoptions(precision=8, sci_mode=False)
    for si, sa, ti, ta in zip(reversed(simgs), reversed(saffines), reversed(timgs), reversed(taffines)):
        print("\n===============================================")
        print("Resolution: ", count, si.size())
        if count == 0:
            Mv2v, losses, m = register(si, ti, centroid_init=True, n=n, device=device)
        else:
            Mv2v_init = torch.inverse(ta) @ Mr2r @ sa
            print("Mv2v_init", Mv2v_init)
            Mv2v, losses, m = register(si, ti, v2v_init=Mv2v_init, centroid_init=False, n=n, device=device)
        Mv2v = Mv2v.double()
        print("Mv2v", Mv2v)
        Mr2r = ta @ Mv2v @ torch.inverse(sa)
        print("Mr2r", Mr2r)
        if debug:
            sname = "pyramidS-rr" + str(count) + ".mgz"
            tname = "pyramidT-rr" + str(count) + ".mgz"
            ltaname = "pyramid_S2T_rr" + str(count) + ".lta"
            smgh = nib.MGHImage(si.squeeze().numpy(), sa.numpy(), src.header)
            tmgh = nib.MGHImage(ti.squeeze().numpy(), ta.numpy(), trg.header)
            smgh.to_filename(sname)
            tmgh.to_filename(tname)
            write_lta(ltaname, Mr2r.numpy(), sname, smgh.header, tname, tmgh.header)
        count = count + 1
    if lta_name is not None:
        print ('Writing final LTA file: ', lta_name, ' ...')
        write_lta(lta_name, Mr2r.numpy(), src.get_filename(), src.header, trg.get_filename(), trg.header)
    if mapped_name is not None:
        # the linear mapping here is not great and smoothes the images a lot.
        # cubic does not work in 3D so it is recommended to map the image
        # on the command line with other tools (like mri_convert from FreeSurfer)
        print('Writing mapped image: ', mapped_name, ' ...')
        mapped = m.map_image(sdata, mode='bilinear').detach()
        mapped_img = nib.MGHImage(mapped.squeeze().numpy(), src.affine, src.header)
        mapped_img.to_filename(mapped_name)
    print("Total time: ", time.perf_counter() - start)
    return Mr2r

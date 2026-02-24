import time

import nibabel as nib
import torch
from torch import Tensor
from typing import Optional, Literal
from pathlib import Path

from .image.pyramid import build_gaussian_pyramid
from .nn.reg_model import RegModel
from .nn.training import training_loop
from .transforms.initialize import get_ixform_centroids
from .transforms.lta import write_lta
from .transforms.headers import header_to_dict, ras_to_vox_transform
from .surface.io import load_surface, load_surface_pair, load_surface_from_subject, get_vox2ras_tkr
from .surface.optimize import BBRModel


def register(
        simg: Tensor,
        timg: Tensor,
        dof: int = 6,
        v2v_init: Tensor | None = None,
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
    src: str | nib.Nifti1Image,
    trg: str | nib.Nifti1Image,
    lta_name: str | None = None,
    mapped_name: str | None = None,
    return_v2v: bool = False,
    centroid_init: bool = True,
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
    return_v2v : bool, optional
        Return vox-to-vox transformation matrix after registration (default is False, returns ras-to-ras instead).
    centroid_init : bool, optional
        Whether to initialize the transformation based on centroid alignment (default is True).
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
    for si, sa, ti, ta in zip(reversed(simgs), reversed(saffines), reversed(timgs), reversed(taffines), strict=False):
        print("\n===============================================")
        print("Resolution: ", count, si.size())
        if count == 0:
            Mv2v, losses, m = register(si, ti, centroid_init=centroid_init, n=n, device=device)
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
    if return_v2v:
        return Mv2v
    return Mr2r


def register_surface(
    mov: str | nib.Nifti1Image,
    lh_surf: Optional[str] = None,
    rh_surf: Optional[str] = None,
    subject_dir: Optional[str] = None,
    lta_name: Optional[str] = None,
    dof: int = 6,
    contrast: Literal['t1', 't2'] = 't2',
    init_type: Literal['header', 'centroid', 'lta'] = 'header',
    init_lta: Optional[str] = None,
    cost_type: Literal['contrast', 'gradient', 'both'] = 'contrast',
    wm_proj_abs: float = 2.0,
    gm_proj_frac: float = 0.5,
    slope: float = 0.5,
    gradient_weight: float = 0.0,
    subsample: int = 1,
    n_iters: int = 100,
    lr: float = 0.01,
    verbose: bool = True,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Register moving image to target anatomy using cortical surface boundaries (BBR).

    This implements boundary-based registration similar to FreeSurfer's bbregister,
    where alignment is optimized by sampling intensities at cortical surface locations.

    Parameters
    ----------
    mov : str or nibabel.Nifti1Image
        Moving image to register (e.g., functional scan)
    lh_surf : str, optional
        Path to left hemisphere surface (e.g., lh.white)
    rh_surf : str, optional
        Path to right hemisphere surface (e.g., rh.white)
    subject_dir : str, optional
        FreeSurfer subject directory. If provided, surfaces are loaded from
        {subject_dir}/surf/lh.white and {subject_dir}/surf/rh.white
    lta_name : str, optional
        Output LTA filename to save transformation
    dof : int
        Degrees of freedom: 6 (rigid), 9 (rigid+scale), 12 (affine)
    contrast : str
        Expected tissue contrast: 't1' (WM>GM) or 't2' (GM>WM, default)
    init_type : str
        Initialization method: 'header', 'centroid', or 'lta'
    init_lta : str, optional
        Path to LTA file for initialization (if init_type='lta')
    cost_type : str
        Cost function: 'contrast' (BBR), 'gradient', or 'both'
    wm_proj_abs : float
        Distance (mm) to project into white matter (default: 2.0)
    gm_proj_frac : float
        Fraction of cortical thickness for GM projection (default: 0.5)
    slope : float
        BBR cost function slope (default: 0.5)
    gradient_weight : float
        Weight for gradient cost if cost_type='both' (default: 0.0)
    subsample : int
        Subsample surface vertices every N (default: 1, no subsampling)
    n_iters : int
        Number of optimization iterations (default: 100)
    lr : float
        Learning rate (default: 0.01)
    verbose : bool
        Print progress (default: True)
    device : str
        Device: 'cpu' or 'cuda'

    Returns
    -------
    torch.Tensor
        Final transformation matrix (4x4) in RAS-to-RAS space

    Examples
    --------
    # Register fMRI to anatomy using surfaces from subject directory
    >>> transform = register_surface(
    ...     mov='bold_mean.nii.gz',
    ...     subject_dir='/data/subjects/sub-01',
    ...     lta_name='bold2anat.lta',
    ...     contrast='t2',
    ...     device='cuda'
    ... )

    # Register with explicit surface paths
    >>> transform = register_surface(
    ...     mov='bold_mean.nii.gz',
    ...     lh_surf='surf/lh.white',
    ...     rh_surf='surf/rh.white',
    ...     lta_name='registration.lta'
    ... )
    """
    start = time.perf_counter()

    # Load moving image
    if isinstance(mov, str):
        mov_img = nib.load(mov)
        mov_path = mov
    else:
        mov_img = mov
        mov_path = mov.get_filename() if hasattr(mov, 'get_filename') else None

    mov_data = torch.from_numpy(mov_img.get_fdata()).float()

    # Load surfaces
    if subject_dir is not None:
        if verbose:
            print(f"Loading surfaces from subject directory: {subject_dir}")
        lh_data = load_surface_from_subject(
            subject_dir, hemi='lh', surf_name='white',
            load_thickness=True, device=device
        )
        rh_data = load_surface_from_subject(
            subject_dir, hemi='rh', surf_name='white',
            load_thickness=True, device=device
        )

        # Load orig.mgz as target reference (like bbregister does)
        # Surfaces are defined in this space
        orig_path = Path(subject_dir) / 'mri' / 'orig.mgz'
        if orig_path.exists():
            trg_img = nib.load(str(orig_path))
            trg_path = str(orig_path)
            if verbose:
                print(f"Target reference: {orig_path}")
                print(f"  Shape: {trg_img.shape[:3]}")
                print(f"  Voxel size: {trg_img.header.get_zooms()[:3]}")
        else:
            # Fallback to using moving image as target
            if verbose:
                print(f"WARNING: orig.mgz not found at {orig_path}")
                print(f"  Using moving image as target reference")
            trg_img = mov_img
            trg_path = mov_path
    else:
        # Load from explicit paths
        if lh_surf is None and rh_surf is None:
            raise ValueError("Must provide either subject_dir or lh_surf/rh_surf")

        lh_data = None
        rh_data = None

        if lh_surf is not None:
            if verbose:
                print(f"Loading left hemisphere surface: {lh_surf}")
            lh_thickness_path = str(Path(lh_surf).parent / 'lh.thickness')
            if not Path(lh_thickness_path).exists():
                lh_thickness_path = None
            lh_data = load_surface(lh_surf, lh_thickness_path, device=device)

        if rh_surf is not None:
            if verbose:
                print(f"Loading right hemisphere surface: {rh_surf}")
            rh_thickness_path = str(Path(rh_surf).parent / 'rh.thickness')
            if not Path(rh_thickness_path).exists():
                rh_thickness_path = None
            rh_data = load_surface(rh_surf, rh_thickness_path, device=device)

        # When using explicit paths, use moving image as target
        trg_img = mov_img
        trg_path = mov_path

    # Get vox2ras_tkr matrix for the moving image
    vox2ras_tkr = torch.from_numpy(get_vox2ras_tkr(mov_img)).float()

    if verbose:
        print(f"Moving image shape: {mov_data.shape}")
        print(f"vox2ras_tkr matrix:\n{vox2ras_tkr}")
        if lh_data is not None:
            print(f"LH vertices: {lh_data['vertices'].shape[0]}")
        if rh_data is not None:
            print(f"RH vertices: {rh_data['vertices'].shape[0]}")

    # Handle initialization
    init_transform = None
    if init_type == 'lta':
        if init_lta is None:
            raise ValueError("init_lta must be provided when init_type='lta'")
        # TODO: Load LTA and convert to transform matrix
        # For now, use identity
        if verbose:
            print(f"WARNING: LTA initialization not yet implemented, using identity")
        init_transform = torch.eye(4, dtype=torch.float32)
    elif init_type == 'centroid':
        # TODO: Implement centroid-based initialization for surfaces
        if verbose:
            print("WARNING: Centroid initialization not yet implemented, using identity")
        init_transform = torch.eye(4, dtype=torch.float32)
    else:  # header
        # Use header-based alignment (identity in tkRAS space)
        init_transform = torch.eye(4, dtype=torch.float32)

    # Create BBR model
    if verbose:
        print(f"\nInitializing BBR model...")
        print(f"  DOF: {dof}")
        print(f"  Contrast: {contrast}")
        print(f"  Cost type: {cost_type}")
        print(f"  Subsample: {subsample}")

    model = BBRModel(
        moving_volume=mov_data,
        lh_white_vertices=lh_data['vertices'] if lh_data is not None else None,
        lh_faces=lh_data['faces'] if lh_data is not None else None,
        rh_white_vertices=rh_data['vertices'] if rh_data is not None else None,
        rh_faces=rh_data['faces'] if rh_data is not None else None,
        lh_thickness=lh_data.get('thickness') if lh_data is not None else None,
        rh_thickness=rh_data.get('thickness') if rh_data is not None else None,
        vox2ras_tkr=vox2ras_tkr,
        dof=dof,
        init_transform=init_transform,
        contrast=contrast,
        wm_proj_abs=wm_proj_abs,
        gm_proj_frac=gm_proj_frac,
        slope=slope,
        cost_type=cost_type,
        gradient_weight=gradient_weight,
        subsample=subsample,
        device=device
    ).to(device)

    # Optimize
    if verbose:
        print(f"\nOptimizing registration...")
        print(f"  Iterations: {n_iters}")
        print(f"  Learning rate: {lr}")

    # Use RMSprop like image-based registration (better for this type of optimization)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    losses = []
    for iteration in range(n_iters):
        optimizer.zero_grad()
        cost = model()
        cost.backward()
        optimizer.step()

        losses.append(cost.item())

        if verbose and (iteration % 10 == 0 or iteration == n_iters - 1):
            print(f"  Iteration {iteration:4d}: cost = {cost.item():.6f}")

    # Get final transformation
    final_transform = model.get_transform_matrix()

    if verbose:
        print(f"\nFinal transformation matrix:")
        print(final_transform)
        print(f"\nTotal time: {time.perf_counter() - start:.2f} seconds")

    # Save LTA if requested
    if lta_name is not None:
        if verbose:
            print(f"Writing LTA file: {lta_name}")

        # Convert nibabel headers to dictionary format
        mov_header_dict = header_to_dict(mov_img)
        trg_header_dict = header_to_dict(trg_img)

        # Convert RAS-to-RAS transform to vox-to-vox using target affine
        # Formula: Mv2v = inv(dst_affine) @ Mr2r @ src_affine
        # This matches bbregister: src=fMRI, dst=orig.mgz
        ras_transform_np = final_transform.detach().cpu().numpy()
        vox_transform = ras_to_vox_transform(
            ras_transform_np,
            mov_img.affine,
            trg_img.affine  # Use target (orig.mgz) affine
        )

        if verbose:
            print(f"  Transform type: VOX_TO_VOX (src → target)")
            print(f"  Source: {mov_path}")
            print(f"  Target: {trg_path}")
            print(f"  Vox-to-vox matrix:\n{vox_transform}")

        # Write LTA with vox-to-vox transform (type=0)
        write_lta(
            lta_name,
            vox_transform,
            mov_path if mov_path else "moving.mgz",
            mov_header_dict,
            trg_path if trg_path else "target.mgz",
            trg_header_dict,
            lta_type=0  # VOX_TO_VOX
        )

    return final_transform.detach()


import torch
import os
import pyvista as pv
import numpy as np
import pandas as pd
import mcubes
import trimesh
from tqdm import tqdm
import glob
import vtk
import imageio
from sklearn.decomposition import PCA
from skimage.io import imsave
from src.models.utils import sample_points
from pathlib import Path
from aicsimageio import AICSImage
from vtk.util import numpy_support
from skimage import morphology as skmorpho
from skimage import filters as skfilters
from skimage.measure import label, marching_cubes


def write_pyvista_latent_walk_gif(out_file, view, mesh_files, expl_var=None):
    n_bins = len(mesh_files)
    if n_bins == 9:
        bin_label_map = {
            1: "-2σ",
            2: "-1.5σ",
            3: "-1σ",
            4: "-0.5σ",
            5: "0",
            6: "0.5σ",
            7: "1σ",
            8: "1.5σ",
            9: "2σ",
        }
    else:
        raise NotImplementedError
    shape_mode = mesh_files[0].split("/")[-2]

    def dummy_callback(arg):
        return

    mesh = pv.read(mesh_files[0])
    if mesh.points.shape[0] == 0:
        vertices = np.array([[0, 0, 0], [1, 0, 0]])
        mesh = pv.PolyData(vertices)
    mesh.clear_data()
    plotter = pv.Plotter(window_size=[650, 600])
    plotter.add_mesh(mesh, color="white")
    plotter.set_background("white")
    plotter.camera_position = view
    plotter.camera.zoom = 0.9
    if view == "xy":
        plotter.add_text(shape_mode, font_size=22, color="black", position=(10, 250))
        if "PC1" == shape_mode:
            plotter.add_text("Top view", color="black", position=(250, 500))
    elif view == "xz":
        if "PC1" == shape_mode:
            plotter.add_text("Side view (Y)", color="black", position=(250, 500))
    elif view == "yz":
        if "PC1" == shape_mode:
            plotter.add_text("Side view (Z)", color="black", position=(250, 500))
        if expl_var is not None:
            if expl_var < 1:
                expl_var = expl_var * 100
            plotter.add_text(
                f"{round(expl_var, 1)}%", color="black", position=(550, 250)
            )
    else:
        raise NotImplementedError
    plotter.open_gif(out_file)
    slider = plotter.add_slider_widget(
        dummy_callback,
        [1, n_bins],
        1,
        color="black",
        pointa=(0.2, 0.1),
        pointb=(0.85, 0.1),
    )
    mesh_files_tmp = mesh_files.copy()
    for i, f in enumerate(mesh_files_tmp):
        new_mesh = pv.read(f)
        if new_mesh.points.shape[0] == 0:
            vertices = np.array([[0, 0, 0], [1, 0, 0]])
            new_mesh = pv.PolyData(vertices)
        new_mesh.clear_data()
        mesh.overwrite(new_mesh)
        slider.GetSliderRepresentation().SetValue(i + 1)
        slider.GetSliderRepresentation().SetTitleText(bin_label_map[i + 1])
        slider.GetSliderRepresentation().SetShowSliderLabel(False)
        plotter.camera.reset_clipping_range()
        plotter.render()
        plotter.write_frame()
    # Make gif continuous
    mesh_files_tmp.reverse()
    for i, f in enumerate(mesh_files_tmp):
        new_mesh = pv.read(f)
        new_mesh.clear_data()
        mesh.overwrite(new_mesh)
        slider.GetSliderRepresentation().SetValue(n_bins - i)
        slider.GetSliderRepresentation().SetTitleText(bin_label_map[n_bins - i])
        slider.GetSliderRepresentation().SetShowSliderLabel(False)
        plotter.camera.reset_clipping_range()
        plotter.render()
        plotter.write_frame()
    plotter.close()


def write_shape_mode_latent_walk_gif(
    shape_mode, latent_meshes_dir, expl_var, num_structures, mesh_file_suffix="vtk"
):
    pv.start_xvfb()

    mesh_files = sorted(glob.glob(f"{latent_meshes_dir}/*.{mesh_file_suffix}"))

    if num_structures > 1:
        mesh_files_2d = []
        num_bins = int(len(mesh_files) / num_structures)
        for i in range(1, num_bins + 1):
            matched_files = glob.glob(f"{latent_meshes_dir}/*{i}.{mesh_file_suffix}")
            mesh_files_2d.append(matched_files)
        mesh_files = mesh_files_2d

    out_file_xy = f"{latent_meshes_dir}/{shape_mode}.gif"
    out_file_xz = f"{latent_meshes_dir}/{shape_mode}_xz.gif"
    out_file_yz = f"{latent_meshes_dir}/{shape_mode}_yz.gif"

    write_pyvista_latent_walk_gif(out_file_xy, "xy", mesh_files)
    write_pyvista_latent_walk_gif(out_file_xz, "xz", mesh_files)
    write_pyvista_latent_walk_gif(out_file_yz, "yz", mesh_files, expl_var)

    print(f"Wrote animated latent walk for {shape_mode} at {out_file_xy}")
    return [out_file_xy, out_file_xz, out_file_yz]


def write_combined_latent_walk_gif(gif_sets_paths, out_path):
    sorted_gif_paths = sorted(
        gif_sets_paths, key=lambda x: int("".join([i for i in x[0] if i.isdigit()]))
    )
    comb_gif_path = Path(out_path)
    gif_stack = []
    for gifs in sorted_gif_paths:
        gif_x = AICSImage(gifs[0]).data.squeeze()
        if len(gif_x.shape) == 3:
            gif_x = np.stack([gif_x, gif_x, gif_x, gif_x], axis=-1)
        gif_y = AICSImage(gifs[1]).data.squeeze()
        if len(gif_y.shape) == 3:
            gif_y = np.stack([gif_y, gif_y, gif_y, gif_y], axis=-1)
        gif_z = AICSImage(gifs[2]).data.squeeze()
        if len(gif_z.shape) == 3:
            gif_z = np.stack([gif_z, gif_z, gif_z, gif_z], axis=-1)
        gif_set = np.concatenate([gif_x, gif_y, gif_z], axis=-2)
        gif_stack.append(gif_set)
    gif_stack = np.array(gif_stack)
    if gif_stack.shape[0] > 8:
        n_cols = int(gif_stack.shape[0] / 8)
        gif_stack = np.concatenate(gif_stack[:], axis=-3)
        newy = int(gif_stack.shape[1] / n_cols)
        col_arrs = []
        for i in range(n_cols):
            col_arr = gif_stack[:, i * newy : (i + 1) * newy, :, :]
            col_arrs.append(col_arr)
        gif_stack = np.concatenate(np.array(col_arrs)[:], axis=-2)
    else:
        gif_stack = np.concatenate(gif_stack[:], axis=-3)
    imageio.mimsave(out_path, gif_stack, loop=0)
    print(f"Wrote {comb_gif_path}")


def get_mesh_from_image(
    image: np.array,
    sigma: float = 0,
    lcc: bool = True,
    denoise: bool = False,
    translate_to_origin: bool = True,
    noise_thresh: int = 80,
):
    """
    Parameters
    ----------
    image : np.array
        Input array where the mesh will be computed on
    Returns
    -------
    mesh : vtkPolyData
        3d mesh in VTK format
    img_output : np.array
        Input image after pre-processing
    centroid : np.array
        x, y, z coordinates of the mesh centroid
    Other parameters
    ----------------
    lcc : bool, optional
        Whether or not to compute the mesh only on the largest
        connected component found in the input connected component,
        default is True.
    denoise : bool, optional
        Whether or not to remove small, potentially noisy objects
        in the input image, default is False.
    sigma : float, optional
        The degree of smooth to be applied to the input image, default
        is 0 (no smooth).
    translate_to_origin : bool, optional
        Wheather or not translate the mesh to the origin (0,0,0),
        default is True.
    """

    img = image.copy()

    # VTK requires YXZ
    img = np.swapaxes(img, 0, 2)

    # Extracting the largest connected component
    if lcc:

        img = skmorpho.label(img.astype(np.uint8))

        counts = np.bincount(img.flatten())

        lcc = 1 + np.argmax(counts[1:])

        img[img != lcc] = 0
        img[img == lcc] = 1

    # Remove small objects in the image
    if denoise:
        img = skmorpho.remove_small_objects(label(img), noise_thresh)

    # Smooth binarize the input image and binarize
    if sigma:

        img = skfilters.gaussian(img.astype(np.float32), sigma=(sigma, sigma, sigma))

        img[img < 1.0 / np.exp(1.0)] = 0
        img[img > 0] = 1

        if img.sum() == 0:
            raise ValueError(
                "No foreground voxels found after pre-processing. Try using sigma=0."
            )

    # Set image border to 0 so that the mesh forms a manifold
    img[[0, -1], :, :] = 0
    img[:, [0, -1], :] = 0
    img[:, :, [0, -1]] = 0
    img = img.astype(np.float32)

    if img.sum() == 0:
        raise ValueError(
            "No foreground voxels found after pre-processing."
            "Is the object of interest centered?"
        )

    # Create vtkImageData
    imgdata = vtk.vtkImageData()
    imgdata.SetDimensions(img.shape)

    img = img.transpose(2, 1, 0)
    img_output = img.copy()
    img = img.flatten()
    arr = numpy_support.numpy_to_vtk(img, array_type=vtk.VTK_FLOAT)
    arr.SetName("Scalar")
    imgdata.GetPointData().SetScalars(arr)

    # Create 3d mesh
    cf = vtk.vtkContourFilter()
    cf.SetInputData(imgdata)
    cf.SetValue(0, 0.5)
    cf.Update()

    mesh = cf.GetOutput()

    # Calculate the mesh centroid
    coords = numpy_support.vtk_to_numpy(mesh.GetPoints().GetData())
    centroid = coords.mean(axis=0, keepdims=True)

    if translate_to_origin is True:

        # Translate to origin
        coords -= centroid
        mesh.GetPoints().SetData(numpy_support.numpy_to_vtk(coords))

    return mesh, img_output, tuple(centroid.squeeze())


def get_mesh_from_sdf(sdf, method="skimage"):
    """
    This function reconstructs a mesh from signed distance function
    values using the marching cubes algorithm.

    Parameters
    ----------
    sdf : np.array
        3D array of shape (N,N,N)

    Returns
    -------
    mesh : pyvista.PolyData
        Reconstructed mesh
    """
    if method == "skimage":
        try:
            vertices, faces, normals, _ = marching_cubes(sdf, level=0)
            mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, vertex_normals=normals
            )
        except:
            # empty mesh
            mesh = pv.PolyData()
    elif method == "vae_output":
        vertices, triangles = mcubes.marching_cubes(sdf, 0)
        mcubes.export_obj(vertices, triangles, "tmp.obj")
        mesh = pv.read("tmp.obj")
        os.remove("tmp.obj")
    else:
        raise NotImplementedError

    mesh = pv.wrap(mesh)
    return mesh


def save_pcloud(xhat, path, name, z_max, z_ind=2):
    """
    Save pointcloud xhat
    z_max - percentage of z_max to place cut at

    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if torch.is_tensor(xhat):
        xhat = xhat.detach().cpu().numpy()
    # this_recon = pv.PolyData(xhat[:, :3])
    # this_recon.save(path / f"{name}.ply", texture=xhat[:, :].astype(np.uint8))
    if len(xhat[0,:]) == 4:
        cols = ['x', 'y', 'z', 's']
    else:
        cols = ['x', 'y', 'z']

    if z_max is None:
        this_recon = pd.DataFrame(xhat, columns=cols)
        this_recon.to_csv(path / f"{name}.csv")
    else:
        max_num = xhat[:, z_ind].max()
        # ratio = 0.5
        ratio = z_max
        inds = np.where(xhat[:, z_ind] < ratio * max_num)[0]
        xhat = xhat[inds]
        inds = np.where(xhat[:, z_ind] > -ratio * max_num)[0]
        xhat = xhat[inds]

        # this_recon = pv.PolyData(xhat[:, :3])
        # this_recon.save(path / f"{name}_center.ply", texture=xhat[:, :].astype(np.uint8))
        this_recon = pd.DataFrame(xhat, columns=cols)
        this_recon.to_csv(path / f"{name}.csv")
    return xhat


def make_canonical_shapes(
    model, df, device, path, slice_key, sub_slice_list, max_embed_dim, key, z_max=None, z_ind=2
):
    model = model.eval()
    cols = [i for i in df.columns if "mu" in i]
    all_xhat = []
    for stage in sub_slice_list:
        this_stage_mu = (
            df.loc[df[slice_key] == stage][cols]
            .iloc[:, :max_embed_dim]
            .dropna(axis=1)
            .values
        )
        with torch.no_grad():
            # z_inf = torch.tensor(this_stage_mu).mean(axis=0).unsqueeze(axis=0)
            # idx = np.random.randint(this_stage_mu.shape[0], size=1)
            # this_stage_mu = this_stage_mu[idx]
            z_inf = torch.tensor(this_stage_mu).mean(axis=0).unsqueeze(axis=0)
            z_inf = z_inf.to(device)
            z_inf = z_inf.float()
            decoder = model.decoder[key]
            xhat = decoder(z_inf).detach().cpu().numpy()

            if len(xhat.shape) > 3:
                xhat = sample_points(xhat.detach().cpu().numpy())

            xhat = save_pcloud(xhat[0], path, stage, z_max, z_ind)
            all_xhat.append(xhat)
    return all_xhat


def extract_digitized_shape_modes(shape_mode, shape_modes_df, pca, map_points):
    def invert(pcs, pca):
        """Matrix has shape NxM, where N is the number of
        samples and M is the number of shape modes."""
        # Inverse PCA here: PCA coords -> shcoeffs
        df = pd.DataFrame(pca.inverse_transform(pcs))
        return df

    def get_coordinates_matrix(coords, comp, n_shape_modes):
        """Coords has shape (N,). Creates a matrix of shape
        (N,M), where M is the reduced dimension. comp is an
        integer from 1 to npcs."""
        npts = len(coords)
        matrix = np.zeros((npts, n_shape_modes), dtype=np.float32)
        matrix[:, comp] = coords
        return matrix

    n_shape_modes = len(shape_modes_df.columns)
    values = shape_modes_df[shape_mode].values.astype(np.float32)
    values -= values.mean()
    active_scale = values.std()
    values /= active_scale
    bin_centers = map_points
    # Line below handle single bins
    binw = 0.5 * np.diff(bin_centers).mean() if len(bin_centers) > 1 else 1
    bin_edges = np.unique([(b - binw, b + binw) for b in bin_centers])
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    meta = shape_modes_df.copy()
    meta["mpId"] = np.digitize(values, bin_edges)
    coords = [m * active_scale for m in bin_centers]

    matrix = get_coordinates_matrix(
        coords, shape_modes_df.columns.get_loc(shape_mode), n_shape_modes
    )
    df_inv = invert(matrix, pca)
    df_inv["shape_mode"] = shape_mode
    df_inv["mpId"] = np.arange(1, 1 + len(bin_centers))

    return df_inv


def stratified_latent_walk(    
    model,
    device,
    df,
    x_label,
    max_embed_dim,
    latent_dim,
    max_num_shapemodes,
    path,
    stratify_key,
    z_max=None,
    latent_walk_range=None,
    z_ind=2,
):
    for strat in df[stratify_key].unique():
        this_df = df.loc[df[stratify_key] == strat].reset_index(drop=True)
        latent_walk(
            model,
            device,
            this_df,
            x_label=x_label,
            max_embed_dim=max_embed_dim,
            latent_dim=latent_dim,
            max_num_shapemodes=max_num_shapemodes,
            path=path,
            z_max=z_max,
            latent_walk_range=latent_walk_range,
            z_ind=z_ind,
            sub_key=strat,
        )


def latent_walk(
    model,
    device,
    df,
    x_label,
    max_embed_dim,
    latent_dim,
    max_num_shapemodes,
    path,
    z_max=None,
    latent_walk_range=None,
    z_ind=2,
    sub_key=None,
):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cols = [i for i in df.columns if "mu" in i]
    all_features = df[cols].iloc[:, :max_embed_dim].dropna(axis=1).values

    if all_features.shape[0] < 256:
        n = 256 - all_features.shape[0]  # for 2 random indices
        index = np.random.choice(all_features.shape[0], n, replace=False) 
        random_sample = all_features[index] + 0.01*np.random.randn(256)
        all_features = np.concatenate([all_features, random_sample], axis=0)

    pca = PCA(n_components=256)
    pca_features = pca.fit_transform(all_features)
    pca_std_list = pca_features.std(axis=0)

    all_recons = {}
    with torch.no_grad():
        if latent_walk_range is None:
            latent_walk_range = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

        for rank in range(max_num_shapemodes):
            if rank == max_num_shapemodes:
                break

            all_recons[f"{rank}"] = {"latent_point": [], "recon": [], "rank": []}

            for value_index, value in enumerate(
                tqdm(latent_walk_range, total=len(latent_walk_range))
            ):
                z_inf = torch.zeros(1, latent_dim)

                z_inf[:, rank] += value * pca_std_list[rank]
                z_inf = pca.inverse_transform(z_inf)
                z_inf = torch.tensor(z_inf)
                z_inf = z_inf.to(device)
                z_inf = z_inf.float()
                decoder = model.decoder[x_label]
                xhat = decoder(z_inf)

                if len(xhat.shape) > 3:
                    xhat = sample_points(xhat.detach().cpu().numpy())
                else:
                    xhat = xhat.detach().cpu().numpy()
                if sub_key is None:
                    name = f"{rank}_{value_index}"
                else:
                    name = f"{sub_key}_{rank}_{value_index}"
                save_pcloud(xhat[0], path, name, z_max, z_ind)

                all_recons[f"{rank}"]["latent_point"].append(value)
                all_recons[f"{rank}"]["recon"].append(xhat)
                all_recons[f"{rank}"]["rank"].append(rank)

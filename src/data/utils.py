import vtk
import os
import numpy as np
from vtk.util import numpy_support
import pyvista as pv
import trimesh
import mcubes
import pandas as pd

from skimage.measure import marching_cubes
from skimage import filters as skfilters
from skimage import morphology as skmorpho
from skimage.measure import label

from aicsshparam import shtools, shparam

from aicscytoparam.cytoparam import voxelize_mesh
from vtk.util.numpy_support import vtk_to_numpy
import math


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
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
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

def vtk_polydata_to_imagedata(polydata, dimensions=(64,64,64), padding=0):
    xi, xf, yi, yf, zi, zf = polydata.GetBounds()
    dx, dy, dz = dimensions
    sx = (xf - xi) / dx
    sy = (yf - yi) / dy
    sz = (zf - zi) / dz
    ox = xi + sx / 2.0
    oy = yi + sy / 2.0
    oz = zi + sz / 2.0

    if padding:
        ox -= sx
        oy -= sy
        oz -= sz

        dx += 2 * padding
        dy += 2 * padding
        dz += 2 * padding

    image = vtk.vtkImageData()
    image.SetSpacing((sx, sy, sz))
    image.SetDimensions((dx, dy, dz))
    image.SetExtent(0, dx - 1, 0, dy - 1, 0, dz - 1)
    image.SetOrigin((ox, oy, oz))
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    inval = 255
    outval = 0

    for i in range(image.GetNumberOfPoints()):
        image.GetPointData().GetScalars().SetTuple1(i, inval)

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(polydata)
    pol2stenc.SetOutputOrigin((ox, oy, oz))
    pol2stenc.SetOutputSpacing((sx, sy, sz))
    pol2stenc.SetOutputWholeExtent(image.GetExtent())
    pol2stenc.Update()
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(image)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(outval)
    imgstenc.Update()

    return imgstenc.GetOutput()


def vtk_image_to_numpy_image(vtk_image):
    dims = vtk_image.GetDimensions()
    data = vtk_image.GetPointData().GetScalars()
    np_image = numpy_support.vtk_to_numpy(data)
    np_image = np_image.reshape(dims, order='F')
    return np_image


def get_image_from_mesh(mesh, img_shape, padding):
    vtk_image = vtk_polydata_to_imagedata(mesh, dimensions=img_shape, padding=padding)
    np_image = vtk_image_to_numpy_image(vtk_image)
    return np_image


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
    print('before translating to centrooid')
    if translate_to_origin is True:
        print('translating to centrooid')
        # Translate to origin
        coords -= centroid
        mesh.GetPoints().SetData(numpy_support.numpy_to_vtk(coords))

    return mesh, img_output, tuple(centroid.squeeze())


def center_polydata(polydata):
    """
    Center a polydata mesh around the object's center of mass

    Parameters
    ----------
    polydata : vtk.PolyData
        Polydata mesh

    Returns
    -------
    polydata : pyvista.PolyData
        Centered mesh

    """
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(polydata)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    center = np.array(centerOfMassFilter.GetCenter())

    transform = vtk.vtkTransform()
    transform.Translate(-center)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetTransform(transform)
    transform_filter.SetInputData(polydata)
    transform_filter.Update()

    polydata = transform_filter.GetOutput()
    return polydata


def get_mesh_bbox_shape(mesh):
    bounds = mesh.GetBounds()
    x_dim = math.ceil(bounds[1] - bounds[0])
    y_dim = math.ceil(bounds[3] - bounds[2])
    z_dim = math.ceil(bounds[5] - bounds[4])
    return (x_dim, y_dim, z_dim)


def rescale_meshed_sdfs_to_full(list_of_meshes, scale_factors, resolution=32):
    rev_scale_factors = []
    resc_meshed_sdfs = []
    for i,m in enumerate(list_of_meshes): 
        orig_max_axis_length = resolution / scale_factors[i]
        rev_xfac = orig_max_axis_length / resolution
        resc_recon_mesh, _ = scale_polydata(m, None, rev_xfac)
        resc_recon_mesh = pv.wrap(center_polydata(resc_recon_mesh))
        rev_scale_factors.append(rev_xfac)
        resc_meshed_sdfs.append(resc_recon_mesh)
    return resc_meshed_sdfs, rev_scale_factors


def voxelize_recon_meshes(recon_meshes):
    vox_recon_meshes = []
    for i,rm in enumerate(recon_meshes):
        target_bounds = get_mesh_bbox_shape(recon_meshes[i])
        recon_vox_mesh_img = get_image_from_mesh(rm, target_bounds, padding=2)
        vox_recon_meshes.append(recon_vox_mesh_img)
    return vox_recon_meshes


def get_scale_factor_for_bounds(polydata, resolution):
    bounds = polydata.GetBounds()
    bounds = tuple([b + int(resolution/3) if b > 0 else b-int(resolution/3) \
                    for b in list(bounds)]) # Increasing bounds to prevent mesh from getting clipped
    x_delta = (bounds[1] - bounds[0])
    y_delta = (bounds[3] - bounds[2])
    z_delta = (bounds[5] - bounds[4])
    
    max_delta = max([x_delta, y_delta, z_delta])
    scale_factor = resolution/max_delta
    return scale_factor


def get_scaled_mesh(mesh, vox_resolution, scale_factor, vpolydata=None):
    vpolydata = pv.wrap(mesh) 
    centered_vpolydata = center_polydata(vpolydata)
    scaled_vpolydata, scale_factor = scale_polydata(centered_vpolydata, int(vox_resolution), scale_factor)
    return scaled_vpolydata, scale_factor


def scale_polydata(polydata, resolution, scale_factor=None):
    """
    Rescale a polydata mesh to fit into specified bounds

    Parameters
    ----------
    polydata : vtk.PolyData
        Polydata mesh
    
    resolution: int
        Bound to be used in all 3 dimensions (Z,Y,X) to rescale the mesh

    Returns
    -------
    scaled_polydata : pyvista.PolyData
        Scaled polydata

    """
    if scale_factor is None:
        scale_factor = get_scale_factor_for_bounds(polydata, resolution)
    
    xform = vtk.vtkTransform()
    xform.Scale(
        scale_factor,
        scale_factor,
        scale_factor,
    )
    xformoperator = vtk.vtkTransformPolyDataFilter()
    xformoperator.SetTransform(xform)
    xformoperator.SetInputData(0, polydata)
    xformoperator.Update()
    
    scaled_polydata = xformoperator.GetOutput()
    return scaled_polydata, scale_factor


def get_sdf_from_mesh_vtk(mesh_path, vox_resolution=32, scale_factor=None):
    """
    Compute 3D signed distance function values of a mesh 

    Parameters
    ----------
    mesh : str
        Path to mesh file

    vox_resolution : int
        SDF representation size in all 3 dimensions (Z,Y,X)

    Returns
    -------
    sdf : np.array
        Resulting SDF with shape (vox_resolution, vox_resolution, vox_resolution)

    """
    vox_shape = (vox_resolution,vox_resolution,vox_resolution)
    scaled_vpolydata, scale_factor = get_scaled_mesh(mesh_path, 
                                                    int(vox_resolution), 
                                                    scale_factor)
    
    pdd = vtk.vtkImplicitPolyDataDistance()
    pdd.SetInput(scaled_vpolydata)

    sdf = np.zeros(vox_shape)
    factor = int(vox_resolution/2) 
    for i in range(-factor,factor):
        for j in range(-factor,factor):
            for k in range(-factor,factor):
                sdf[i+factor, j+factor, k+factor] = pdd.EvaluateFunction([i, j, k])
    return sdf, scale_factor
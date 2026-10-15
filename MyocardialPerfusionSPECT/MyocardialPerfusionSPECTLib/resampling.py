"""Resampling of SPECT volumes onto the cardiac axes (VTK and numpy only, no Slicer).

Why higher-order interpolation
    Reconstructed perfusion SPECT has coarse voxels (typically 4-7 mm). Linear interpolation is
    continuous but its gradient jumps at every input voxel boundary, so a volume resampled onto
    2 mm oblique slices is a patchwork of flat facets aligned with the *original* voxel grid.
    After reorientation that grid is tilted against the new slices and the facets read as blocks.
    A cubic B-spline is smooth across voxel boundaries (continuous first and second derivative),
    which removes the facets without blurring: with the prefilter it passes exactly through the
    original voxel values.

Modes
    bspline  cubic B-spline interpolation (default): smooth, exact at the voxel centers
    sinc     Lanczos windowed sinc (half width 3): sharpest, slight ringing at hot edges
    linear   trilinear: no overshoot, faceted on coarse input grids
B-spline and sinc can overshoot a little next to steep edges; their output is clipped at zero
so counts stay non-negative.
"""

import numpy as np
import vtk
import vtk.util.numpy_support as numpy_support

INTERPOLATION_MODES = ("bspline", "sinc", "linear")
INTERPOLATION_LABELS = {
    "bspline": "Cubic B-spline (smooth)",
    "sinc": "Windowed sinc (Lanczos, sharp)",
    "linear": "Linear",
}
DEFAULT_INTERPOLATION = "bspline"


def _interpolator(mode):
    """(VTK interpolator, needs B-spline coefficients as input)."""
    if mode == "bspline":
        interpolator = vtk.vtkImageBSplineInterpolator()
        interpolator.SetSplineDegree(3)
        prefilter = True
    elif mode == "sinc":
        interpolator = vtk.vtkImageSincInterpolator()
        interpolator.SetWindowFunctionToLanczos()
        interpolator.SetWindowHalfWidth(3)
        interpolator.AntialiasingOff()   # we upsample; antialiasing only matters when shrinking
        prefilter = False
    elif mode == "linear":
        interpolator = vtk.vtkImageInterpolator()
        interpolator.SetInterpolationModeToLinear()
        prefilter = False
    else:
        raise ValueError(f"Unknown interpolation mode '{mode}'")
    interpolator.SetBorderModeToClamp()
    return interpolator, prefilter


def _bsplineCoefficients(imageData):
    coefficients = vtk.vtkImageBSplineCoefficients()
    coefficients.SetInputData(imageData)
    coefficients.SetSplineDegree(3)
    coefficients.SetBorderModeToClamp()
    coefficients.SetOutputScalarTypeToFloat()
    coefficients.Update()
    return coefficients.GetOutput()


def reslice(imageData, resliceAxes, outputDimensions, mode=DEFAULT_INTERPOLATION,
            outputOrigin=(0.0, 0.0, 0.0), outputSpacing=(1.0, 1.0, 1.0), tolerance=None):
    """Sample imageData at output grid points mapped through resliceAxes (output -> input IJK
    when imageData has unit spacing and zero origin, as Slicer volume image data does).

    outputDimensions  (ni, nj, nk)
    tolerance         voxels outside the input extent by less than this (in voxels) are
                      clamped to the edge instead of set to the background
    Returns a float32 array [k, j, i]."""
    ni, nj, nk = (int(n) for n in outputDimensions)
    interpolator, prefilter = _interpolator(mode)
    if tolerance is not None:
        interpolator.SetTolerance(float(tolerance))
    source = _bsplineCoefficients(imageData) if prefilter else imageData

    filter_ = vtk.vtkImageReslice()
    filter_.SetInputData(source)
    if resliceAxes is not None:
        filter_.SetResliceAxes(resliceAxes)
    filter_.SetInterpolator(interpolator)
    if mode == "linear":
        # vtkImageReslice pushes its own mode (default: nearest) into a plain vtkImageInterpolator
        filter_.SetInterpolationModeToLinear()
    filter_.SetOutputScalarType(vtk.VTK_FLOAT)
    filter_.SetOutputOrigin(*outputOrigin)
    filter_.SetOutputSpacing(*outputSpacing)
    filter_.SetOutputExtent(0, ni - 1, 0, nj - 1, 0, nk - 1)
    filter_.SetBackgroundLevel(0.0)
    filter_.Update()
    values = numpy_support.vtk_to_numpy(filter_.GetOutput().GetPointData().GetScalars())
    values = np.array(values, dtype=np.float32).reshape(nk, nj, ni)
    if mode != "linear":
        np.maximum(values, 0.0, out=values)
    return values


def arrayToImageData(array):
    """numpy [k, j, i] (or [j, i]) -> vtkImageData with unit spacing and zero origin."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    nk, nj, ni = array.shape
    image = vtk.vtkImageData()
    image.SetDimensions(ni, nj, nk)
    scalars = numpy_support.numpy_to_vtk(np.ascontiguousarray(array).reshape(-1), deep=True,
                                         array_type=vtk.VTK_FLOAT)
    image.GetPointData().SetScalars(scalars)
    return image


def zoomInPlane(stack, height, width, mode=DEFAULT_INTERPOLATION):
    """Resize every slice of a [n, h, w] stack to [n, height, width] (pixel centers aligned,
    edges clamped), for display at screen resolution before colour mapping."""
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[None]
    n, h, w = stack.shape
    height, width = max(1, int(height)), max(1, int(width))
    if (h, w) == (height, width):
        return stack.copy()
    scaleX, scaleY = w / float(width), h / float(height)
    return reslice(arrayToImageData(stack), None, (width, height, n), mode,
                   outputOrigin=(0.5 * scaleX - 0.5, 0.5 * scaleY - 0.5, 0.0),
                   outputSpacing=(scaleX, scaleY, 1.0), tolerance=0.5)

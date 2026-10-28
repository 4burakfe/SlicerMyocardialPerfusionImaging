"""Cardiac-frame geometry helpers: vectors, matrices, ellipsoid masks (numpy and VTK only)."""

import numpy as np
import vtk

SUPERIOR = np.array([0.0, 0.0, 1.0])
PATIENT_LEFT = np.array([-1.0, 0.0, 0.0])


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def sampleTrilinear(array, ijk):
    """Trilinear samples of a [k, j, i] array at N continuous (i, j, k) points; 0 outside the grid."""
    ijk = np.asarray(ijk, dtype=float)
    shape = np.array(array.shape[::-1])
    base = np.floor(ijk).astype(int)
    frac = ijk - base
    values = np.zeros(len(ijk))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                index = base + np.array([di, dj, dk])
                weight = ((frac[:, 0] if di else 1.0 - frac[:, 0]) * (frac[:, 1] if dj else 1.0 - frac[:, 1])
                          * (frac[:, 2] if dk else 1.0 - frac[:, 2]))
                inside = np.all((index >= 0) & (index < shape), axis=1)
                values[inside] += weight[inside] * array[index[inside, 2], index[inside, 1], index[inside, 0]]
    return values


def worldToLocalMatrix(node):
    """World RAS -> the node's own RAS (identity unless the node is under a transform)."""
    matrix = vtk.vtkMatrix4x4()
    parent = node.GetParentTransformNode()
    if parent:
        if not parent.IsTransformToWorldLinear():
            raise ValueError(f"'{node.GetName()}' is under a non-linear transform. Harden the transform first.")
        parent.GetMatrixTransformFromWorld(matrix)
    return matrix


def ellipsoidInsideGrid(shape, ijkToRas, centerWorld, axesWorld, semiAxes):
    """Boolean [k, j, i] mask of a grid (given its 4x4 IJK->RAS as numpy) inside an ellipsoid."""
    nk, nj, ni = shape
    m = np.asarray(ijkToRas, dtype=float)
    i = np.arange(ni)[None, None, :]
    j = np.arange(nj)[None, :, None]
    k = np.arange(nk)[:, None, None]
    offset = [m[a, 0] * i + m[a, 1] * j + m[a, 2] * k + (m[a, 3] - centerWorld[a]) for a in range(3)]
    quadratic = 0.0
    for direction, semi in zip(axesWorld, np.maximum(np.asarray(semiAxes, dtype=float), 1e-3)):
        projection = direction[0] * offset[0] + direction[1] * offset[1] + direction[2] * offset[2]
        quadratic = quadratic + (projection / semi) ** 2
    return quadratic <= 1.0


def ijkToRasMatrix(iColumn, jColumn, kColumn, origin):
    matrix = vtk.vtkMatrix4x4()
    for r in range(3):
        matrix.SetElement(r, 0, float(iColumn[r]))
        matrix.SetElement(r, 1, float(jColumn[r]))
        matrix.SetElement(r, 2, float(kColumn[r]))
        matrix.SetElement(r, 3, float(origin[r]))
    return matrix

import typing

import numpy


def build_patch_contrast_laplacian() -> numpy.ndarray:
    D = numpy.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            idx = 3 * i + j
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < 3 and 0 <= nj < 3:
                    nidx = 3 * ni + nj
                    D[idx, idx] += 1
                    D[nidx, nidx] += 1
                    D[idx, nidx] -= 1
                    D[nidx, idx] -= 1
    return D


def mumford_basis(D: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
    eigvals, eigvecs = numpy.linalg.eigh(D)
    eigvals, eigvecs = eigvals[eigvals > 1e-9], eigvecs[:, eigvals > 1e-9]
    eigvecs = eigvecs / numpy.sqrt(eigvals)[None, :]
    pivot = numpy.argmax(numpy.abs(eigvecs), axis=0)
    signs = numpy.sign(eigvecs[pivot, numpy.arange(eigvecs.shape[1])])
    return eigvecs * signs[None, :], eigvals


PATCH_CONTRAST_LAPLACIAN = build_patch_contrast_laplacian()
MUMFORD_BASIS, MUMFORD_EIGVALS = mumford_basis(PATCH_CONTRAST_LAPLACIAN)

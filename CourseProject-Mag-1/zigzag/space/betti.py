import typing

import numpy
import gudhi


def collect(complex: gudhi.AlphaComplex, field: int, max_dim: int = 2) -> typing.List[numpy.ndarray]:
    diag = complex.persistence(homology_coeff_field=field, min_persistence=0.0)
    dgms = [[] for _ in range(max_dim + 1)]
    for dim, (b, d) in diag:
        if dim > max_dim:
            continue
        b_a = numpy.sqrt(max(b, 0.0))
        d_a = numpy.inf if numpy.isinf(d) else numpy.sqrt(max(d, 0.0))
        dgms[dim].append([b_a, d_a])
    return [numpy.array(d) if d else numpy.empty((0, 2)) for d in dgms]


def measured_betti(dgms: typing.List[numpy.ndarray], min_life: float = 0.15) -> typing.Tuple[int]:
    betti = []
    for dgm in dgms:
        finite = dgm[numpy.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            betti.append(0)
            continue
        lifes = finite[:, 1] - finite[:, 0]
        ps = numpy.sort(lifes[lifes > min_life])[::-1]
        if len(ps) < 2:
            betti.append(len(ps))
            continue
        diffs = ps[:-1] / ps[1:]
        betti.append(numpy.argmax(diffs) + 1)
    betti[0] += 1
    return tuple(betti)


def get_space_betti(
    hidden_states: numpy.ndarray, coeffs: typing.Tuple[int] = (2, 3)
) -> typing.Dict[int, typing.Tuple[int]]:
    complex = gudhi.AlphaComplex(points=hidden_states.tolist())
    simplex_tree = complex.create_simplex_tree()
    return {coeff: measured_betti(collect(simplex_tree, coeff, hidden_states.shape[1] - 1)) for coeff in coeffs}

"""Cubic LSMPS type-A particle diffusion (Matsunaga et al., 2015, Sec. 2.3).

Three-dimensional extension of Eqs. (8), (19)--(27). Concentration is a
passive scalar. Axis-aligned box walls use equal-concentration mirror
particles (zero normal flux). No concentration clipping or mass correction
is performed: collocation errors remain visible in the diagnostics.
"""

import math

import warp as wp


Vec19 = wp.types.vector(length=19, dtype=wp.float32)
Mat19 = wp.types.matrix(shape=(19, 19), dtype=wp.float32)


@wp.func
def cubic_basis(r: wp.vec3):
    x, y, z = r[0], r[1], r[2]
    return Vec19(x, y, z, x*x, x*y, x*z, y*y, y*z, z*z,
                 x*x*x, x*x*y, x*x*z, x*y*y, x*y*z, x*z*z,
                 y*y*y, y*y*z, y*z*z, z*z*z)


@wp.func
def mirror_sample(xi: wp.vec3, xj: wp.vec3, mask: int,
                  lo: wp.vec3, hi: wp.vec3):
    # For support < half the box extent, at most one wall per axis can
    # contribute. Masks 1..7 also produce edge/corner mirror combinations.
    point = xj
    for axis in range(3):
        if (mask & (1 << axis)) != 0:
            wall = lo[axis]
            if xi[axis] > 0.5 * (lo[axis] + hi[axis]):
                wall = hi[axis]
            point[axis] = 2.0 * wall - xj[axis]
    return point


@wp.func
def sample_weight(offset: wp.vec3, support: float):
    r = wp.length(offset) / support
    weight = float(0.0)
    if r < 1.0 and r > 1.0e-7:
        weight = 1.0 / (r + 1.0e-6) - 1.0 / (1.0 + 1.0e-6)
    return weight


@wp.func
def solve_laplacian_fit(moment: Mat19, spacing: float, regularization: float, terms: int):
    # Diagonal equilibration and a small, explicit Tikhonov term keep the
    # float32 local solve usable on distorted APIC neighborhoods. Solve
    # M*a=e_laplacian rather than forming a complete matrix inverse.
    scale = Vec19(0.0)
    for a in range(terms):
        if moment[a, a] <= 1.0e-12:
            return Vec19(0.0), float(0.0), int(0)
        scale[a] = wp.sqrt(wp.max(moment[a, a], 1.0e-20))
    factor = Mat19(0.0)
    minimum_pivot = float(1.0)
    for a in range(terms):
        for b in range(a + 1):
            value = moment[a, b] / (scale[a] * scale[b])
            if a == b:
                value += regularization
            for k in range(b):
                value -= factor[a, k] * factor[b, k]
            if a == b:
                minimum_pivot = wp.min(minimum_pivot, value)
                if value <= wp.max(1.0e-9, 5.0 * regularization) or not wp.isfinite(value):
                    return Vec19(0.0), float(0.0), int(0)
                factor[a, b] = wp.sqrt(value)
            else:
                factor[a, b] = value / factor[b, b]
    rhs = Vec19(0.0)
    for a in range(terms):
        value = float(0.0)
        if a == 3 or a == 6 or a == 8:
            value = 2.0 / scale[a]
        for b in range(a):
            value -= factor[a, b] * rhs[b]
        rhs[a] = value / factor[a, a]
    solution = Vec19(0.0)
    for reverse in range(terms):
        a = terms - 1 - reverse
        value = rhs[a]
        for b in range(a + 1, terms):
            value -= factor[b, a] * solution[b]
        solution[a] = value / factor[a, a]
    for a in range(terms):
        solution[a] /= scale[a] * spacing * spacing
    return solution, minimum_pivot, int(1)


@wp.kernel
def build_laplacian_fit(
    grid: wp.uint64, positions: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=Vec19), pivots: wp.array(dtype=float),
    fit_orders: wp.array(dtype=int), invalid: wp.array(dtype=int),
    spacing: float, support: float, lo: wp.vec3, hi: wp.vec3,
    regularization: float, allow_degenerate: int,
):
    i = wp.tid()
    xi = positions[i]
    moment = Mat19(0.0)
    query = wp.hash_grid_query(grid, xi, support)
    for j in query:
        for mask in range(8):
            offset = mirror_sample(xi, positions[j], mask, lo, hi) - xi
            weight = sample_weight(offset, support)
            if weight > 0.0:
                p = cubic_basis(offset / spacing)
                for a in range(19):
                    for b in range(19):
                        moment[a, b] += weight * p[a] * p[b]
    solution, minimum_pivot, valid = solve_laplacian_fit(moment, spacing, regularization, 19)
    order = int(3)
    if valid == 0 and allow_degenerate != 0:
        solution, minimum_pivot, valid = solve_laplacian_fit(moment, spacing, regularization, 9)
        order = 2
    if valid == 0:
        wp.atomic_add(invalid, 0, 1)
        order = 0
    coefficients[i] = solution
    pivots[i] = minimum_pivot
    fit_orders[i] = order


@wp.kernel
def evaluate_laplacian(
    grid: wp.uint64, positions: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=Vec19), concentration: wp.array(dtype=float),
    laplacian: wp.array(dtype=float), max_row_norm: wp.array(dtype=float),
    spacing: float, support: float, lo: wp.vec3, hi: wp.vec3,
):
    i = wp.tid()
    xi = positions[i]
    fit = coefficients[i]
    result = float(0.0)
    off_diagonal_norm = float(0.0)
    row_sum = float(0.0)
    query = wp.hash_grid_query(grid, xi, support)
    for j in query:
        for mask in range(8):
            offset = mirror_sample(xi, positions[j], mask, lo, hi) - xi
            weight = sample_weight(offset, support)
            if weight > 0.0 and j != i:
                coefficient = weight * wp.dot(fit, cubic_basis(offset / spacing))
                result += coefficient * (concentration[j] - concentration[i])
                off_diagonal_norm += wp.abs(coefficient)
                row_sum += coefficient
    laplacian[i] = result
    wp.atomic_max(max_row_norm, 0, off_diagonal_norm + wp.abs(row_sum))


@wp.kernel
def explicit_diffusion(concentration: wp.array(dtype=float),
                       laplacian: wp.array(dtype=float), diffusion_dt: float):
    i = wp.tid()
    concentration[i] += diffusion_dt * laplacian[i]


@wp.kernel
def evaluate_conservative_laplacian(
    grid: wp.uint64, positions: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=Vec19), fit_orders: wp.array(dtype=int),
    concentration: wp.array(dtype=float), laplacian: wp.array(dtype=float),
    max_row_norm: wp.array(dtype=float), spacing: float, support: float,
):
    """Positive symmetric pair flux for equal-volume particles.

    This optional free-surface stabilization changes the original collocation
    stencil. It preserves total dye and bounds without clipping concentration.
    A particle with an unresolved fit does not exchange dye with its neighbors.
    """
    i = wp.tid()
    result = float(0.0)
    row = float(0.0)
    if fit_orders[i] != 0:
        query = wp.hash_grid_query(grid, positions[i], support)
        for j in query:
            if j != i and fit_orders[j] != 0:
                offset = positions[j]-positions[i]
                w = sample_weight(offset, support)
                a = wp.dot(coefficients[i], cubic_basis(offset/spacing))
                b = wp.dot(coefficients[j], cubic_basis(-offset/spacing))
                conductance = wp.max(0.0, 0.5*w*(a+b))
                result += conductance*(concentration[j]-concentration[i])
                row += conductance
    laplacian[i] = result
    wp.atomic_max(max_row_norm, 0, 2.0*row)


class LSMPSDiffusion:
    """Explicit Euler diffusion on a fixed-volume, all-active particle set.

    Call step at the old particle positions *before* the APIC update, matching
    the explicit ordering in Fig. 5. APIC supplies advection and its own
    particle separation; the scalar module never changes particle positions.
    """

    def __init__(self, count, spacing, bound_lo, bound_hi, device,
                 diffusivity=1.0e-3, regularization=1.0e-6,
                 allow_degenerate=False, conservative=False):
        values = (spacing, diffusivity, regularization, *bound_lo, *bound_hi)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("LSMPS parameters must be finite")
        if count < 1 or spacing <= 0 or diffusivity < 0 or regularization < 0:
            raise ValueError("count/spacing must be positive; diffusivity/regularization nonnegative")
        self.count, self.capacity, self.spacing = count, count, spacing
        self.allow_degenerate, self.conservative = allow_degenerate, conservative
        self.support = 2.5 * spacing
        if min(b-a for a, b in zip(bound_lo, bound_hi)) <= 2.0 * self.support:
            raise ValueError("Each box extent must exceed 5 particle spacings for mirror support")
        self.lo, self.hi = wp.vec3(*bound_lo), wp.vec3(*bound_hi)
        self.device = wp.get_device(device)
        self.diffusivity, self.regularization = diffusivity, regularization
        dims = [max(4, math.ceil((b-a) / self.support)) for a, b in zip(bound_lo, bound_hi)]
        self.grid = wp.HashGrid(*dims, device=self.device)
        self.coefficients = wp.empty(count, dtype=Vec19, device=self.device)
        self.pivots = wp.empty(count, dtype=float, device=self.device)
        self.fit_orders = wp.empty(count, dtype=int, device=self.device)
        self._laplacian = wp.empty(count, dtype=float, device=self.device)
        self._max_row_norm = wp.zeros(1, dtype=float, device=self.device)
        self._invalid = wp.zeros(1, dtype=int, device=self.device)
        self.positions = None
        self.last_substeps = 0
        self.last_min_pivot = 0.0
        self.last_skipped = self.last_quadratic = 0

    def resize(self, count):
        """Select an active prefix within the constructor's reserved capacity."""
        if count < 0 or count > self.capacity:
            raise ValueError("LSMPS active count exceeds reserved capacity")
        self.count = count
        self.positions = None

    def prepare(self, positions):
        if positions.shape[0] != self.count:
            raise ValueError("Particle count changed")
        self.positions = positions
        if self.count == 0:
            self.last_min_pivot = 0.0
            return
        self.grid.build(points=positions, radius=self.support)
        self._invalid.zero_()
        wp.launch(build_laplacian_fit, dim=self.count,
                  inputs=[self.grid.id, positions, self.coefficients, self.pivots, self.fit_orders,
                          self._invalid, self.spacing, self.support, self.lo, self.hi,
                          self.regularization, int(self.allow_degenerate)], device=self.device)
        self.last_skipped = int(self._invalid.numpy()[0])
        if self.last_skipped and not self.allow_degenerate:
            raise RuntimeError("Singular LSMPS neighborhood; reduce advection timestep or improve particle distribution")
        orders = self.fit_orders.numpy()[:self.count]
        self.last_quadratic = int((orders == 2).sum())
        valid_pivots = self.pivots.numpy()[:self.count][orders != 0]
        self.last_min_pivot = float(valid_pivots.min()) if len(valid_pivots) else 0.0

    def laplacian(self, concentration):
        if self.positions is None:
            raise RuntimeError("Call prepare before evaluating diffusion")
        self._max_row_norm.zero_()
        if self.conservative:
            wp.launch(evaluate_conservative_laplacian, dim=self.count,
                      inputs=[self.grid.id, self.positions, self.coefficients, self.fit_orders,
                              concentration, self._laplacian, self._max_row_norm,
                              self.spacing, self.support], device=self.device)
        else:
            wp.launch(evaluate_laplacian, dim=self.count,
                      inputs=[self.grid.id, self.positions, self.coefficients, concentration,
                              self._laplacian, self._max_row_norm, self.spacing, self.support,
                              self.lo, self.hi], device=self.device)
        return self._laplacian[:self.count]

    def step(self, positions, concentration, dt):
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.last_substeps = 0
        if self.diffusivity == 0.0 or self.count == 0:
            return
        self.prepare(positions)
        self.laplacian(concentration)
        rate = float(self._max_row_norm.numpy()[0])
        if not math.isfinite(rate):
            raise RuntimeError("Non-finite LSMPS operator")
        # Conservative timestep scale based on the actual stencil row norm.
        # Cubic collocation does not guarantee a discrete maximum principle.
        steps = max(1, math.ceil(dt * self.diffusivity * rate / 0.5))
        if steps > 256:
            raise RuntimeError("LSMPS needs >256 diffusion substeps; reduce diffusivity or timestep")
        for step in range(steps):
            if step:
                self.laplacian(concentration)
            wp.launch(explicit_diffusion, dim=self.count,
                      inputs=[concentration, self._laplacian, self.diffusivity * dt / steps],
                      device=self.device)
        self.last_substeps = steps

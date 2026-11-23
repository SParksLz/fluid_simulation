"""Volume-preserving ellipsoidal render kernels fitted to particle neighbors.

The neighborhood PCA and center smoothing are inspired by Yu and Turk (2010).
The renderer intersects the fitted ellipsoids analytically. Sparse droplets
retain spherical kernels.
"""

import warp as wp


@wp.kernel
def prepare_render_positions(positions: wp.array(dtype=wp.vec3), radii: wp.array(dtype=float),
                             render_positions: wp.array(dtype=wp.vec3),
                             grid_positions: wp.array(dtype=wp.vec3),
                             inverse_kernels: wp.array(dtype=wp.mat33), radius_scale: float):
    i = wp.tid()
    point = positions[i]
    render_positions[i] = point
    radius = radii[i] * radius_scale
    inverse_kernels[i] = wp.identity(n=3, dtype=float) / wp.max(radius, 1.0e-12)
    valid = (wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])
             and wp.isfinite(radius) and radius > 0.0 and wp.max(wp.abs(point)) < 1.0e6)
    grid_positions[i] = point if valid else wp.vec3(1.0e6)


@wp.kernel
def fit_render_kernels(
    grid: wp.uint64, positions: wp.array(dtype=wp.vec3), radii: wp.array(dtype=float),
    render_positions: wp.array(dtype=wp.vec3), inverse_kernels: wp.array(dtype=wp.mat33),
    radius_scale: float,
):
    i = wp.tid()
    point = positions[i]
    radius = radii[i]
    if (not wp.isfinite(radius) or radius <= 0.0
            or not wp.isfinite(point[0]) or not wp.isfinite(point[1]) or not wp.isfinite(point[2])
            or wp.max(wp.abs(point)) >= 1.0e6):
        return
    support = 5.0 * radius
    total = float(0.0)
    mean = wp.vec3(0.0)
    moment = wp.mat33(0.0)
    for j in wp.hash_grid_query(grid, point, support):
        offset = (positions[j] - point) / radius
        distance = wp.length(offset)
        if distance < 5.0 and radii[j] > 0.0 and wp.isfinite(radii[j]):
            weight = 1.0 - wp.pow(distance/5.0, 3.0)
            total += weight
            mean += weight * offset
            moment += weight * wp.outer(offset, offset)
    # The weight sum varies continuously when a neighbor enters the support.
    # A hard particle-count switch would make sparse silhouettes pop in time.
    confidence = wp.clamp((total-3.0)/6.0, 0.0, 1.0)
    if confidence <= 0.0:
        return
    mean /= total
    covariance = moment/total - wp.outer(mean, mean)
    covariance += wp.identity(n=3, dtype=float) * 0.01
    rotation, values, unused_right = wp.svd3(covariance)
    largest = wp.max(values[0], wp.max(values[1], values[2]))
    # Use covariance eigenvalues as kernel stretches, as in Yu and Turk's
    # reconstruction. Limit the principal-axis ratio to four, then normalize
    # the product of stretches to one to retain the original sphere volume.
    axes = wp.vec3(0.0)
    for axis in range(3):
        axes[axis] = wp.max(values[axis], largest/4.0)
    geometric_mean = wp.pow(axes[0]*axes[1]*axes[2], 1.0/3.0)
    stretches = wp.vec3(0.0)
    for axis in range(3):
        stretches[axis] = wp.exp(confidence*wp.log(axes[axis]/geometric_mean))
    inverse_axes = wp.vec3(1.0/stretches[0], 1.0/stretches[1], 1.0/stretches[2])
    inverse_kernels[i] = (rotation * wp.diag(inverse_axes) * wp.transpose(rotation)
                          / (radius_scale*radius))
    displacement = 0.2 * confidence * radius * mean
    length = wp.length(displacement)
    if length > 0.25*radius:
        displacement *= 0.25*radius/length
    render_positions[i] = point + displacement

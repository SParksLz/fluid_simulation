import warp as wp
import numpy as np


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def fifth(x: float):
    return x * x * x * x * x

@wp.func
def get_cubic(r_norm : float, radius : float):
    """
    Cubic spline smoothing kernel.
    """
    res = 0.0
    h = radius
    k = 8.0 / (wp.pi * cube(h))
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = square(q)
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1.0)
        else:
            res = 2.0 * k * cube((1.0 - q))
            # res = cube(res)
    return res

# @wp.kernel
# def get_v_max(
#         particle_v : wp.array(dtype=wp.vec3),
#         v_max : float,
#     ):
#     tid = wp.tid()
#     wp.atomic_max(particle_v, 0, wp.length(particle_v[tid]))

@wp.func
def get_cubic_derivative(r: wp.vec3, smoothing_length: float):
    """
        Derivative of cubic spline smoothing kernel.
    """
    # res = ti.Vector.zero(gs.ti_float, 3)
    res = wp.vec3(0.0, 0.0, 0.0)
    

    # r_norm = radius.norm()
    r_norm = wp.length(r)
    h = smoothing_length
    k = 8.0 / (wp.pi * cube(h))
    q = r_norm / h
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = 6.0 * k * q * (3.0 * q - 2.0) * grad_q
        else:
            q_term = 1.0 - q
            res = -6.0 * k * q_term * q_term * grad_q
    return res

@wp.func
def density_kernel(xyz: wp.vec3, smoothing_length: float, volume: float):
    # calculate distance

    distance = wp.sqrt(wp.dot(xyz, xyz))
    # distance = wp.length(xyz)
    return volume * get_cubic(distance, smoothing_length)

# @wp.func
# def non_pressure(mass: float, gamma:) :

@wp.kernel
def rho(
        particle_rho: wp.array(dtype=float), 
        exponent: wp.array(dtype=float),
        stiffness: wp.array(dtype=float),
        particle_x: wp.array(dtype=wp.vec3),
        rest_density: wp.array(dtype=float),
        volume: float, 
        smoothing_length: float,
        grid_id: wp.uint64,
        pressure: wp.array(dtype=float),
    ) :
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)
    
    # particle rho will multiply by rest_density at last because the origin term is mass
    # particle_rho[i] = volume * get_cubic(0.0, smoothing_length) # wp.hash_grid_query will detect self
    x = particle_x[i]
    rho_0 = rest_density[i]


    rho_temp = float(0.0)
    neighbors = wp.hash_grid_query(grid_id, x, smoothing_length)
    for index in neighbors :
        rho_0_nei = rest_density[index]
        distance = x - particle_x[index]
        r_norm = wp.length(distance)
        mass_nei = rho_0_nei * volume
        rho_temp += mass_nei * get_cubic(r_norm, smoothing_length)
        # rho_temp += density_kernel(distance, smoothing_length, volume)
    # particle_rho[i] = rest_density * rho_temp
    particle_rho[i] = rho_temp
    # particle_rho[i] += rho_temp
    # particle_rho[i] *= rho_0
    # particle_rho[i] = wp.max(particle_rho[i], rho_0)

    exp = exponent[i]
    stiff = stiffness[i]


    # get pressure by the way 
    pressure[i] = stiff * (wp.pow(particle_rho[i] / rho_0, exp) - 1.0)

# @wp.func
# def cal_pressure():


@wp.func
def cal_acc_with_non_pressure(
    a: wp.vec3,
    # rho: float,
    rho_nei: float,
    vel: wp.vec3,
    vel_nei: wp.vec3,
    mass: float,
    mass_nei: float,
    mu: float,
    gamma: float,
    d_current_nei: wp.vec3,
    e_dist: float,
    distance: float,
    smoothing_length: float,
    
):
    a -= gamma / mass * mass_nei * d_current_nei * get_cubic(e_dist, smoothing_length)
    v_current_nei = wp.dot((vel - vel_nei), d_current_nei)
    d = 2.0 * (3.0 + 2.0)

    f_v = (d 
        * mu
        * (mass_nei / rho_nei) 
        * v_current_nei 
        / (distance * distance + 0.01 * (smoothing_length * smoothing_length)) 
        * get_cubic_derivative(d_current_nei, smoothing_length)
    )
    a += f_v

    return a


@wp.func
def cal_acc_with_pressure(
    a: wp.vec3,
    volume: float,
    d: wp.vec3,
    pressure: float,
    pressure_nei: float,
    rho_0: float,
    rho_0_nei: float,
    rho: float,
    rho_nei: float,
    smoothing_length: float,
):
    dp_i = pressure / (rho * rho)
    # rho_nei = rho_nei * rho_0 / rho_0
    dp_nei = pressure_nei / (rho_nei * rho_nei)
    a += (
        -rho_0_nei
        * volume
        * (dp_i + dp_nei)
        * get_cubic_derivative(d, smoothing_length)
    )
    return a


@wp.kernel
def acceleration(
    particle_x : wp.array(dtype=wp.vec3),
    particle_v : wp.array(dtype=wp.vec3),
    particle_rho_0: wp.array(dtype=float),
    particle_rho : wp.array(dtype=float),
    particle_a : wp.array(dtype=wp.vec3),
    particle_size: float,
    particle_stiffness: wp.array(dtype=float),
    particle_exponent: wp.array(dtype=float),
    particle_pressure: wp.array(dtype=float),
    volume: float,
    mass_: wp.array(dtype=float),
    gamma_: wp.array(dtype=float),
    mu_: wp.array(dtype=float),
    gravity: float,
    smoothing_length : float,
    grid_id : wp.uint64,
) :
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)
    rho = particle_rho[i]
    x = particle_x[i]
    a = particle_a[i]
    mass = mass_[i]
    gamma = gamma_[i]
    mu = mu_[i]

    # data from material
    rho_0 = particle_rho_0[i]
    pressure = particle_pressure[i]
    stiffness = particle_stiffness[i]
    exponent = particle_exponent[i]

    acc = wp.vec3(0.0, 0.0, 0.0)

    neighbors = wp.hash_grid_query(grid_id, x, smoothing_length)
    for index in neighbors :
        if index != i :
            nei_x = particle_x[index]
            dir_current_nei = x - nei_x
            # distance = wp.sqrt(wp.dot(x, nei_x))
            distance = wp.length(dir_current_nei)
            e_dist = wp.max(distance, particle_size)
            rho_nei = particle_rho[index]
            rho_0_nei = particle_rho_0[index]
            nei_mass = mass_[index]
            nei_gamma = gamma_[index]
            pressure_nei = particle_pressure[index]

            # non pressure acceleration
            acc = cal_acc_with_non_pressure(
                acc, 
                rho_nei, 
                particle_v[i], particle_v[index],  # velocity of current and nei particles
                mass_[i], mass_[index], # mass of current and nei particles
                mu,
                gamma,
                dir_current_nei, 
                e_dist, 
                distance, 
                smoothing_length)

            # pressure = stiffness * (rho - rho_0) ** exponent
            
            # pressure acceleration
            acc = cal_acc_with_pressure(
                acc,
                volume,
                dir_current_nei, # position of current and nei particles
                pressure, pressure_nei,# pressure of current and nei particles
                rho_0, rho_0_nei, # rest density of current and nei particles
                rho, rho_nei, # density of current and nei particles
                smoothing_length)

    particle_a[i] = acc + wp.vec3(0.0, 0.0, gravity)
    # particle_a[i] = acc

@wp.kernel
def kick(particle_v: wp.array(dtype=wp.vec3), particle_a: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(particle_x: wp.array(dtype=wp.vec3), particle_v: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x[tid] = x + particle_v[tid] * dt


@wp.kernel
def initialize_particles(
    particle_x: wp.array(dtype=wp.vec3), particle_distance: float, width: float, height: float, length: float, offset: wp.vec3
):
    tid = wp.tid()

    particle_spacing = particle_distance
    # grid size
    nr_x = wp.int32(width / particle_spacing)
    nr_y = wp.int32(height / particle_spacing)
    nr_z = wp.int32(length / particle_spacing)

    # calculate particle position
    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = particle_distance * wp.vec3(x, y, z) + offset

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * particle_distance * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))

    # set position
    particle_x[tid] = pos

@wp.kernel
def get_neighbor(
    grid: wp.uint64,
    pt_array: wp.array(dtype=wp.vec3),
    smoothing_length: float,
    nei_count: wp.array(dtype=wp.int32),
) :
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    x = pt_array[i]
    nei = wp.hash_grid_query(grid, x, smoothing_length)
    # count = 0
    nei_count[i] = 0
    for index in nei :
        # nei_x = pt_array[index]
        nei_count[i] += 1
        # print(index)
    # nei_count[i] = count

@wp.kernel
def apply_bounds(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    size: wp.vec3,
    damping_coef: float,
):
    tid = wp.tid()

    # get pos and velocity
    x = particle_x[tid]
    v = particle_v[tid]


    if x[0] < -size.x:
        x = wp.vec3(-size.x, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp x right
    if x[0] > size.x:
        x = wp.vec3(size.x, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    if x[1] > size.y :
        x = wp.vec3(x[0], size.y, x[2])
        v = wp.vec3(v[0], v[1] *damping_coef, v[2])
    # clamp y bot
    if x[1] < -size.y:
        x = wp.vec3(x[0], -size.y, x[2])
        v = wp.vec3(v[0], v[1] * damping_coef, v[2])

    # clamp z left
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # clamp z right
    if x[2] > size.z * 2.0:
        x = wp.vec3(x[0], x[1], size.z * 2.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # apply clamps
    particle_x[tid] = x
    particle_v[tid] = v


# for dfsph
@wp.kernel
def compute_factor(
        grid_id: wp.uint64,
        x: wp.array(dtype=wp.vec3),
        smoothing_length: float,
        volume: float,
        factor: wp.array(dtype=float),
    ) :
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_id, tid)
    current_x = x[i]

    neighbors = wp.hash_grid_query(grid_id, current_x, smoothing_length)

    grad_p_i = wp.vec3(0.0, 0.0, 0.0)
    sum_grad_p_k = wp.float(0.0)
    ret = wp.vec4(0.0, 0.0, 0.0, 0.0)  # 0-2 ：gradj 3: sum_grad_pressure_k 


    for index in neighbors :
        if index != i :
            nei_x = x[index]
            d = current_x - nei_x
            
            grad_j = -volume * get_cubic_derivative(d, smoothing_length)
            ret.w += wp.length(grad_j) * wp.length(grad_j)
            # ret.xyz -= grad_j
            ret.x -= grad_j.x
            ret.y -= grad_j.y
            ret.z -= grad_j.z
    sum_grad_p_k = ret.w
    grad_p_i = wp.vec3(ret.x, ret.y, ret.z)

    sum_grad_p_k += wp.length(grad_p_i) * wp.length(grad_p_i)

    factor_temp = float(0.0)
    if sum_grad_p_k > 1e-6:
        factor_temp = -1.0 / sum_grad_p_k
    else :
        factor_temp = 0.0
    factor[i] = factor_temp



#utils
@wp.func
def scale_position(
    particle_x: wp.vec3,
    scale: float,
    offset: wp.vec3,
) :
    z_temp = float(0.0)
    x = particle_x
    # z_temp = x.z
    # x.z = x.y
    # x.y = z_temp
    # x.x *= -1.0
    x *= scale
    x += offset

    particle_x = x
    return particle_x

@wp.kernel
def to_micro_world(
    position: wp.array(dtype=wp.vec3),
    scale: float,
    offset: wp.vec3,
):
    tid = wp.tid()
    # current_position = wp.vec3(0.0, 0.0, 0.0)
    current_position = position[tid]
    new_position = scale_position(current_position, scale, offset)
    position[tid] = new_position
@wp.kernel
def to_real_world(
    micro_position: wp.array(dtype=wp.vec3),
    real_position : wp.array(dtype=wp.vec3),
    scale: float,
    offset: wp.vec3,
):
    tid = wp.tid()
    micro_x = micro_position[tid]
    micro_x -= offset
    micro_x *= scale
    real_position[tid] = micro_x





# collision

@wp.kernel(enable_backward=False)
def update_collider_with_tri_mesh(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    # cd: wp.array(dtype=int),
    # mesh_transform: wp.mat44,
    mesh_id: wp.uint64,
    radius: float,
    restitution: float,
    friction: float,
):
    tid = wp.tid()
    particle_pos = x[tid]
    particle_vel = v[tid]

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)


    # 查询 mesh
    res = wp.mesh_query_point_sign_normal(mesh_id, particle_pos, radius, sign, face_index, face_u, face_v)
    # q = wp.mesh_query_point(mesh_id, p, radius)
    if res:
        # cd[tid] = 1

        mesh_pos = wp.mesh_eval_position(mesh_id, face_index, face_u, face_v)
        # mesh_normal = wp.mesh_eval_face_normal(mesh_id, face_index, )
        # mesh_vel = wp.mesh_eval_velocity(mesh_id, face_index, face_u, face_v)

        delta = particle_pos - mesh_pos
        n = wp.normalize(delta) * sign
        v_length = wp.length(particle_vel)
        particle_vel = (particle_vel - n * v_length) * restitution

        # rel_vel = particle_vel - mesh_vel

        # vn = wp.dot(rel_vel, n)
        # vt = rel_vel - n * vn

        

    x[tid] = particle_pos
    v[tid] = particle_vel

# =========================sph->mesh============================

@wp.kernel
def rasterize_particles_to_nvdb_volume(
    particles: wp.array(dtype=wp.vec3),
    volume: wp.uint64,  # wp.Volume 的 id（用于坐标转换）
    density_buffer: wp.array3d(dtype=float),  # 临时累积缓冲区
    smoothing_length: float,
    p_volume: float,
    resolution_x: int,
    resolution_y: int,
    resolution_z: int,
    debug_info: wp.array(dtype=wp.vec3)  # 调试信息：[0]=第一个粒子的体素索引, [1]=第一个粒子的位置, [2]=体素大小
):
    tid = wp.tid()
    pos = particles[tid]
    
    # 使用 Volume 的坐标转换函数将世界坐标转换为体素索引
    voxel_index_f = wp.volume_world_to_index(volume, pos)
    voxel_center = wp.vec3i(
        int(voxel_index_f[0]),
        int(voxel_index_f[1]),
        int(voxel_index_f[2])
    )
    
    # 调试：记录第一个粒子的信息
    if tid == 0:
        debug_info[0] = voxel_index_f  # 体素索引（浮点）
        debug_info[1] = pos  # 粒子位置
        voxel_size_vec = wp.volume_index_to_world_dir(volume, wp.vec3(1.0, 0.0, 0.0))
        debug_info[2] = voxel_size_vec  # 体素大小向量
    
    # 获取体素大小（用于计算影响范围）
    voxel_size_vec = wp.volume_index_to_world_dir(volume, wp.vec3(1.0, 0.0, 0.0))
    voxel_size = wp.length(voxel_size_vec)
    
    # 计算影响的体素范围
    kernel_size = int(smoothing_length / voxel_size) + 1
    
    # 调试：记录第一个粒子影响的体素数量
    affected_count = int(0)
    
    for dx in range(-kernel_size, kernel_size + 1):
        for dy in range(-kernel_size, kernel_size + 1):
            for dz in range(-kernel_size, kernel_size + 1):
                vx = voxel_center[0] + dx
                vy = voxel_center[1] + dy
                vz = voxel_center[2] + dz
                
                # 检查边界（使用 density_buffer 的分辨率）
                if 0 <= vx < resolution_x and 0 <= vy < resolution_y and 0 <= vz < resolution_z:
                    # 计算体素中心的世界坐标
                    voxel_center_world = wp.volume_index_to_world(volume, wp.vec3(float(vx), float(vy), float(vz)))
                    distance = wp.length(pos - voxel_center_world)
                    
                    if distance < smoothing_length:
                        weight = get_cubic(distance, smoothing_length)
                        density_contribution = weight * p_volume
                        # 使用原子操作累加到 density_buffer
                        wp.atomic_add(density_buffer, vx, vy, vz, density_contribution)
                        affected_count += 1
@wp.kernel
def copy_density_to_volume(
    density_buffer: wp.array3d(dtype=float),
    volume: wp.uint64,
    resolution_x: int,
    resolution_y: int,
    resolution_z: int
):
    i, j, k = wp.tid()
    if i < resolution_x and j < resolution_y and k < resolution_z:
        density_value = density_buffer[i, j, k]
        if density_value > 0.0:  # 只写入非零值（Volume 是稀疏的）
            wp.volume_store_f(volume, i, j, k, density_value)


@wp.kernel
def write_sphere(volume: wp.uint64, center: wp.vec3, radius: float):
    i, j, k = wp.tid()

    p = wp.volume_index_to_world(volume, wp.vec3f(float(i), float(j), float(k)))
    d = wp.length(p - center) - radius

    wp.volume_store_f(volume, i, j, k, -d)
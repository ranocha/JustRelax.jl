## CPU

# manually vectorized version of trilinear kernel
function _vscattering(p::NTuple{3,A}, xi::NTuple{3,B}, F::Array{Float64,3}) where {A,B}
    # unpack tuples
    dx, dy, dz = @. 1 / (x[2] - x[1], y[2] - y[1], z[2] - z[1])
    px, py, pz = p
    x, y, z = xi

    # indices of lowermost-left corner of the cell 
    # containing the particle
    idx_x, idx_y, idx_z = parent_cell(p, dxi)

    # distance from particle to lowermost-left corner of the cell 
    dx_particle = px - x[idx_x]
    dy_particle = py - y[idx_y]
    dz_particle = pz - z[idx_z]

    a1 = VectorizationBase.Vec(
        (
            F[idx_x + 1, idx_y, idx_z], # 3
            F[idx_x + 1, idx_y + 1, idx_z], # 4
            F[idx_x, idx_y + 1, idx_z], # 2
            F[idx_x, idx_y, idx_z], # 1
        )...,
    )
    a2 = VectorizationBase.Vec(
        (
            F[idx_x + 1, idx_y, idx_z + 1], # 3
            F[idx_x + 1, idx_y + 1, idx_z + 1], # 4
            F[idx_x, idx_y + 1, idx_z + 1], # 2
            F[idx_x, idx_y, idx_z + 1], # 1
        )...,
    )

    b1 = VectorizationBase.Vec(
        (
            (1 - dx_particle * dx) * (dy_particle * dy) * (1 - dz_particle * dz),     # 3
            (dx_particle * dx) * (dy_particle * dy) * (1 - dz_particle * dz),       # 4
            (dx_particle * dx) * (1 - dy_particle * dy) * (1 - dz_particle * dz),     # 2
            (1 - dx_particle * dx) * (1 - dy_particle * dy) * (1 - dz_particle * dz),   # 1
        )...,
    )
    b2 = VectorizationBase.Vec(
        (
            (1 - dx_particle * dx) * (dy_particle * dy) * (dz_particle * dz),   # 3
            (dx_particle * dx) * (dy_particle * dy) * (dz_particle * dz),     # 4
            (dx_particle * dx) * (1 - dy_particle * dy) * (dz_particle * dz),   # 2
            (1 - dx_particle * dx) * (1 - dy_particle * dy) * (dz_particle * dz), # 1
        )...,
    )

    # Interpolate field F onto particle
    Fp =
        VectorizationBase.vsum(VectorizationBase.vmul(a1, b1)) +
        VectorizationBase.vsum(VectorizationBase.vmul(a2, b2))

    return Fp
end

function _vscattering(p::NTuple{3,A}, xi::NTuple{3,B}, F::Array{Float32,3}) where {A,B}
    # unpack tuples
    dx, dy, dz = @. 1 / (x[2] - x[1], y[2] - y[1], z[2] - z[1])
    px, py, pz = p
    x, y, z = xi

    # indices of lowermost-left corner of the cell 
    # containing the particle
    idx_x, idx_y, idx_z = parent_cell(p, dxi)

    # distance from particle to lowermost-left corner of the cell 
    dx_particle = px - x[idx_x]
    dy_particle = py - y[idx_y]
    dz_particle = pz - z[idx_z]

    a = VectorizationBase.Vec(
        (
            F[idx_x + 1, idx_y, idx_z], # 3
            F[idx_x + 1, idx_y + 1, idx_z], # 4
            F[idx_x, idx_y + 1, idx_z], # 2
            F[idx_x, idx_y, idx_z], # 1
            F[idx_x + 1, idx_y, idx_z + 1], # 3
            F[idx_x + 1, idx_y + 1, idx_z + 1], # 4
            F[idx_x, idx_y + 1, idx_z + 1], # 2
            F[idx_x, idx_y, idx_z + 1], # 1
        )...,
    )

    b = VectorizationBase.Vec(
        (
            (1 - dx_particle * dx) * (dy_particle * dy) * (1 - dz_particle * dz),     # 3
            (dx_particle * dx) * (dy_particle * dy) * (1 - dz_particle * dz),       # 4
            (dx_particle * dx) * (1 - dy_particle * dy) * (1 - dz_particle * dz),     # 2
            (1 - dx_particle * dx) * (1 - dy_particle * dy) * (1 - dz_particle * dz),   # 1
            (1 - dx_particle * dx) * (dy_particle * dy) * (dz_particle * dz),       # 3
            (dx_particle * dx) * (dy_particle * dy) * (dz_particle * dz),         # 4
            (dx_particle * dx) * (1 - dy_particle * dy) * (dz_particle * dz),       # 2
            (1 - dx_particle * dx) * (1 - dy_particle * dy) * (dz_particle * dz),     # 1
        )...,
    )

    # Interpolate field F onto particle
    Fp = VectorizationBase.vsum(VectorizationBase.vmul(a, b))

    return Fp
end

function _scattering(
    p::NTuple{3,A}, dxi::NTuple{3,A}, xi::NTuple{3,B}, F::Array{C,3}
) where {A,B,C}
    # unpack tuples
    dx, dy, dz = dxi
    px, py, pz = p
    x, y, z = xi

    # check that the particle is inside the grid
    isinside(px, py, pz, x, y, z)

    # indices of lowermost-left corner of the cell 
    # containing the particle
    idx_x, idx_y, idx_z = parent_cell(p, dxi)

    # distance from particle to lowermost-left corner of the cell 
    tx = (px - x[idx_x]) / dx
    ty = (py - y[idx_y]) / dy
    tz = (pz - z[idx_z]) / dz

    # Interpolate field F onto particle
    Fp = trilinear(
        tx,
        ty,
        tz,
        F[idx_x, idx_y, idx_z],   # v000
        F[idx_x + 1, idx_y, idx_z],   # v100
        F[idx_x, idx_y, idx_z + 1], # v001
        F[idx_x + 1, idx_y, idx_z + 1], # v101
        F[idx_x, idx_y + 1, idx_z],   # v010
        F[idx_x + 1, idx_y + 1, idx_z],   # v110
        F[idx_x, idx_y + 1, idx_z + 1], # v011
        F[idx_x + 1, idx_y + 1, idx_z + 1], # v111
    )
    return Fp
end

function scattering(xi, F::Array{T,3}, particle_coords) where {T}
    # unpack tuples
    dxi = grid_size(xi)
    np = length(particle_coords[1])
    Fp = zeros(T, np)

    Threads.@threads for i in 1:np
        @inbounds Fp[i] = _scattering(
            (particle_coords[1][i], particle_coords[2][i], particle_coords[3][i]),
            dxi,
            xi,
            F,
        )
    end

    return Fp
end

function scattering!(Fp, xi, F::Array{T,3}, particle_coords) where {T}
    # unpack tuples
    x, y, z = xi
    dxi = (x[2] - x[1], y[2] - y[1], z[2] - z[1])
    np = length(particle_coords[1])

    Threads.@threads for i in 1:np
        @inbounds Fp[i] = _scattering(
            (particle_coords[1][i], particle_coords[2][i], particle_coords[3][i]),
            dxi,
            xi,
            F,
        )
    end
end

## CUDA

function _scattering!(
    Fp::CuDeviceVector{T,1},
    p::NTuple{3,CuDeviceVector{T,1}},
    dxi::NTuple{3,T},
    xi::NTuple{3,A},
    F::CuDeviceArray{T,3},
) where {A,T}
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # unpack tuples
    dx, dy, dz = dxi
    px, py, pz = p
    x, y, z = xi

    @inbounds if ix ≤ length(px)

        # check that the particle is inside the grid
        isinside(px[ix], py[ix], pz[ix], x, y, z)

        # indices of lowermost-left corner of the cell 
        # containing the particle
        idx_x, idx_y, idx_z = parent_cell((px[ix], py[ix], pz[ix]), dxi)

        # # distance from particle to lowermost-left corner of the cell 
        tx = (px[ix] - x[idx_x]) / dx
        ty = (py[ix] - y[idx_y]) / dy
        tz = (pz[ix] - z[idx_z]) / dz

        # Interpolate field F onto particle
        Fp[ix] = trilinear(
            tx,
            ty,
            tz,
            F[idx_x, idx_y, idx_z],   # v000
            F[idx_x + 1, idx_y, idx_z],   # v100
            F[idx_x, idx_y, idx_z + 1], # v001
            F[idx_x + 1, idx_y, idx_z + 1], # v101
            F[idx_x, idx_y + 1, idx_z],   # v010
            F[idx_x + 1, idx_y + 1, idx_z],   # v110
            F[idx_x, idx_y + 1, idx_z + 1], # v011
            F[idx_x + 1, idx_y + 1, idx_z + 1], # v111
        )
    end

    return nothing
end

function scattering(
    xi, Fd::CuArray{T,3}, particle_coords::NTuple{3,CuArray}; nt=512
) where {T}
    x, y, z = xi
    dxi = (x[2] - x[1], y[2] - y[1], z[2] - z[1])
    N = length(particle_coords[1])
    Fpd = CuArray{T,1}(undef, N)

    numblocks = ceil(Int, N / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _scattering!(
            Fpd, particle_coords, dxi, xi, Fd
        )
    end

    return Fpd
end

function scattering!(
    Fpd, xi, Fd::CuArray{T,3}, particle_coords::NTuple{3,CuArray}; nt=512
) where {T}
    x, y, z = xi
    dxi = (x[2] - x[1], y[2] - y[1], z[2] - z[1])
    N = length(particle_coords[1])
    numblocks = ceil(Int, N / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _scattering!(
            Fpd, particle_coords, dxi, xi, Fd
        )
    end
end

## CPU

function _scattering(p::NTuple{2,A}, xi::NTuple{2,B}, F::Array{C,2}) where {A,B,C}
    # unpack tuples
    x, y = xi
    dx, dy = x[2] - x[1], y[2] - y[1]
    px, py = p

    # check that the particle is inside the grid
    isinside(px, py, x, y)

    # indices of lowermost-left corner of the cell 
    # containing the particle
    idx_x, idx_y = parent_cell(p, dxi)

    # normalize particle coordinates
    tx = (px - x[idx_x]) / dx
    ty = (py - y[idx_y]) / dy

    # Interpolate field F onto particle
    Fp = bilinear(
        tx,
        ty,
        F[idx_x, idx_y],
        F[idx_x + 1, idx_y],
        F[idx_x, idx_y + 1],
        F[idx_x + 1, idx_y + 1],
    )

    return Fp
end

function scattering(xi, F::Array{T,2}, particle_coords) where {T}
    np = length(particle_coords[1])
    Fp = zeros(T, np)

    Threads.@threads for i in 1:np
        @inbounds Fp[i] = _scattering((particle_coords[1][i], particle_coords[2][i]), xi, F)
    end

    return Fp
end

##  CUDA

function _scattering!(
    Fp::CuDeviceVector{T,1}, p::NTuple{2,A}, xi::NTuple{2,B}, F::CuDeviceMatrix{T,1}
) where {A,B,T}
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # unpack tuples
    dx, dy = x[2] - x[1], y[2] - y[1]
    px, py = p
    x, y = xi

    @inbounds if ix ≤ length(px)
        # check that the particle is inside the grid
        isinside(px[ix], py[ix], x, y)

        # indices of lowermost-left corner of the cell 
        # containing the particle
        idx_x, idx_y = parent_cell((px[ix], py[ix]), dxi)

        # normalize particle coordinates
        tx = (px[ix] - x[idx_x]) / dx
        ty = (py[ix] - y[idx_y]) / dy

        # Interpolate field F onto particle
        Fp[ix] = bilinear(
            tx,
            ty,
            F[idx_x, idx_y],
            F[idx_x + 1, idx_y],
            F[idx_x, idx_y + 1],
            F[idx_x + 1, idx_y + 1],
        )
    end

    return nothing
end

function scattering(
    xi, Fd::CuArray{T,2}, particle_coords::NTuple{2,CuArray}; nt=512
) where {T}
    N = length(particle_coords[1])
    Fpd = CuArray{T}(undef, N)

    numblocks = ceil(Int, N / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _scattering!(Fpd, particle_coords, xi, Fd)
    end
end

function scattering!(
    Fpd, xi, Fd::CuArray{T,2}, particle_coords::NTuple{2,CuArray}; nt=512
) where {T}
    N = length(particle_coords[1])
    numblocks = ceil(Int, N / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _scattering!(Fpd, particle_coords, xi, Fd)
    end
end

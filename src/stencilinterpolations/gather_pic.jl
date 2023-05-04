@inline function distance_weight(a::NTuple{N,T}, b::NTuple{N,T}; order=2) where {N,T}
    return inv(distance(a, b)^order)
end

@inline @generated function bilinear_weight(
    a::NTuple{N,T}, b::NTuple{N,T}, dxi::NTuple{N,T}
) where {N,T}
    quote
        val = one(T)
        Base.Cartesian.@nexprs $N i ->
            @inbounds val *= one(T) - abs(a[i] - b[i]) * inv(dxi[i])
        return val
    end
end

# CPU 

function gathering!(F::Array{T,N}, Fp::Array{T,1}, xi, particle_coords) where {T,N}
    upper = [zeros(size(F)) for _ in 1:Threads.nthreads()]
    lower = [zeros(size(F)) for _ in 1:Threads.nthreads()]

    return gathering!(F, Fp, xi, particle_coords, upper, lower)
end

## CPU 2D

@inbounds function _gathering_xvertex!(
    F, Fp, inode, jnode, xi::NTuple{2,T}, p, dxi
) where {T}
    px, py = p # particle coordinates
    nx, ny = size(F)
    xvertex = (xi[1][inode], xi[2][jnode]) # cell lower-left coordinates
    ω, ωxF = 0.0, 0.0 # init weights
    max_xcell = size(px, 1) # max particles per cell

    # iterate over cells around i-th node
    for joffset in -1:0
        jvertex = joffset + jnode
        for ioffset in -1:0
            ivertex = ioffset + inode
            # make sure we stay within the grid
            if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny)
                # iterate over cell
                @inbounds for i in 1:max_xcell
                    p_i = (px[i, ivertex, jvertex], py[i, ivertex, jvertex])
                    # ignore lines below for unused allocations
                    isnan(p_i[1]) && continue
                    ω_i = bilinear_weight(xvertex, p_i, dxi)
                    ω += ω_i
                    ωxF += ω_i * Fp[i, ivertex, jvertex]
                end
            end
        end
    end

    return F[inode, jnode] = ωxF / ω
end

function gathering_xvertex!(
    F::AbstractArray, Fp::AbstractArray, xi::NTuple{2,T}, particle_coords
) where {T}
    dxi = grid_size(xi)
    Threads.@threads for jnode in axes(F, 2)
        for inode in axes(F, 1)
            _gathering_xvertex!(F, Fp, inode, jnode, xi, particle_coords, dxi)
        end
    end
end

## CPU 3D

@inbounds function _gathering_xvertex!(
    F, Fp, inode, jnode, knode, xi::NTuple{3,T}, p, dxi
) where {T}
    px, py, pz = p # particle coordinates
    nx, ny, nz = size(F)
    xvertex = (xi[1][inode], xi[2][jnode], xi[3][knode]) # cell lower-left coordinates
    ω, ωxF = 0.0, 0.0 # init weights
    max_xcell = size(px, 1) # max particles per cell

    # iterate over cells around i-th node
    for koffset in -1:0
        kvertex = koffset + knode
        for joffset in -1:0
            jvertex = joffset + jnode
            for ioffset in -1:0
                ivertex = ioffset + inode
                # make sure we stay within the grid
                if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny) && (1 ≤ kvertex < nz)
                    # iterate over cell
                    @inbounds for i in 1:max_xcell
                        p_i = (
                            px[i, ivertex, jvertex, kvertex],
                            py[i, ivertex, jvertex, kvertex],
                            pz[i, ivertex, jvertex, kvertex],
                        )
                        # ignore lines below for unused allocations
                        isnan(p_i[1]) && continue
                        ω_i = bilinear_weight(xvertex, p_i, dxi)
                        ω += ω_i
                        ωxF += ω_i * Fp[i, ivertex, jvertex, kvertex]
                    end
                end
            end
        end
    end

    return F[inode, jnode, knode] = ωxF / ω
end

function gathering_xvertex!(
    F::AbstractArray, Fp::AbstractArray, xi::NTuple{3,T}, particle_coords
) where {T}
    dxi = grid_size(xi)
    Threads.@threads for knode in axes(F, 3)
        for jnode in axes(F, 2), inode in axes(F, 1)
            _gathering_xvertex!(F, Fp, inode, jnode, knode, xi, particle_coords, dxi)
        end
    end
end

## CUDA 2D

function gathering_xvertex!(
    F::CuArray, Fp::CuArray, xi::NTuple{2,T}, particle_coords
) where {T}
    px, = particle_coords
    dxi = grid_size(xi)

    # first kernel that computes ∑ωᵢFᵢ and ∑ωᵢ
    _, ny, nz = size(px)
    nblocksx = ceil(Int, ny / 32)
    nblocksy = ceil(Int, nz / 32)
    threadsx = 32
    threadsy = 32

    CUDA.@sync begin
        @cuda threads = (threadsx, threadsy) blocks = (nblocksx, nblocksy) _gathering_xvertex!!(
            F, Fp, xi, particle_coords, dxi
        )
    end
end

function _gathering_xvertex!!(F, Fp, xi, p, dxi::NTuple{2,T}) where {T}
    inode = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jnode = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    @inbounds if (inode ≤ length(xi[1])) && (jnode ≤ length(xi[2]))
        px, py = p # particle coordinates
        nx, ny = size(F)
        xvertex = (xi[1][inode], xi[2][jnode]) # cell lower-left coordinates
        ω, ωxF = 0.0, 0.0 # init weights
        max_xcell = size(px, 1) # max particles per cell

        # iterate over cells around i-th node
        for joffset in -1:0
            jvertex = joffset + jnode
            for ioffset in -1:0
                ivertex = ioffset + inode
                # make sure we stay within the grid
                if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny)
                    # iterate over cell
                    @inbounds for i in 1:max_xcell
                        p_i = (px[i, ivertex, jvertex], py[i, ivertex, jvertex])
                        # ignore lines below for unused allocations
                        isnan(p_i[1]) && continue
                        ω_i = bilinear_weight(xvertex, p_i, dxi)
                        ω += ω_i
                        ωxF += ω_i * Fp[i, ivertex, jvertex]
                    end
                end
            end
        end

        F[inode, jnode] = ωxF / ω
    end
    return nothing
end

## CUDA 3D

function gathering_xvertex!(
    F::CuArray, Fp::CuArray, xi::NTuple{3,T}, particle_coords
) where {T}
    px, = particle_coords
    dxi = grid_size(xi)

    # first kernel that computes ∑ωᵢFᵢ and ∑ωᵢ
    _, nx, ny, nz = size(px)
    nblocksx = ceil(Int, nx / 16)
    nblocksy = ceil(Int, ny / 16)
    nblocksz = ceil(Int, nz / 4)
    threadsx = 16
    threadsy = 16
    threadsz = 4

    CUDA.@sync begin
        @cuda threads = (threadsx, threadsy, threadsz) blocks = (
            nblocksx, nblocksy, nblocksz
        ) _gathering_xvertex!!(F, Fp, xi, particle_coords, dxi)
    end
end

function _gathering_xvertex!!(F, Fp, xi, p, dxi::NTuple{3,T}) where {T}
    inode = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jnode = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    knode = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    lx, ly, lz = length.(xi)

    @inbounds if (inode ≤ lx) && (jnode ≤ ly) && (knode ≤ lz)
        px, py, pz = p # particle coordinates
        nx, ny, nz = size(F)
        xvertex = (xi[1][inode], xi[2][jnode], xi[3][knode]) # cell lower-left coordinates
        ω, ωxF = 0.0, 0.0 # init weights
        max_xcell = size(px, 1) # max particles per cell

        # iterate over cells around i-th node
        for koffset in -1:0
            kvertex = koffset + knode
            for joffset in -1:0
                jvertex = joffset + jnode
                for ioffset in -1:0
                    ivertex = ioffset + inode
                    # make sure we stay within the grid
                    if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny) && (1 ≤ kvertex < nz)
                        # iterate over cell
                        @inbounds for i in 1:max_xcell
                            p_i = (
                                px[i, ivertex, jvertex, kvertex],
                                py[i, ivertex, jvertex, kvertex],
                                pz[i, ivertex, jvertex, kvertex],
                            )
                            # ignore lines below for unused allocations
                            isnan(p_i[1]) && continue
                            ω_i = bilinear_weight(xvertex, p_i, dxi)
                            ω += ω_i
                            ωxF += ω_i * Fp[i, ivertex, jvertex, kvertex]
                        end
                    end
                end
            end
        end

        F[inode, jnode, knode] = ωxF / ω
    end
    return nothing
end

##
@inbounds function _gathering!(upper, lower, Fpi, p, xci, x, y, dxi, order)

    # check that the particle is inside the grid
    # isinside(p, (x, y))

    # indices of lowermost-left corner of   
    # the cell containing the particlex
    idx_x, idx_y = parent_cell(p, dxi, xci)

    # ω = (
    #     distance_weight((x[idx_x], y[idx_y]), p; order=order),
    #     distance_weight((x[idx_x + 1], y[idx_y]), p; order=order),
    #     distance_weight((x[idx_x], y[idx_y + 1]), p; order=order),
    #     distance_weight((x[idx_x + 1], y[idx_y + 1]), p; order=order),
    # )

    ω = (
        bilinear_weight((x[idx_x], y[idx_y]), p, dxi),
        bilinear_weight((x[idx_x + 1], y[idx_y]), p, dxi),
        bilinear_weight((x[idx_x], y[idx_y + 1]), p, dxi),
        bilinear_weight((x[idx_x + 1], y[idx_y + 1]), p, dxi),
    )

    nt = Threads.threadid()

    upper[nt][idx_x, idx_y] += ω[1] * Fpi
    upper[nt][idx_x + 1, idx_y] += ω[2] * Fpi
    upper[nt][idx_x, idx_y + 1] += ω[3] * Fpi
    upper[nt][idx_x + 1, idx_y + 1] += ω[4] * Fpi
    lower[nt][idx_x, idx_y] += ω[1]
    lower[nt][idx_x + 1, idx_y] += ω[2]
    lower[nt][idx_x, idx_y + 1] += ω[3]

    return lower[nt][idx_x + 1, idx_y + 1] += ω[4]
end

function gathering!(
    F::Array{T,2}, Fp::Vector{T}, xi, particle_coords, upper, lower; order=2
) where {T}
    fill!.(upper, zero(T))
    fill!.(lower, zero(T))

    # unpack tuples
    px, py = particle_coords
    x, y = xi
    dxi = (x[2] - x[1], y[2] - y[1])

    # number of particles
    np = length(Fp)
    # origin of the domain 
    xci = minimum.(xi)

    # compute ∑ωᵢFᵢ and ∑ωᵢ
    Threads.@threads for i in 1:np
        if !isnan(px[i]) && !isnan(py[i])
            _gathering!(upper, lower, Fp[i], (px[i], py[i]), xci, x, y, dxi, order)
        end
    end

    # compute Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    Threads.@threads for i in eachindex(F)
        @inbounds F[i] =
            sum(upper[nt][i] for nt in 1:Threads.nthreads()) /
            sum(lower[nt][i] for nt in 1:Threads.nthreads())
    end
end

@inbounds function _gathering_xcell!(F, Fp, icell, jcell, xi, p, dxi, order)
    px, py = p # particle coordinates
    xc_cell = (xi[1][icell], xi[2][jcell]) # cell center coordinates
    ω, ωxF = 0.0, 0.0 # init weights
    max_xcell = size(px, 1) # max particles per cell

    for i in 1:max_xcell
        p_i = (px[i, icell, jcell], py[i, icell, jcell])
        any(isnan, p_i) && continue # ignore unused allocations
        ω_i = bilinear_weight(xc_cell, p_i, dxi)
        ω += ω_i
        ωxF += ω_i * Fp[i, icell, jcell]
    end

    return F[icell + 1, jcell + 1] = ωxF / ω
end

function gathering_xcell!(
    F::Array{T,2}, Fp::AbstractArray{T}, xi, particle_coords; order=2
) where {T}
    dxi = (xi[1][2] - xi[1][1], xi[2][2] - xi[2][1])
    nx, ny = size(F)
    Threads.@threads for jcell in 1:(ny - 2)
        for icell in 1:(nx - 2)
            _gathering_xcell!(F, Fp, icell, jcell, xi, particle_coords, dxi, order)
        end
    end
end

## CPU 3D

@inbounds function _gathering!(upper, lower, Fpi, p, xci, x, y, z, dxi, order)
    # check that the particle is inside the grid
    # isinside(p, x, y, z)

    # indices of lowermost-left corner of   
    # the cell containing the particle
    idx_x, idx_y, idx_z = parent_cell(p, dxi, xci)

    ω = (
        distance_weight((x[idx_x], y[idx_y], z[idx_z]), p; order=order),
        distance_weight((x[idx_x + 1], y[idx_y], z[idx_z]), p; order=order),
        distance_weight((x[idx_x], y[idx_y + 1], z[idx_z]), p; order=order),
        distance_weight((x[idx_x + 1], y[idx_y + 1], z[idx_z]), p; order=order),
        distance_weight((x[idx_x], y[idx_y], z[idx_z + 1]), p; order=order),
        distance_weight((x[idx_x + 1], y[idx_y], z[idx_z + 1]), p; order=order),
        distance_weight((x[idx_x], y[idx_y + 1], z[idx_z + 1]), p; order=order),
        distance_weight((x[idx_x + 1], y[idx_y + 1], z[idx_z + 1]), p; order=order),
    )

    nt = Threads.threadid()

    upper[nt][idx_x, idx_y, idx_z] += ω[1] * Fpi
    upper[nt][idx_x + 1, idx_y, idx_z] += ω[2] * Fpi
    upper[nt][idx_x, idx_y + 1, idx_z] += ω[3] * Fpi
    upper[nt][idx_x + 1, idx_y + 1, idx_z] += ω[4] * Fpi
    upper[nt][idx_x, idx_y, idx_z + 1] += ω[5] * Fpi
    upper[nt][idx_x + 1, idx_y, idx_z + 1] += ω[6] * Fpi
    upper[nt][idx_x, idx_y + 1, idx_z + 1] += ω[7] * Fpi
    upper[nt][idx_x + 1, idx_y + 1, idx_z + 1] += ω[8] * Fpi
    lower[nt][idx_x, idx_y, idx_z] += ω[1]
    lower[nt][idx_x + 1, idx_y, idx_z] += ω[2]
    lower[nt][idx_x, idx_y + 1, idx_z] += ω[3]
    lower[nt][idx_x + 1, idx_y + 1, idx_z] += ω[4]
    lower[nt][idx_x, idx_y, idx_z + 1] += ω[5]
    lower[nt][idx_x + 1, idx_y, idx_z + 1] += ω[6]
    lower[nt][idx_x, idx_y + 1, idx_z + 1] += ω[7]

    return lower[nt][idx_x + 1, idx_y + 1, idx_z + 1] += ω[8]
end

function gathering!(
    F::Array{T,3}, Fp::Vector{T}, xi, particle_coords, upper, lower; order=2
) where {T}
    fill!(upper, zero(T))
    fill!(lower, zero(T))

    # unpack tuples
    px, py, pz = particle_coords
    x, y, z = xi
    dxi = (x[2] - x[1], y[2] - y[1], z[2] - z[1])

    # number of particles
    np = length(Fp)

    # origin of the domain 
    xci = minimum.(xi)

    # compute ∑ωᵢFᵢ and ∑ωᵢ
    Threads.@threads for i in 1:np
        if !isnan(px[i]) && !isnan(py[i]) && !isnan(pz[i])
            _gathering!(
                upper, lower, Fp[i], (px[i], py[i], pz[i]), xci, x, y, z, dxi, order
            )
        end
    end

    # compute Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    Threads.@threads for i in eachindex(F)
        @inbounds F[i] =
            sum(upper[nt][i] for nt in 1:Threads.nthreads()) /
            sum(lower[nt][i] for nt in 1:Threads.nthreads())
    end
end

# CUDA 

function gathering!(
    Fd::CuArray{T,N}, Fpd::CuArray{T,1}, xi, particle_coords; nt=512
) where {T,N}
    upper = CUDA.zeros(T, size(Fd))
    lower = CUDA.zeros(T, size(Fd))

    return gathering!(Fd, Fpd, xi, particle_coords, upper, lower; nt=nt)
end

## CUDA 2D

function _gather1!(
    upper::CuDeviceMatrix{T,1},
    lower::CuDeviceMatrix{T,1},
    Fpd::CuDeviceVector{T,1},
    xci,
    xi,
    dxi,
    p;
    order=2,
) where {T}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # unpack tuples
    px, py = p
    x, y = xi

    @inbounds if idx ≤ length(px)
        # check that the particle is inside the grid
        # isinside(px[idx], py[idx], x, y)

        p_idx = (px[idx], py[idx])

        if !any(isnan, p_idx)
            # indices of lowermost-left corner of
            # the cell containing the particle
            idx_x, idx_y = parent_cell(p_idx, dxi, xci)

            ω1::Float64 = distance_weight((x[idx_x], y[idx_y]), p_idx; order=order)
            ω2::Float64 = distance_weight((x[idx_x + 1], y[idx_y]), p_idx; order=order)
            ω3::Float64 = distance_weight((x[idx_x], y[idx_y + 1]), p_idx; order=order)
            ω4::Float64 = distance_weight((x[idx_x + 1], y[idx_y + 1]), p_idx; order=order)

            Fpi::Float64 = Fpd[idx]

            CUDA.@atomic upper[idx_x, idx_y] += ω1 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y] += ω2 * Fpi
            CUDA.@atomic upper[idx_x, idx_y + 1] += ω3 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y + 1] += ω4 * Fpi
            CUDA.@atomic lower[idx_x, idx_y] += ω1
            CUDA.@atomic lower[idx_x + 1, idx_y] += ω2
            CUDA.@atomic lower[idx_x, idx_y + 1] += ω3
            CUDA.@atomic lower[idx_x + 1, idx_y + 1] += ω4
        end
    end

    return nothing
end

function _gather2!(Fd::CuDeviceArray{T,2}, upper, lower) where {T}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (idx ≤ size(Fd, 1)) && (idy ≤ size(Fd, 2))
        @inbounds Fd[idx, idy] = upper[idx, idy] / lower[idx, idy]
    end

    return nothing
end

function gathering!(
    Fd::CuArray{T,2}, Fpd::CuArray{T,1}, xi, particle_coords, upper, lower; nt=512
) where {T}
    fill!(upper, zero(T))
    fill!(lower, zero(T))

    x, y = xi
    dxi = (x[2] - x[1], y[2] - y[1])
    # origin of the domain 
    xci = minimum.(xi)

    # first kernel that computes ∑ωᵢFᵢ and ∑ωᵢ
    N = length(Fpd)
    numblocks = ceil(Int, N / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gather1!(
            upper, lower, Fpd, xci, xi, dxi, particle_coords
        )
    end

    # seond and final kernel that computes Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    nx, ny = size(Fd)
    nblocksx = ceil(Int, nx / 32)
    nblocksy = ceil(Int, ny / 32)
    CUDA.@sync begin
        @cuda threads = (32, 32) blocks = (nblocksx, nblocksy) _gather2!(Fd, upper, lower)
    end
end

function gathering_xcell!(
    F::CuArray{T,2}, Fp::CuArray{T,N}, xi, particle_coords
) where {T,N}
    px, py = particle_coords
    x, y = xi
    dxi = (x[2] - x[1], y[2] - y[1])

    # first kernel that computes ∑ωᵢFᵢ and ∑ωᵢ
    nxcell, ny, nz = size(px)
    nblocksx = ceil(Int, ny / 32)
    nblocksy = ceil(Int, nz / 32)
    threadsx = 32
    threadsy = 32

    shmem_size = (3 * sizeof(T) * nxcell * threadsx * threadsy)

    CUDA.@sync begin
        @cuda threads = (threadsx, threadsy) blocks = (nblocksx, nblocksy) _gather_xcell!(
            F, Fp, xi, px, py, dxi
        )
    end
end

function _gather_xcell!(F::CuDeviceArray{T,2}, Fp, xi, px, py, dxi) where {T}
    icell = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jcell = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (icell ≤ size(px, 2)) && (jcell ≤ size(px, 3))

        # unpack tuples
        xc_cell = (xi[1][icell], xi[2][jcell]) # cell center coordinates
        ω, ωxF = 0.0, 0.0 # init weights
        max_xcell = size(px, 1) # max particles per cell

        for i in 1:max_xcell
            p_i = (px[i, icell, jcell], py[i, icell, jcell])
            any(isnan, p_i) && continue # ignore unused allocations
            ω_i = bilinear_weight(xc_cell, p_i, dxi)
            ω += ω_i
            ωxF += ω_i * Fp[i, icell, jcell]
        end

        F[icell + 1, jcell + 1] = ωxF / ω
    end

    return nothing
end

## CUDA 3D 

function _gather1!(
    upper::CuDeviceArray{T,3},
    lower::CuDeviceArray{T,3},
    Fpd::CuDeviceVector{T,1},
    xci,
    xi,
    dxi,
    p;
    order=2,
) where {T}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # unpack tuples
    px, py, pz = p
    x, y, z = xi

    @inbounds if idx ≤ length(px)
        # check that the particle is inside the grid
        # isinside(px[idx], py[idx], pz[idx], x, y, z)

        p_idx = (px[idx], py[idx], pz[idx])

        if !any(isnan, p_idx)
            # indices of lowermost-left corner of
            # the cell containing the particle
            idx_x, idx_y, idx_z = parent_cell(p_idx, dxi, xci)

            ω1::Float64 = distance_weight(
                (x[idx_x], y[idx_y], z[idx_z]), p_idx; order=order
            )
            ω2::Float64 = distance_weight(
                (x[idx_x + 1], y[idx_y], z[idx_z]), p_idx; order=order
            )
            ω3::Float64 = distance_weight(
                (x[idx_x], y[idx_y + 1], z[idx_z]), p_idx; order=order
            )
            ω4::Float64 = distance_weight(
                (x[idx_x + 1], y[idx_y + 1], z[idx_z]), p_idx; order=order
            )
            ω5::Float64 = distance_weight(
                (x[idx_x], y[idx_y], z[idx_z + 1]), p_idx; order=order
            )
            ω6::Float64 = distance_weight(
                (x[idx_x + 1], y[idx_y], z[idx_z + 1]), p_idx; order=order
            )
            ω7::Float64 = distance_weight(
                (x[idx_x], y[idx_y + 1], z[idx_z + 1]), p_idx; order=order
            )
            ω8::Float64 = distance_weight(
                (x[idx_x + 1], y[idx_y + 1], z[idx_z + 1]), p_idx; order=order
            )

            Fpi::Float64 = Fpd[idx]

            CUDA.@atomic upper[idx_x, idx_y, idx_z] += ω1 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y, idx_z] += ω2 * Fpi
            CUDA.@atomic upper[idx_x, idx_y + 1, idx_z] += ω3 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y + 1, idx_z] += ω4 * Fpi
            CUDA.@atomic upper[idx_x, idx_y, idx_z + 1] += ω5 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y, idx_z + 1] += ω6 * Fpi
            CUDA.@atomic upper[idx_x, idx_y + 1, idx_z + 1] += ω7 * Fpi
            CUDA.@atomic upper[idx_x + 1, idx_y + 1, idx_z + 1] += ω8 * Fpi
            CUDA.@atomic lower[idx_x, idx_y, idx_z] += ω1
            CUDA.@atomic lower[idx_x + 1, idx_y, idx_z] += ω2
            CUDA.@atomic lower[idx_x, idx_y + 1, idx_z] += ω3
            CUDA.@atomic lower[idx_x + 1, idx_y + 1, idx_z] += ω4
            CUDA.@atomic lower[idx_x, idx_y, idx_z + 1] += ω5
            CUDA.@atomic lower[idx_x + 1, idx_y, idx_z + 1] += ω6
            CUDA.@atomic lower[idx_x, idx_y + 1, idx_z + 1] += ω7
            CUDA.@atomic lower[idx_x + 1, idx_y + 1, idx_z + 1] += ω8
        end
    end

    return nothing
end

function _gather2!(
    Fd::CuDeviceArray{T,3}, upper::CuDeviceArray{T,3}, lower::CuDeviceArray{T,3}
) where {T}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    idz = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (idx ≤ size(Fd, 1)) && (idy ≤ size(Fd, 2)) && (idz ≤ size(Fd, 3))
        @inbounds Fd[idx, idy, idz] = upper[idx, idy, idz] / lower[idx, idy, idz]
    end

    return nothing
end

function gathering!(
    Fd::CuArray{T,3}, Fpd::CuArray{T,1}, xi, particle_coords, upper, lower; nt=512
) where {T}
    fill!(upper, zero(T))
    fill!(lower, zero(T))

    x, y, z = xi
    dxi = (x[2] - x[1], y[2] - y[1], z[2] - z[1])
    # origin of the domain 
    xci = minimum.(xi)

    # first kernel that computes ∑ωᵢFᵢ and ∑ωᵢ
    N = length(Fpd)
    numblocks = ceil(Int, N / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _gather1!(
            upper, lower, Fpd, xci, xi, dxi, particle_coords
        )
    end

    # second and final kernel that computes Fᵢ=∑ωᵢFpᵢ/∑ωᵢ
    nx, ny, nz = size(Fd)
    nblocksx = ceil(Int, nx / 8)
    nblocksy = ceil(Int, ny / 8)
    nblocksz = ceil(Int, nz / 8)
    CUDA.@sync begin
        @cuda threads = (8, 8, 8) blocks = (nblocksx, nblocksy, nblocksz) _gather2!(
            Fd, upper, lower
        )
    end
end

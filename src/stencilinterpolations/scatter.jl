## CPU 2D
function grid2particle_xvertex!(Fp::Array, xvi, F::Array{T,2}, particle_coords) where {T}
    dxi = grid_size(xvi)
    nx, ny = length.(xvi)
    max_xcell = size(particle_coords[1], 1)
    Threads.@threads for jnode in 1:(ny - 1)
        for inode in 1:(nx - 1)
            _grid2particle_xvertex!(
                Fp, particle_coords, xvi, dxi, F, max_xcell, (inode, jnode)
            )
        end
    end
    return nothing
end

function _grid2particle_xvertex(
    p_i::NTuple, xvi::NTuple, dxi::NTuple, F::AbstractArray, idx
)
    # F at the cell corners
    Fi = field_corners(F, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(p_i, xvi, dxi, idx)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)
    return Fp
end

## CPU 3D
function grid2particle_xvertex!(Fp::Array, xvi, F::Array{T,3}, particle_coords) where {T}
    # cell dimensions
    dxi = grid_size(xvi)
    nx, ny, nz = length.(xvi)
    max_xcell = size(particle_coords[1], 1)
    Threads.@threads for knode in 1:(nz - 1)
        for jnode in 1:(ny - 1), inode in 1:(nx - 1)
            _grid2particle_xvertex!(
                Fp, particle_coords, xvi, dxi, F, max_xcell, (inode, jnode, knode)
            )
        end
    end
end

function _grid2particle_xvertex!(
    Fp::Array, p::NTuple, xvi::NTuple, dxi::NTuple, F::Array, max_xcell, idx
)
    @inline function particle2tuple(ip::Integer, idx::NTuple{N,T}) where {N,T}
        return ntuple(i -> p[i][ip, idx...], Val(N))
    end

    for i in 1:max_xcell
        # check that the particle is inside the grid
        # isinside(p, xi)

        p_i = particle2tuple(i, idx)

        any(isnan, p_i) && continue

        # F at the cell corners
        Fi = field_corners(F, idx)

        # normalize particle coordinates
        ti = normalize_coordinates(p_i, xvi, dxi, idx)

        # Interpolate field F onto particle
        Fp[i, idx...] = ndlinear(ti, Fi)
    end
end

## CUDA 2D
function grid2particle_xvertex!(
    Fp::CuArray, xvi, F::CuArray{T,2}, particle_coords; nt=512
) where {T}
    # cell dimensions
    dxi = grid_size(xvi)
    max_xcell, ny, nz = size(particle_coords[1])
    nblocksx = ceil(Int, ny / 32)
    nblocksy = ceil(Int, nz / 32)
    threadsx = 32
    threadsy = 32

    CUDA.@sync begin
        @cuda threads = (threadsx, threadsy) blocks = (nblocksx, nblocksy) _grid2particle_xvertex!(
            Fp, particle_coords, xvi, dxi, F, max_xcell
        )
    end
end

function _grid2particle_xvertex!(
    Fp::CuDeviceArray, p::NTuple, xvi::NTuple, dxi::NTuple{2,T}, F::CuDeviceArray, max_xcell
) where {T}
    @inline function particle2tuple(ip::Integer, idx::NTuple{N,T}) where {N,T}
        return ntuple(i -> p[i][ip, idx...], Val(N))
    end

    inode = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jnode = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (inode ≤ size(p[1], 2)) && (jnode ≤ size(p[1], 3))
        idx = (inode, jnode)

        for i in 1:max_xcell
            # check that the particle is inside the grid
            # isinside(p, xi)

            p_i = particle2tuple(i, idx)

            any(isnan, p_i) && continue

            # F at the cell corners
            Fi = field_corners(F, idx)

            # normalize particle coordinates
            ti = normalize_coordinates(p_i, xvi, dxi, idx)

            # Interpolate field F onto particle
            Fp[i, inode, jnode] = ndlinear(ti, Fi)
        end
    end

    return nothing
end

## CPU
function _grid2particle(p::NTuple, xci::Tuple, xi::NTuple, dxi::NTuple, F::AbstractArray)

    # indices of lowermost-left corner of the cell 
    # containing the particle
    idx = parent_cell(p, dxi, xci)

    # normalize particle coordinates
    ti = normalize_coordinates(p, xi, dxi, idx)

    # F at the cell corners
    Fi = field_corners(F, idx)

    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)

    return Fp
end

function _grid2particle_xcell_centered(
    p_i::NTuple, xi::NTuple, xci_augmented, dxi::NTuple, F::AbstractArray, icell, jcell
)

    # cell indices
    idx = (icell, jcell)
    # F at the cell corners
    Fi, xci = field_centers(F, p_i, xi, xci_augmented, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(p_i, xci, dxi)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)

    return Fp
end

function grid2particle(xi, F::Array{T,N}, particle_coords) where {T,N}
    Fp = zeros(T, size(particle_coords[1])...)

    # cell dimensions
    dxi = grid_size(xi)

    # origin of the domain 
    xci = minimum.(xi)

    Threads.@threads for i in eachindex(particle_coords[1])
        @inbounds Fp[i] = _grid2particle(
            ntuple(j -> particle_coords[j][i], Val(N)), xci, xi, dxi, F
        )
    end

    return Fp
end

function grid2particle!(Fp, xi, F::Array{T,N}, particle_coords) where {T,N}
    # cell dimensions
    dxi = grid_size(xi)
    # origin of the domain 
    xci = minimum.(xi)
    Threads.@threads for i in eachindex(particle_coords[1])
        if !any(isnan, ntuple(j -> particle_coords[j][i], Val(N)))
            @inbounds Fp[i] = _grid2particle(
                ntuple(j -> particle_coords[j][i], Val(N)), xci, xi, dxi, F
            )
        end
    end
end

function grid2particle_xcell!(Fp, xi, F::Array{T,N}, particle_coords) where {T,N}
    # cell dimensions
    dxi = grid_size(xi)
    xci_augmented = ntuple(Val(N)) do i
        (xi[i][1] - dxi[i]):dxi[i]:(xi[i][end] + dxi[i])
    end
    nx, ny = length.(xi)
    max_xcell = size(particle_coords[1], 1)
    Threads.@threads for jcell in 1:ny
        for icell in 1:nx
            _grid2particle_xcell!(
                Fp, particle_coords, xi, xci_augmented, dxi, F, max_xcell, icell, jcell
            )
        end
    end
end

function _grid2particle_xcell!(
    Fp,
    p::NTuple,
    xi::NTuple,
    xci_augmented,
    dxi::NTuple,
    F::AbstractArray,
    max_xcell,
    icell,
    jcell,
)
    idx = (icell, jcell)

    @inline function particle2tuple(ip::Integer, idx::NTuple{N,T}) where {N,T}
        return ntuple(i -> p[i][ip, idx...], Val(N))
    end

    for i in 1:max_xcell
        # check that the particle is inside the grid
        # isinside(p, xi)

        p_i = particle2tuple(i, idx)

        any(isnan, p_i) && continue

        # F at the cell corners
        Fi, xci = field_centers(F, p_i, xi, xci_augmented, idx)

        # normalize particle coordinates
        ti = normalize_coordinates(p_i, xci, dxi)

        # Interpolate field F onto particle
        Fp[i, icell, jcell] = ndlinear(ti, Fi)
    end
end

## CUDA

function grid2particle(
    xi, Fd::CuArray{T,N}, particle_coords::NTuple{N,CuArray}; nt=512
) where {T,N}
    dxi = grid_size(xi)
    n = length(particle_coords[1])
    Fpd = CuArray{T,1}(undef, n)
    # origin of the domain 
    xci = minimum.(xi)
    numblocks = ceil(Int, n / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _grid2particle!(
            Fpd, particle_coords, dxi, xci, xi, Fd, n
        )
    end

    return Fpd
end

function grid2particle!(
    Fpd::CuArray{T,1}, xi, Fd::CuArray{T,N}, particle_coords::NTuple{N,CuArray}; nt=512
) where {T,N}
    dxi = grid_size(xi)
    n = length(particle_coords[1])
    # origin of the domain 
    xci = minimum.(xi)
    numblocks = ceil(Int, n / nt)
    CUDA.@sync begin
        @cuda threads = nt blocks = numblocks _grid2particle!(
            Fpd, particle_coords, dxi, xci, xi, Fd, n
        )
    end
end

function _grid2particle!(
    Fp::CuDeviceVector{T,1},
    p::NTuple{N,CuDeviceVector{T,1}},
    dxi::NTuple{N,T},
    xci::NTuple{N,B},
    xi::NTuple{N,A},
    F::CuDeviceArray{T,N},
    n::Integer,
) where {T,A,B,N}
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    @inbounds if ix ≤ n
        pix = particle2tuple(p, ix)

        if !any(isnan, pix)
            # check that the particle is inside the grid
            # isinside(pix, xi)

            # indices of lowermost-left corner of the cell 
            # containing the particle
            idx = parent_cell(pix, dxi, xci)

            # normalize particle coordinates
            ti = normalize_coordinates(pix, xi, dxi, idx)

            # F at the cell corners
            Fi = field_corners(F, idx)

            # Interpolate field F onto particle
            Fp[ix] = ndlinear(ti, Fi)
        end
    end

    return nothing
end

function grid2particle_xcell!(
    Fp::CuArray, xi, F::CuArray{T,N}, particle_coords::NTuple{N,CuArray}
) where {T,N}
    dxi = grid_size(xi)
    xci_augmented = ntuple(Val(N)) do i
        (xi[i][1] - dxi[i]):dxi[i]:(xi[i][end] + dxi[i])
    end

    max_xcell, ny, nz = size(particle_coords[1])
    nblocksx = ceil(Int, ny / 32)
    nblocksy = ceil(Int, nz / 32)
    threadsx = 32
    threadsy = 32

    CUDA.@sync begin
        @cuda threads = (threadsx, threadsy) blocks = (nblocksx, nblocksy) _grid2particle_xcell!(
            Fp, particle_coords, dxi, xi, xci_augmented, F, max_xcell
        )
    end
end

function _grid2particle_xcell!(
    Fp::CuDeviceArray,
    p::NTuple{N,CuDeviceArray},
    dxi::NTuple{N,T},
    xi::NTuple{N,A},
    xci_augmented::NTuple{N,B},
    F::CuDeviceArray{T,N},
    max_xcell::Integer,
) where {T,A,B,N}
    @inline function particle2tuple(ip::Integer, idx::Vararg{T,N}) where {N,T}
        return ntuple(i -> p[i][ip, idx...], Val(N))
    end

    icell = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    jcell = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (icell ≤ size(p[1], 2)) && (jcell ≤ size(p[1], 3))
        idx = (icell, jcell)

        for i in 1:max_xcell
            p_i = particle2tuple(i, icell, jcell)

            any(isnan, p_i) && continue

            # F at the cell corners
            Fi, xci = field_centers(F, p_i, xi, xci_augmented, idx)

            # normalize particle coordinates
            ti = normalize_coordinates(p_i, xci, dxi)

            # Interpolate field F onto particle
            Fp[i, icell, jcell] = ndlinear(ti, Fi)
        end
    end

    return nothing
end

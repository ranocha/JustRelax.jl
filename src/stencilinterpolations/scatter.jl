# LAUNCHERS

## CPU 2D

function grid2particle_xvertex!(Fp::AbstractArray, xvi, F::Array{T,2}, particle_coords) where {T}
    dxi = grid_size(xvi)
    nx, ny = length.(xvi)
    # max_xcell = cellnum(particle_coords[1])
    Threads.@threads for jnode in 1:(ny - 1)
        for inode in 1:(nx - 1)
            _grid2particle_xvertex!(
                Fp, particle_coords, xvi, dxi, F, (inode, jnode)
            )
        end
    end
    return nothing
end

## CPU 3D

function grid2particle_xvertex!(Fp::AbstractArray, xvi, F::Array{T,3}, particle_coords) where {T}
    # cell dimensions
    dxi = grid_size(xvi)
    nx, ny, nz = length.(xvi)
    # max_xcell = size(particle_coords[1], 1)
    Threads.@threads for knode in 1:(nz - 1)
        for jnode in 1:(ny - 1), inode in 1:(nx - 1)
            _grid2particle_xvertex!(
                Fp, particle_coords, xvi, dxi, F, (inode, jnode, knode)
            )
        end
    end
end

## CUDA 2D

function grid2particle_xvertex!(
    Fp, xvi, F::CuArray{T,2}, particle_coords
) where {T}
    # cell dimensions
    dxi = grid_size(xvi)
    # max_xcell = cellnum(particle_coords[1]) 
    nx, ny   = size(particle_coords[1])
    nblocksx = ceil(Int, nx / 32)
    nblocksy = ceil(Int, ny / 32)
    threadsx = 32
    threadsy = 32

    CUDA.@sync begin
        @cuda threads = (threadsx, threadsy) blocks = (nblocksx, nblocksy) _grid2particle_xvertex!(
            Fp, particle_coords, xvi, dxi, F
        )
    end
end

function grid2particle_xvertex!(
    Fp, xvi, F::CuArray{T,3}, particle_coords
) where {T}

    # cell dimensions
    dxi = grid_size(xvi)
    # max_xcell = cellnum(particle_coords[1]) 
    nx, ny, nz = size(particle_coords[1])
    nblocksx   = ceil(Int, nx / 16)
    nblocksy   = ceil(Int, ny / 16)
    nblocksz   = ceil(Int, nz / 4)
    threadsx   = 16
    threadsy   = 16
    threadsz   = 4
    nthreads   = threadsx, threadsy, threadsz
    nblocks    = nblocksx, nblocksy, nblocksz
        
    CUDA.@sync begin
        @cuda threads = nthreads blocks = nblocks _grid2particle_xvertex!(Fp, particle_coords, xvi, dxi, F)
    end
end

# INNER INTERPOLATION KERNEL

@inline function inner_grid2particle_xvertex!(Fp, p, xvi, dxi, F, idx)
    # iterate over all the particles within the cells of index `idx` 
    for ip in cellaxes(Fp)
        # cache particle coordinates 
        p_i = ntuple(i -> (@cell p[i][ip, idx...]), Val(N))

        any(isnan, p_i) && continue # empty memory

        # F at the cell corners
        Fi = field_corners(F, idx)

        # normalize particle coordinates
        ti = normalize_coordinates(p_i, xvi, dxi, idx)

        # Interpolate field F onto particle
        @cell Fp[ip, idx...] = ndlinear(ti, Fi)
    end
end

# CPU DIMENSION AGNOSTIC KERNEL

_grid2particle_xvertex!(Fp, p::Tuple, xvi::Tuple, dxi::Tuple, F::Array, idx) = inner_grid2particle_xvertex!(Fp, p, xvi, dxi, F, idx)
    
    # # @inline function particle2tuple(ip::Integer, idx::NTuple{N,T}) where {N,T}
    # #     return ntuple(i -> @cell(p[i][ip, idx...]), Val(N))
    # # end

    # for i in cellaxes(Fp)
    #     # cache particle coordinates 
    #     p_i = particle2tuple(p, i, idx)

    #     any(isnan, p_i) && continue

    #     # F at the cell corners
    #     Fi = field_corners(F, idx)

    #     # normalize particle coordinates
    #     ti = normalize_coordinates(p_i, xvi, dxi, idx)

    #     # Interpolate field F onto particle
    #     @cell Fp[i, idx...] = ndlinear(ti, Fi)
    # end

# CUDA DIMENSION AGNOSTIC KERNEL

function _grid2particle_xvertex!(
    Fp, p::T, xvi::T, dxi::T, F::CuDeviceArray,
) where T<:NTuple{N, Any}

    idx = cuda_indices(T)
    if all(idx .≤ size(Fp))
        inner_grid2particle_xvertex!(Fp, p, xvi, dxi, F, idx)
    end

    return nothing
end

@inline function cuda_indices(::Type{NTuple{2, Any}})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    return i, j
end

@inline function cuda_indices(::Type{NTuple{3, Any}})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    return i, j, k
end


# CUDA 3D KERNEL

# function _grid2particle_xvertex!(
#     Fp, p::NTuple, xvi::NTuple, dxi::NTuple{3,T}, F::CuDeviceArray,
# ) where {T}

#     inode = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     jnode = (blockIdx().y - 1) * blockDim().y + threadIdx().y
#     knode = (blockIdx().z - 1) * blockDim().z + threadIdx().z

#     if (inode ≤ size(Fp, 1)) && (jnode ≤ size(Fp, 2)) && (knode ≤ size(Fp, 3))
#         idx = (inode, jnode, knode)
#         inner_grid2particle_xvertex!(Fp, p, xvi, dxi, F, idx)
#     end

#     return nothing
# end

# ## CPU
# function _grid2particle(p::NTuple, xci::Tuple, xi::NTuple, dxi::NTuple, F::AbstractArray)

#     # indices of lowermost-left corner of the cell 
#     # containing the particle
#     idx = parent_cell(p, dxi, xci)

#     # normalize particle coordinates
#     ti = normalize_coordinates(p, xi, dxi, idx)

#     # F at the cell corners
#     Fi = field_corners(F, idx)

#     # Interpolate field F onto particle
#     Fp = ndlinear(ti, Fi)

#     return Fp
# end

# function _grid2particle_xcell_centered(
#     p_i::NTuple, xi::NTuple, xci_augmented, dxi::NTuple, F::AbstractArray, icell, jcell
# )

#     # cell indices
#     idx = (icell, jcell)
#     # F at the cell corners
#     Fi, xci = field_centers(F, p_i, xi, xci_augmented, idx)
#     # normalize particle coordinates
#     ti = normalize_coordinates(p_i, xci, dxi)
#     # Interpolate field F onto particle
#     Fp = ndlinear(ti, Fi)

#     return Fp
# end

# function grid2particle(xi, F::Array{T,N}, particle_coords) where {T,N}
#     Fp = zeros(T, size(particle_coords[1])...)

#     # cell dimensions
#     dxi = grid_size(xi)

#     # origin of the domain 
#     xci = minimum.(xi)

#     Threads.@threads for i in eachindex(particle_coords[1])
#         @inbounds Fp[i] = _grid2particle(
#             ntuple(j -> particle_coords[j][i], Val(N)), xci, xi, dxi, F
#         )
#     end

#     return Fp
# end

# function grid2particle!(Fp, xi, F::Array{T,N}, particle_coords) where {T,N}
#     # cell dimensions
#     dxi = grid_size(xi)
#     # origin of the domain 
#     xci = minimum.(xi)
#     Threads.@threads for i in eachindex(particle_coords[1])
#         if !any(isnan, ntuple(j -> particle_coords[j][i], Val(N)))
#             @inbounds Fp[i] = _grid2particle(
#                 ntuple(j -> particle_coords[j][i], Val(N)), xci, xi, dxi, F
#             )
#         end
#     end
# end

# function grid2particle_xcell!(Fp, xi, F::Array{T,N}, particle_coords) where {T,N}
#     # cell dimensions
#     dxi = grid_size(xi)
#     xci_augmented = ntuple(Val(N)) do i
#         (xi[i][1] - dxi[i]):dxi[i]:(xi[i][end] + dxi[i])
#     end
#     nx, ny = length.(xi)
#     max_xcell = size(particle_coords[1], 1)
#     Threads.@threads for jcell in 1:ny
#         for icell in 1:nx
#             _grid2particle_xcell!(
#                 Fp, particle_coords, xi, xci_augmented, dxi, F, max_xcell, icell, jcell
#             )
#         end
#     end
# end

# function _grid2particle_xcell!(
#     Fp,
#     p::NTuple,
#     xi::NTuple,
#     xci_augmented,
#     dxi::NTuple,
#     F::AbstractArray,
#     max_xcell,
#     icell,
#     jcell,
# )
#     idx = (icell, jcell)

#     @inline function particle2tuple(ip::Integer, idx::NTuple{N,T}) where {N,T}
#         return ntuple(i -> p[i][ip, idx...], Val(N))
#     end

#     for i in 1:max_xcell
#         # check that the particle is inside the grid
#         # isinside(p, xi)

#         p_i = particle2tuple(i, idx)

#         any(isnan, p_i) && continue

#         # F at the cell corners
#         Fi, xci = field_centers(F, p_i, xi, xci_augmented, idx)

#         # normalize particle coordinates
#         ti = normalize_coordinates(p_i, xci, dxi)

#         # Interpolate field F onto particle
#         Fp[i, icell, jcell] = ndlinear(ti, Fi)
#     end
# end

# ## CUDA

# function grid2particle(
#     xi, Fd::CuArray{T,N}, particle_coords::NTuple{N, CuArray}; nt=512
# ) where {T,N}
#     dxi = grid_size(xi)
#     n = length(particle_coords[1])
#     Fpd = CuArray{T,1}(undef, n)
#     # origin of the domain 
#     xci = minimum.(xi)
#     numblocks = ceil(Int, n / nt)
#     CUDA.@sync begin
#         @cuda threads = nt blocks = numblocks _grid2particle!(
#             Fpd, particle_coords, dxi, xci, xi, Fd, n
#         )
#     end

#     return Fpd
# end

# function grid2particle!(
#     Fpd::CuArray{T,1}, xi, Fd::CuArray{T,N}, particle_coords::NTuple{N,CuArray}; nt=512
# ) where {T,N}
#     dxi = grid_size(xi)
#     n = length(particle_coords[1])
#     # origin of the domain 
#     xci = minimum.(xi)
#     numblocks = ceil(Int, n / nt)
#     CUDA.@sync begin
#         @cuda threads = nt blocks = numblocks _grid2particle!(
#             Fpd, particle_coords, dxi, xci, xi, Fd, n
#         )
#     end
# end

# function _grid2particle!(
#     Fp::CuDeviceVector{T,1},
#     p::NTuple{N,CuDeviceVector{T,1}},
#     dxi::NTuple{N,T},
#     xci::NTuple{N,B},
#     xi::NTuple{N,A},
#     F::CuDeviceArray{T,N},
#     n::Integer,
# ) where {T,A,B,N}
#     ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x

#     @inbounds if ix ≤ n
#         pix = particle2tuple(p, ix)

#         if !any(isnan, pix)
#             # check that the particle is inside the grid
#             # isinside(pix, xi)

#             # indices of lowermost-left corner of the cell 
#             # containing the particle
#             idx = parent_cell(pix, dxi, xci)

#             # normalize particle coordinates
#             ti = normalize_coordinates(pix, xi, dxi, idx)

#             # F at the cell corners
#             Fi = field_corners(F, idx)

#             # Interpolate field F onto particle
#             Fp[ix] = ndlinear(ti, Fi)
#         end
#     end

#     return nothing
# end

# function grid2particle_xcell!(
#     Fp::CuArray, xi, F::CuArray{T, 2}, particle_coords::NTuple{2, Any}
# ) where {T}
#     dxi = grid_size(xi)
#     xci_augmented = ntuple(Val(N)) do i
#         (xi[i][1] - dxi[i]):dxi[i]:(xi[i][end] + dxi[i])
#     end

#     max_xcell, ny, nz = size(particle_coords[1])
#     nblocksx = ceil(Int, ny / 32)
#     nblocksy = ceil(Int, nz / 32)
#     threadsx = 32
#     threadsy = 32

#     CUDA.@sync begin
#         @cuda threads = (threadsx, threadsy) blocks = (nblocksx, nblocksy) _grid2particle_xcell!(
#             Fp, particle_coords, dxi, xi, xci_augmented, F, max_xcell
#         )
#     end
# end

# function _grid2particle_xcell!(
#     Fp::CuDeviceArray,
#     p::NTuple{N,CuDeviceArray},
#     dxi::NTuple{N,T},
#     xi::NTuple{N,A},
#     xci_augmented::NTuple{N,B},
#     F::CuDeviceArray{T,N},
#     max_xcell::Integer,
# ) where {T,A,B,N}

#     icell = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     jcell = (blockIdx().y - 1) * blockDim().y + threadIdx().y

#     if (icell ≤ size(p[1], 2)) && (jcell ≤ size(p[1], 3))
#         idx = (icell, jcell)

#         for i in 1:max_xcell
#             p_i = particle2tuple(p, i, icell, jcell)

#             any(isnan, p_i) && continue

#             # F at the cell corners
#             Fi, xci = field_centers(F, p_i, xi, xci_augmented, idx)

#             # normalize particle coordinates
#             ti = normalize_coordinates(p_i, xci, dxi)

#             # Interpolate field F onto particle
#             Fp[i, icell, jcell] = ndlinear(ti, Fi)
#         end
#     end

#     return nothing
# end
function shuffle_particles_vertex!(particles::Particles, grid::NTuple{2,T}, args) where {T}
    # unpack
    (; coords, index) = particles
    nxi = length.(grid)
    nx, ny = nxi
    # px, py = particle_coords
    dxi = compute_dx(grid)

    # offsets = ((1, 0, 0), (2, 0, 0), (1, 0, 1), (1, 1, 0))
    n_i = ceil(Int, nx * 0.5)
    n_j = ceil(Int, ny * 0.5)

    # for offset_i in offsets
    #     offset, offset_x, offset_y = offset_i
    for offset_x in 1:2, offset_y in 1:2
        @parallel (1:n_i, 1:n_j) shuffle_particles_vertex_ps!(
            coords, grid, dxi, nxi, index, offset_x, offset_y, args
        )
    end

    return nothing
end

function shuffle_particles_vertex!(particles::Particles, grid::NTuple{3,T}, args) where {T}
    # unpack
    (; coords, index) = particles
    nxi = length.(grid)
    nx, ny = nxi
    dxi = compute_dx(grid)

    n_i = ceil(Int, nx * 0.5)
    n_j = ceil(Int, ny * 0.5)
    n_k = ceil(Int, nz * 0.5)

    for offset_x in 1:2, offset_y in 1:2, offset_z in 1:2
        @parallel (1:n_i, 1:n_j, 1:n_k) shuffle_particles_vertex_ps!(
                coords, grid, dxi, nxi, index, offset_x, offset_y, offset_z, args,
            )
    end

    return nothing
end

@parallel_indices (icell, jcell) function shuffle_particles_vertex_ps!(
    particle_coords, grid, dxi::NTuple{2,T}, nxi, index, offset_x, offset_y, args
) where {T}
    nx, ny = nxi
    i = 2 * (icell - 1) + offset_x
    j = 2 * (jcell - 1) + offset_y

    if (i ≤ nx - 1) && (j ≤ ny - 1)
        _shuffle_particles_vertex!(particle_coords, grid, dxi, nxi, index, (i, j), args)
    end
    return nothing
end

@parallel_indices (icell, jcell, kcell) function shuffle_particles_vertex_ps!(
    particle_coords,
    grid,
    dxi::NTuple{3,T},
    nxi,
    index,
    offset_x,
    offset_y,
    offset_z,
    args,
) where {T}
    nx, ny = nxi
    i = 2 * (icell - 1) + offset_x
    j = 2 * (jcell - 1) + offset_y
    k = 2 * (kcell - 1) + offset_z

    if (i ≤ nx - 1) && (j ≤ ny - 1) && (k ≤ nz - 1)
        _shuffle_particles_vertex!(particle_coords, grid, dxi, nxi, index, (i, j, k), args)
    end
    return nothing
end

# function _shuffle_particles_vertex!(
#     particle_coords, grid, dxi, nxi, index, parent_cell::NTuple{2,Int64}, args
# )

#     # coordinate of the lower-most-left coordinate of the parent cell 
#     corner_xi = corner_coordinate(grid, parent_cell)
#     # iterate over neighbouring (child) cells
#     for j in -1:1, i in -1:1
#         idx_loop = (i, j)
#         __shuffle_particles_vertex!(
#             particle_coords, corner_xi, dxi, nxi, index, parent_cell, args, idx_loop
#         )
#     end

#     return nothing
# end

# function _shuffle_particles_vertex!(
#     particle_coords, grid, dxi, nxi, index, parent_cell::NTuple{3,Int64}, args
# )
#     # coordinate of the lower-most-left coordinate of the parent cell 
#     corner_xi = corner_coordinate(grid, parent_cell)
#     # iterate over neighbouring (child) cells
#     for k in -1:1, j in -1:1, i in -1:1
#         idx_loop = (i, j, k)
#         if idx_loop != (0,0,0)
#             __shuffle_particles_vertex!(
#                 particle_coords, corner_xi, dxi, nxi, index, parent_cell, args, idx_loop
#             )
#         end
#     end

#     return nothing
# end

@generated function _shuffle_particles_vertex!(
    particle_coords, grid, dxi, nxi, index, parent_cell::NTuple{N,Int64}, args
) where N
    quote
        # coordinate of the lower-most-left coordinate of the parent cell 
        # iterate over neighbouring (child) cells
        corner_xi = corner_coordinate(grid, parent_cell)
        if $N==2
            for j in -1:1, i in -1:1
                idx_loop = (i, j)
                __shuffle_particles_vertex!(
                    particle_coords, corner_xi, dxi, nxi, index, parent_cell, args, idx_loop
                )
            end

        elseif $N==3
            for k in -1:1, j in -1:1, i in -1:1
                idx_loop = (i, j, k)
                if idx_loop != (0,0,0)
                    __shuffle_particles_vertex!(
                        particle_coords, corner_xi, dxi, nxi, index, parent_cell, args, idx_loop
                    )
                end
            end
        end
    end

    return nothing
end

function __shuffle_particles_vertex!(
    particle_coords,
    corner_xi,
    dxi,
    nxi,
    index,
    parent_cell::NTuple{N1,Int64},
    args::NTuple{N2,T},
    idx_loop::NTuple{N1,Int64},
) where {N1,N2,T}
    idx_child = child_index(parent_cell, idx_loop)
    # ignore parent cell and "ghost" cells outside the domain
    @inbounds if indomain(idx_child, nxi)
        # iterate over particles in child cell 
        for ip in axes(index, 1)
            if index[ip, idx_child...] # true if memory allocation is filled with a particle
                p_child = cache_particle(particle_coords, ip, idx_child)

                # check whether the incoming particle is inside the cell and move it
                if isincell(p_child, corner_xi, dxi) && !isparticleempty(p_child)
                    # hold particle variables
                    current_p = p_child
                    current_args = cache_args(args, ip, idx_child)

                    # remove particle from child cell
                    index[ip, idx_child...] = false
                    empty_particle!(particle_coords, ip, idx_child)
                    empty_particle!(args, ip, idx_child)

                    # check whether there's empty space in parent cell
                    free_idx = find_free_memory(index, idx_child)
                    free_idx == 0 && continue
                    
                    # move particle and its fields to the first free memory location
                    index[free_idx, parent_cell...] = true
                    fill_particle!(particle_coords, current_p, free_idx, parent_cell)
                    fill_particle!(args, current_args, free_idx, parent_cell)
                end
            end
        end
    end
end

@generated function find_free_memory(index, I::NTuple{N,Int64}) where {N}
    quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i -> @inbounds index[i, I...] == 0 && return i
        return 0
    end
end

@generated function indomain(idx_child::NTuple{N,Integer}, nxi::NTuple{N,Integer}) where {N}
    quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i ->
            @inbounds (1 ≤ idx_child[i] ≤ nxi[i] - 1) == false && return false
        return true
    end
end

@generated function isparticleempty(p::NTuple{N,T}) where {N,T}
    quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N i -> @inbounds isnan(p[i]) && return true
        return false
    end
end

@inline function cache_args(args::NTuple{N1,T}, ip, I::NTuple{N2,Int64}) where {T,N1,N2}
    return ntuple(i -> @inbounds(args[i][ip, I...]), Val(N1))
end

function cache_particle(p::NTuple{N1,T}, ip, I::NTuple{N2,Int64}) where {T,N1,N2}
    return cache_args(p, ip, I)
end

@inline function child_index(parent_cell::NTuple{N,Int64}, I::NTuple{N,Int64}) where {N}
    return ntuple(i -> parent_cell[i] + I[i], Val(N))
end

@generated function empty_particle!(p::NTuple{N1,T}, ip, I::NTuple{N2,Int64}) where {N1, N2, T}
    quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N1 i -> @inbounds p[i][ip, I...] = NaN
    end
end

@generated function fill_particle!(
    p::NTuple{N1,T1}, field::NTuple{N1,T2}, ip, I::NTuple{N2,Int64}
) where {N1,N2,T1,T2}
    quote
        Base.@_inline_meta
        Base.Cartesian.@nexprs $N1 i -> p[i][ip, I...] = field[i]
    end
end

# function _shuffle_particles_vertex!(
#     px, py, grid, dxi, nxi, index, icell, jcell, args::NTuple{N,T}
# ) where {N,T}
#     nx, ny = nxi

#     # closures --------------------------------------
#     @inline child_index(i, j) = (icell + i, jcell + j)
#     @inline function cache_args(
#         args::NTuple{N1,T}, ip, child::Vararg{Int64,N2}
#     ) where {T,N1,N2}
#         return ntuple(i -> args[i][ip, child...], Val(N1))
#     end
#     @inline function find_free_memory(icell, jcell)
#         for i in axes(index, 1)
#             @inbounds index[i, icell, jcell] == 0 && return i
#         end
#         return 0
#     end
#     # -----------------------------------------------

#     # coordinate of the lower-most-left coordinate of the parent cell 
#     corner_xi = corner_coordinate(grid, icell, jcell)
#     # cell where we check for incoming particles
#     parent = icell, jcell
#     # iterate over neighbouring (child) cells
#     for j in -1:1, i in -1:1
#         ichild, jchild = child_index(i, j)
#         # ignore parent cell
#         @inbounds if parent != (ichild, jchild) &&
#             (1 ≤ ichild ≤ nx - 1) &&
#             (1 ≤ jchild ≤ ny - 1)

#             # iterate over particles in child cell 
#             for ip in axes(px, 1)
#                 if index[ip, ichild, jchild] # true if memory allocation is filled with a particle
#                     p_child = (px[ip, ichild, jchild], py[ip, ichild, jchild])

#                     # check whether the incoming particle is inside the cell and move it
#                     if isincell(p_child, corner_xi, dxi)

#                         # hold particle variables
#                         current_px = px[ip, ichild, jchild]
#                         current_py = py[ip, ichild, jchild]
#                         # cache out fields
#                         current_args = cache_args(args, ip, ichild, jchild)

#                         (isnan(current_px) || isnan(current_py)) && continue

#                         # remove particle from child cell
#                         index[ip, ichild, jchild] = false
#                         px[ip, ichild, jchild] = NaN
#                         py[ip, ichild, jchild] = NaN

#                         for t in eachindex(args)
#                             args[t][ip, ichild, jchild] = NaN
#                         end

#                         # check whether there's empty space in parent cell
#                         free_idx = find_free_memory(icell, jcell)
#                         free_idx == 0 && continue

#                         # move it to the first free memory location
#                         index[free_idx, icell, jcell] = true
#                         px[free_idx, icell, jcell] = current_px
#                         py[free_idx, icell, jcell] = current_py

#                         # println("($ip, $icell, $jcell) moved to ($free_idx, $ichild, $jchild)")
#                         # move fields in the memory
#                         for t in eachindex(args)
#                             args[t][free_idx, icell, jcell] = current_args[t]
#                         end
#                     end
#                 end
#             end
#         end
#     end

#     return nothing
# end


function shuffle_particles!(
    particles::Particles, grid, dxi, nxi::NTuple{N,T}, args
) where {N,T}
    # unpack
    (; coords, index, inject, max_xcell, min_xcell) = particles
    nx, ny = nxi
    px, py = coords

    offsets = ((1, 0, 0), (2, 0, 0), (1, 0, 1), (1, 1, 0))
    n_i = ceil(Int, nx * (1 / N))
    n_j = ceil(Int, ny * (1 / N))

    for offset_i in offsets
        offset, offset_x, offset_y = offset_i
        @parallel (1:n_i, 1:n_j) shuffle_particles_ps!(
            px,
            py,
            grid,
            dxi,
            nxi,
            index,
            inject,
            max_xcell,
            min_xcell,
            offset,
            offset_x,
            offset_y,
            args,
        )
    end

    # @assert (px, py) === particles.coords
    @assert index === particles.index

    # (px, py) != particles.coords &&
    #     (@parallel (1:length(px)) copy_vectors!(particles.coords, (px, py)))

    return nothing
end

@parallel_indices (icell, jcell) function shuffle_particles_ps!(
    px,
    py,
    grid,
    dxi::NTuple{2,T},
    nxi,
    index,
    inject,
    max_xcell,
    min_xcell,
    offset,
    offset_x,
    offset_y,
    args,
) where {T}
    nx, ny = nxi
    i = offset + 2 * (icell - 1) + offset_x
    j = offset + 2 * (jcell - 1) + offset_y

    if (i ≤ nx) && (j ≤ ny)
        _shuffle_particles!(
            px, py, grid, dxi, nxi, index, inject, max_xcell, min_xcell, i, j, args
        )
    end
    return nothing
end

function _shuffle_particles!(
    px,
    py,
    grid,
    dxi,
    nxi,
    index,
    inject,
    max_xcell,
    min_xcell,
    icell,
    jcell,
    args::NTuple{N,T},
) where {N,T}
    nx, ny = nxi

    # closures --------------------------------------
    @inline child_index(i, j) = (clamp(icell + i, 1, nx), clamp(jcell + j, 1, ny))
    @inline function cache_args(
        args::NTuple{N1,T}, ip, child::Vararg{Int64,N2}
    ) where {T,N1,N2}
        return ntuple(i -> args[i][ip, child...], Val(N1))
    end
    @inline function find_free_memory(icell, jcell)
        for i in axes(index, 1)
            @inbounds index[i, icell, jcell] == 0 && return i
        end
        return 0
    end
    # -----------------------------------------------

    # coordinate of the lower-most-left coordinate of the parent cell 
    corner_xi = corner_coordinate(grid, icell, jcell) .- dxi .* 0.5
    # cell where we check for incoming particles
    parent = icell, jcell
    # iterate over neighbouring (child) cells
    for i in -1:1, j in -1:1
        ichild, jchild = child_index(i, j)
        # ignore parent cell
        if parent != (ichild, jchild)

            # iterate over particles in child cell 
            for ip in axes(px, 1)
                if index[ip, ichild, jchild]
                    p_child = (px[ip, ichild, jchild], py[ip, ichild, jchild])

                    # check whether the incoming particle is inside the cell and move it
                    if isincell(p_child, corner_xi, dxi)

                        # hold particle variables to move
                        current_px = px[ip, ichild, jchild]
                        current_py = py[ip, ichild, jchild]
                        # cache out fields to move in the memory
                        current_args = cache_args(args, ip, ichild, jchild)

                        (isnan(current_px) || isnan(current_py)) && continue

                        # remove particle from child cell
                        index[ip, ichild, jchild] = false
                        px[ip, ichild, jchild] = NaN
                        py[ip, ichild, jchild] = NaN

                        for t in eachindex(args)
                            args[t][ip, ichild, jchild] = NaN
                        end

                        # check whether there's empty space in parent cell
                        free_idx = find_free_memory(icell, jcell)
                        free_idx == 0 && continue

                        # move it to the first free memory location
                        index[free_idx, icell, jcell] = true
                        px[free_idx, icell, jcell] = current_px
                        py[free_idx, icell, jcell] = current_py

                        # println("($ip, $icell, $jcell) moved to ($free_idx, $ichild, $jchild)")
                        # move fields in the memory
                        for t in eachindex(args)
                            args[t][free_idx, icell, jcell] = current_args[t]
                        end
                    end
                end
            end

        else # double check particle is inside parent cell

            # # iterate over particles in child cell 
            # for ip in axes(px, 1)
            #     if index[ip, icell, jcell]
            #         p_child = (px[ip, icell, jcell], py[ip, icell, jcell])

            #         # check whether the particle is inside the cell and move it
            #         if !isincell(p_child, corner_xi, dxi)

            #             @show p_child, corner_xi
            #             index[ip, icell, jcell] = false
            #             px[ip, icell, jcell] = NaN
            #             py[ip, icell, jcell] = NaN

            #             for t in eachindex(args)
            #                 args[t][ip, icell, jcell] = NaN
            #             end

            #         end
            #     end
            # end

        end
    end

    # true if cell is totally empty (i.e. we need to inject new particles in it)
    # inject[icell, jcell] = isemptycell(icell, jcell, index, min_xcell)

    return nothing
end

@inline function find_free_memory(icell, jcell)
    for i in axes(index, 1)
        @inbounds index[i, icell, jcell] == 0 && return i
    end
    return 0
end


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

    if (i < nx) && (j < ny)
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
    first_cell_index(i) = (i - 1) * max_xcell + 1
    idx_range(i) = i:(i + max_xcell - 1)

    function find_free_memory(indices)
        for i in indices
            index[i] == 0 && return i
        end
        return 0
    end
    # -----------------------------------------------

    # current (parent) cell (i.e. cell in the center of the cell-block)
    parent = cart2lin(icell, jcell, nx - 1)
    # coordinate of the lower-most-left coordinate of the parent cell 
    corner_xi = corner_coordinate(grid, icell, jcell)
    i0_parent = first_cell_index(parent)

    # # iterate over neighbouring (child) cells
    for child in neighbouring_cells(icell, jcell, nx, ny)

        # ignore parent cell
        if parent != child
            # index where particles inside the child cell start in the particle array
            i0_child = first_cell_index(child)

            # iterate over particles in child cell 
            for j in idx_range(i0_child)
                if index[j]
                    p_child = (px[j], py[j])

                    # check that the particle is inside the grid
                    if isincell(p_child, corner_xi, dxi)
                        # hold particle variables to move
                        current_px = px[j]
                        current_py = py[j]

                        (isnan(current_px) || isnan(current_py)) && continue

                        current_args = ntuple(i -> args[i][j], Val(N))

                        # remove particle from old cell
                        index[j] = false
                        px[j] = NaN
                        py[j] = NaN

                        for k in eachindex(args)
                            args[k][j] = NaN
                        end

                        # move particle to new cell
                        free_idx = find_free_memory(idx_range(i0_parent))
                        free_idx == 0 && continue

                        # move it to the first free memory location
                        index[free_idx] = true
                        px[free_idx] = current_px
                        py[free_idx] = current_py

                        for k in eachindex(args)
                            args[k][free_idx] = current_args[k]
                        end
                    end
                end
            end
        end
    end

    # true if cell is totally empty (i.e. we need to inject new particles in it)
    inject[parent] = isemptycell(i0_parent, index, max_xcell, min_xcell)

    return nothing
end

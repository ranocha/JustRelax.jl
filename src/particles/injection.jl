# function inject_particles!(particles::Particles, grid, nxi, dxi)
#     # unpack
#     (; inject, coords, nxcell, max_xcell) = particles
#     dx, dy = dxi
#     px, py = coords

#     # closures 
#     first_cell_index(i) = (i - 1) * max_xcell + 1
#     myrand() = (1.0 + (rand()-0.5)*0.25)

#     # linear to cartesian object
#     i2s = CartesianIndices(nxi.-1)

#     for (cell, injection) in enumerate(inject)
#         if injection
#             icell, jcell = i2s[cell].I
#             xc, yc = corner_coordinate(grid, icell, jcell)
#             idx = first_cell_index(cell)

#             # add 4 new particles in a 2x2 manner + some small random perturbation
#             px[idx]   = xc + dx*(1/3)*myrand()
#             px[idx+1] = xc + dx*(2/3)*myrand()
#             px[idx+2] = xc + dx*(1/3)*myrand()
#             px[idx+3] = xc + dx*(2/3)*myrand()
#             py[idx]   = yc + dy*(1/3)*myrand()
#             py[idx+1] = yc + dy*(1/3)*myrand()
#             py[idx+2] = yc + dy*(2/3)*myrand()
#             py[idx+3] = yc + dy*(2/3)*myrand()

#             for i in idx:(idx+nxcell-1)
#                 particles.index[i] = true
#             end

#             inject[cell] = false
#         end
#     end

# end

# function inject_particles!(particles::Particles, grid, nxi, dxi)
#     # unpack
#     (; inject, coords, index, nxcell, max_xcell) = particles
#     # linear to cartesian object
#     i2s = CartesianIndices(nxi .- 1)
#     ncells = length(inject)
#     @parallel (1:ncells) inject_particles!(
#         inject, coords, index, nxcell, max_xcell, grid, dxi, i2s
#     )
# end

# @parallel_indices (cell) function inject_particles!(
#     inject, coords, index, nxcell, max_xcell, grid, dxi, i2s
# )
#     if cell ≤ length(inject)
#         _inject_particles!(inject, coords, index, nxcell, max_xcell, grid, dxi, i2s, cell)
#     end
#     return nothing
# end

# function _inject_particles!(inject, coords, index, nxcell, max_xcell, grid, dxi, i2s, cell)
#     dx, dy = dxi
#     # px, py = coords

#     # closures -----------------------------------
#     first_cell_index(i) = (i - 1) * max_xcell + 1
#     myrand() = (1.0 + (rand() - 0.5) * 0.25)
#     # --------------------------------------------

#     if inject[cell]
#         icell, jcell = i2s[cell].I
#         xc, yc = corner_coordinate(grid, icell, jcell)
#         idx = first_cell_index(cell)
#         # add 4 new particles in a 2x2 manner + some small random perturbation
#         coords[1][idx] = xc + dx * (1 / 3) * myrand()
#         coords[1][idx + 1] = xc + dx * (2 / 3) * myrand()
#         coords[1][idx + 2] = xc + dx * (1 / 3) * myrand()
#         coords[1][idx + 3] = xc + dx * (2 / 3) * myrand()
#         coords[2][idx] = yc + dy * (1 / 3) * myrand()
#         coords[2][idx + 1] = yc + dy * (1 / 3) * myrand()
#         coords[2][idx + 2] = yc + dy * (2 / 3) * myrand()
#         coords[2][idx + 3] = yc + dy * (2 / 3) * myrand()
#         for i in idx:(idx + nxcell - 1)
#             index[i] = true
#         end
#         inject[cell] = false
#     end
# end

@inline check_injection(inject::AbstractArray) = sum(inject) > 0 ? true : false

function check_injection(particles::Particles{N,A,B,C,D,E}) where {N,A,B,C,D,E}
    (; inject, index, min_xcell) = particles
    # nx, ny = size(particles.index, 2), size(particles.index, 3)
    _, nxi... = size(particles.index)
    ranges = ntuple(i -> 1:nxi[i], Val(N))

    @parallel ranges check_injection!(inject, index, min_xcell)

    return check_injection(particles.inject)
end

@parallel_indices (icell, jcell) function check_injection!(inject, index, min_xcell)
    if icell ≤ size(index, 2) && jcell ≤ size(index, 3)
        inject[icell, jcell] = isemptycell(icell, jcell, index, min_xcell)
    end
    return nothing
end

@parallel_indices (icell, jcell, kcell) function check_injection!(inject, index, min_xcell)
    if icell ≤ size(index, 2) && jcell ≤ size(index, 3) && kcell ≤ size(index, 4)
        inject[icell, jcell, kcell] = isemptycell(icell, jcell, index, min_xcell)
    end
    return nothing
end

function inject_particles!(particles::Particles, args, fields, grid::NTuple{2,T}) where {T}
    # unpack
    (; inject, coords, index, nxcell) = particles
    # linear to cartesian object
    icell, jcell = size(inject)
    dxi = compute_dx(grid)

    @parallel (1:icell, 1:jcell) inject_particles!(
        inject, args, fields, coords, index, grid, dxi, nxcell
    )
end

@parallel_indices (icell, jcell) function inject_particles!(
    inject, args, fields, coords, index, grid, dxi::NTuple{2,T}, nxcell
) where {T}
    if (icell ≤ size(inject, 1)) && (jcell ≤ size(inject, 2))
        _inject_particles!(
            inject, args, fields, coords, index, grid, dxi, nxcell, (icell, jcell)
        )
    end
    return nothing
end

function inject_particles!(particles::Particles, args, fields, grid::NTuple{3,T}) where {T}
    # unpack
    (; inject, coords, index, nxcell) = particles
    # linear to cartesian object
    icell, jcell, kcell = size(inject)
    dxi = compute_dx(grid)

    @parallel (1:icell, 1:jcell, 1:kcell) inject_particles!(
        inject, args, fields, coords, index, grid, dxi, nxcell
    )
end

@parallel_indices (icell, jcell, kcell) function inject_particles!(
    inject, args, fields, coords, index, grid, dxi::NTuple{3,T}, nxcell
) where {T}
    if (icell ≤ size(inject, 1)) && (jcell ≤ size(inject, 2)) && (kcell ≤ size(inject, 3))
        _inject_particles!(
            inject, args, fields, coords, index, grid, dxi, nxcell, (icell, jcell, kcell)
        )
    end
    return nothing
end

function _inject_particles!(
    inject, args, fields, coords, index, grid, dxi, nxcell, idx_cell
)
    max_xcell = size(index, 1)

    # closures -----------------------------------
    first_cell_index(i) = (i - 1) * max_xcell + 1
    # --------------------------------------------

    @inbounds if inject[idx_cell...]
        # count current number of particles inside the cell
        particles_num = false
        for i in 1:max_xcell
            particles_num += index[i, idx_cell...]
        end

        # coordinates of the lower-left center
        xvi = corner_coordinate(grid, idx_cell)

        for i in 1:max_xcell
            if index[i, idx_cell...] === false
                particles_num += 1

                # add at cellcenter + small random perturbation
                p_new = new_particle(xvi, dxi)
                fill_particle!(coords, p_new, i, idx_cell)
                index[i, idx_cell...] = true

                for (arg_i, field_i) in zip(args, fields)
                    tmp = _grid2particle_xvertex(p_new, grid, dxi, field_i, idx_cell)
                    arg_i[i, idx_cell...] = clamp(tmp, extrema(field_i)...)
                end
            end

            particles_num == nxcell && break
        end
    end

    return inject[idx_cell...] = false
end

function new_particle(xvi::NTuple{N,T}, dxi::NTuple{N,T}) where {N,T}
    f() = rand(-1:2:1) * rand() * 0.25
    p_new = ntuple(Val(N)) do i
        # xvi[i] + dxi[i] * 0.5 * (1.0 + f())
        xvi[i] + dxi[i] * rand()
    end

    return p_new
end

# function _inject_particles!(
#     inject, args, fields, coords, index, grid, dxi, nxcell, icell, jcell
# )
#     dx, dy = dxi
#     max_xcell = size(index, 1)

#     # closures -----------------------------------
#     first_cell_index(i) = (i - 1) * max_xcell + 1
#     myrand() = rand(-1:2:1) * rand() * 0.5
#     # --------------------------------------------

#     @inbounds if inject[icell, jcell]
#         # count current number of particles inside the cell
#         particles_num = false
#         for i in 1:max_xcell
#             particles_num += index[i, icell, jcell]
#         end

#         # coordinates of the lower-left center
#         xv, yv = corner_coordinate(grid, (icell, jcell))

#         for i in 1:max_xcell
#             if index[i, icell, jcell] === false
#                 particles_num += 1

#                 # add at cellcenter + small random perturbation
#                 px_new = xv + dx * 0.5 * (1.0 + myrand())
#                 py_new = yv + dy * 0.5 * (1.0 + myrand())
#                 p_new = (px_new, py_new)
#                 coords[1][i, icell, jcell] = px_new
#                 coords[2][i, icell, jcell] = py_new
#                 index[i, icell, jcell] = true

#                 for (arg_i, field_i) in zip(args, fields)
#                     tmp = _grid2particle_xvertex(p_new, grid, dxi, field_i, icell, jcell)
#                     arg_i[i, icell, jcell] = clamp(tmp, extrema(field_i)...)
#                     # arg_i[i, icell, jcell] = field_i[icell, jcell]
#                     # arg_i[i, icell, jcell] = 0.0
#                 end
#             end

#             particles_num == nxcell && break
#         end
#     end

#     return inject[icell, jcell] = false
# end

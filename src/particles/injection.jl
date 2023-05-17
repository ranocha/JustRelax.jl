@inline check_injection(inject::AbstractArray) = count(inject) > 0 # ? true : false

function check_injection(particles::Particles{N, A}) where {N, A}
    (; inject, index, min_xcell) = particles
    nxi = size(index)
    ranges = ntuple(i -> 1:nxi[i], Val(N))

    @parallel ranges check_injection!(inject, index, min_xcell)

    return check_injection(particles.inject)
end

@parallel_indices (icell, jcell) function check_injection!(inject::AbstractMatrix, index, min_xcell)
    if icell ≤ size(index, 1) && jcell ≤ size(index, 2)
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
    max_xcell = cellnum(index)

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
                    local_field = cell_field(field_i, idx_cell...)
                    upper = maximum(local_field)
                    lower = minimum(local_field)
                    tmp = _grid2particle_xvertex(p_new, grid, dxi, field_i, idx_cell)
                    tmp < lower && (tmp = lower)
                    tmp > upper && (tmp = upper)
                    arg_i[i, idx_cell...] = tmp
                    # arg_i[i, idx_cell...] = clamp(tmp, extrema(field_i)...)
                end
            end

            particles_num == nxcell && break
        end
    end

    return inject[idx_cell...] = false
end


function inject_particles_phase!(particles::Particles, particles_phases, args, fields, grid::NTuple{2,T}) where {T}
    # unpack
    (; inject, coords, index, min_xcell) = particles
    # linear to cartesian object
    icell, jcell = size(inject)
    dxi = compute_dx(grid)

    @parallel (1:icell, 1:jcell) inject_particles_phase!(
        inject, particles_phases, args, fields, coords, index, grid, dxi, min_xcell
    )
end

@parallel_indices (icell, jcell) function inject_particles_phase!(
    inject, particles_phases, args, fields, coords, index, grid, dxi::NTuple{2,T}, min_xcell
) where {T}
    if (icell ≤ size(inject, 1)) && (jcell ≤ size(inject, 2))
        _inject_particles_phase!(
            inject, particles_phases, args, fields, coords, index, grid, dxi, min_xcell, (icell, jcell)
        )
    end
    return nothing
end

function _inject_particles_phase!(
    inject, particles_phases, args, fields, coords, index, grid, dxi, min_xcell, idx_cell
)

    if inject[idx_cell...]
      
        # count current number of particles inside the cell
        # particles_num = reduce(+, @cell(index[i, idx_cell...]) for i in cellaxes(index))
        particles_num = false
        for i in cellaxes(index)
            particles_num += @cell index[i, idx_cell...]
        end
      
        # coordinates of the lower-left center
        xvi = corner_coordinate(grid, idx_cell)

        for i in cellaxes(index)
            if !(@cell(index[i, idx_cell...]))
                particles_num += 1

                # add at cellcenter + small random perturbation
                p_new = new_particle(xvi, dxi)

                # add phase to new particle
                # idx_min = index_min_distance(coords, p_new, i, idx_cell...)
                # particles_phases[i, idx_cell...] = particles_phases[idx_min, idx_cell...]
                particle_idx, i_idx, j_idx = index_min_distance(coords, p_new, index, i, idx_cell...)
                new_phase = @cell particles_phases[particle_idx, i_idx, j_idx]
                @cell particles_phases[i, idx_cell...] = new_phase

                fill_particle!(coords, p_new, i, idx_cell)
                @cell index[i, idx_cell...] = true

                for (arg_i, field_i) in zip(args, fields)
                    local_field = cell_field(field_i, idx_cell...)
                    upper = maximum(local_field)
                    lower = minimum(local_field)
                    tmp = _grid2particle_xvertex(p_new, grid, dxi, field_i, idx_cell)
                    tmp < lower && (tmp = lower)
                    tmp > upper && (tmp = upper)
                    @cell arg_i[i, idx_cell...] = tmp
                    # arg_i[i, idx_cell...] = clamp(tmp, extrema(field_i)...)
                end
            end

            particles_num ≥ min_xcell && break
        end
    end

    inject[idx_cell...] = false

    return nothing
end

@inline distance2(x, y) = mapreduce(x -> (x[1]-x[2])^2, +, zip(x,y)) |> sqrt

# function index_min_distance(coords, pn, current_cell, idx_cell::Vararg{Int, N}) where N
#     idx_min = 0
#     dist_min = Inf
#     px, py = coords
#     for ip in axes(px, 1)
#         ip==current_cell && continue
#         isnan(px[ip, idx_cell...]) && continue
#         pxi = px[ip, idx_cell...], py[ip, idx_cell...]
#         d = distance(pxi, pn)
#         if d < dist_min
#             idx_min = ip
#             dist_min = d
#         end
#     end

#     idx_min
# end

function index_min_distance(coords, pn, index, current_cell, icell, jcell)
   
    particle_idx_min = i_idx_min = j_idx_min =  0
    dist_min = Inf
    px, py = coords
    nx, ny = size(px, 1), size(px, 2)

    for j in jcell-1:jcell+1, i in icell-1:icell+1, ip in cellaxes(index)
        
        # early escape conditions
        ((i < 1) || (j < 1)) && continue # out of the domain
        ((i > nx) || (j > ny)) && continue # out of the domain
        (i == icell) && (j == jcell) && (ip == current_cell) && continue # current injected particle
        !(@cell index[ip, i, j]) && continue

        # distance from new point to the existing particle        
        pxi = @cell(px[ip, i, j]), @cell(py[ip, i, j])
        d = distance(pxi, pn)

        if d < dist_min
            particle_idx_min = ip
            i_idx_min = i
            j_idx_min = j
            dist_min = d
        end
    end

    particle_idx_min, i_idx_min, j_idx_min
end


@inline cell_field(field, i, j) = field[i, j], field[i+1, j], field[i, j+1], field[i+1, j+1]

function new_particle(xvi::NTuple{N,T}, dxi::NTuple{N,T}) where {N,T}
    
    p_new = ntuple(Val(N)) do i
        xvi[i] + dxi[i] * rand(0.05:1e-5: 0.95)
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
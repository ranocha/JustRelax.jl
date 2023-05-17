# import .StencilInterpolations: normalize_coordinates, ndlinear

# INTERPOLATION METHODS


function _grid2particle_xcell_edge(
    p_i::NTuple, xi_vx::NTuple, dxi::NTuple, F::AbstractArray, idx
)

    # F at the cell corners
    Fi, xci = edge_nodes(F, p_i, xi_vx, dxi, idx)
    # normalize particle coordinates
    ti = normalize_coordinates(p_i, xci, dxi)
    # Interpolate field F onto particle
    Fp = ndlinear(ti, Fi)

    return Fp
end

# Get field F at the centers of a given cell
@inline @inbounds function edge_nodes(
    F::AbstractArray{T,2}, p_i, xi_vx, dxi, idx::NTuple{2,Integer}
) where {T}
    # unpack
    idx_x, idx_y = idx
    px, py = p_i
    dx, dy = dxi
    x_vx, y_vx = xi_vx
    @inbounds xv = x_vx[idx_x]
    @inbounds yv = y_vx[idx_y]
    # compute offsets and corrections
    # offset_x = (px - xv) > 0 ? 0 : 1
    # offset_y = (py - yv) > 0 ? 0 : 1
    offset_x = vertex_offset(xv, px, dx)
    offset_y = vertex_offset(yv, py, dy)
    # cell indices
    idx_x += offset_x
    idx_y += offset_y
    # coordinates of lower-left corner of the cell
    @inbounds xcell = x_vx[idx_x]
    @inbounds ycell = y_vx[idx_y]

    # F at the four centers
    Fi = @inbounds (
        F[idx_x, idx_y], F[idx_x + 1, idx_y], F[idx_x, idx_y + 1], F[idx_x + 1, idx_y + 1]
    )

    return Fi, (xcell, ycell)
end

@inline normalised_distance(xi, pxi, di) = (pxi-xi)/di

@inline function vertex_offset(xi, pxi, di)
    dist = normalised_distance(xi, pxi, di)
        dist >  2 && return  2
    2 > dist >  1 && return  1
   -1 < dist <  0 && return -1
        dist < -1 && return -2
    return 0
end

@inline function edge_nodes(
    F::AbstractArray{T,3}, p_i, xi_vx, dxi, idx::NTuple{3,Integer}
) where {T}
    # unpack
    idx_x, idx_y, idx_z = idx
    @inbounds px = p_i[1]
    @inbounds dx = dxi[1]
    x_vx, y_vx, z_vx = xi_vx
    @inbounds xc = x_vx[idx_x]
    xv = xc + 0.5 * dx
    # compute offsets and corrections
    offset_x = (px - xv) > 0 ? 0 : 1
    offset_y = (py - yv) > 0 ? 0 : 1
    offset_z = (pz - zv) > 0 ? 0 : 1
    # cell indices
    idx_x += offset_x
    idx_y += offset_y
    idx_z += offset_z
    # coordinates of lower-left corner of the cell
    @inbounds xcell = x_vx[idx_x]
    @inbounds ycell = y_vx[idx_y]
    @inbounds zcell = z_vx[idx_z]

    # F at the four centers
    Fi = @inbounds  (
        F[idx_x, idx_y, idx_z], # v000
        F[idx_x + 1, idx_y, idx_z], # v100
        F[idx_x, idx_y, idx_z + 1], # v001
        F[idx_x + 1, idx_y, idx_z + 1], # v101
        F[idx_x, idx_y + 1, idx_z], # v010
        F[idx_x + 1, idx_y + 1, idx_z], # v110
        F[idx_x, idx_y + 1, idx_z + 1], # v011
        F[idx_x + 1, idx_y + 1, idx_z + 1], # v111
    )

    return Fi, (xcell, ycell, zcell)
end

# ADVECTION METHODS 

function advection_RK2!(
    particles::Particles, V, grid_vx::NTuple{2,T}, grid_vy::NTuple{2,T}, dt, α
) where {T}
    # unpack 
    (; coords, index, max_xcell) = particles
    px, = coords
    # compute some basic stuff
    dxi = compute_dx(grid_vx)
    # grid_lims = (extrema(grid_vx[1]), extrema(grid_vy[2]))
    # clamped_limits = clamp_grid_lims(grid_lims, dxi)
    clamped_limits = (
        extrema(grid_vx[1]),
        extrema(grid_vy[2])
    )

    nx, ny = size(px)
    # _, nx, ny = size(px)
    # Need to transpose grid_vy and Vy to reuse interpolation kernels
    grid_vi = grid_vx, grid_vy
    # launch parallel advection kernel
    @parallel (1:max_xcell, 1:nx, 1:ny) advection_RK2_edges!(
        coords, V, index, grid_vi, clamped_limits, dxi, dt, α
    )

    return nothing
end

@parallel_indices (ipart, icell, jcell) function advection_RK2_edges!(
    p,
    V::NTuple{2,AbstractArray{T,N}},
    index::AbstractArray,
    grid,
    clamped_limits,
    dxi,
    dt,
    α,
) where {T,N}
    
    px, py = p

    if icell ≤ size(px, 1) && jcell ≤ size(px, 2) && @cell index[ipart, icell, jcell]
        pᵢ = (@cell(px[ipart, icell, jcell]), @cell(py[ipart, icell, jcell]))
        if !any(isnan, pᵢ)
            px_new, py_new = _advection_RK2_edges(
                pᵢ, V, grid, dxi, clamped_limits, dt, (icell, jcell), α
            )
            @cell px[ipart, icell, jcell] = px_new
            @cell py[ipart, icell, jcell] = py_new
        end
    end

    return nothing
end

@parallel_indices (icell, jcell, kcell) function advection_RK2_edges!(
    p,
    V::NTuple{3,AbstractArray{T,N}},
    index::AbstractArray,
    grid,
    clamped_limits,
    dxi,
    dt,
    α,
) where {T,N}
    px, py, pz = p

    for ipart in axes(px, 1)
        if icell ≤ size(px, 2) &&
            jcell ≤ size(px, 3) &&
            kcell ≤ size(px, 4) &&
            index[ipart, icell, jcell, kcell]
            pᵢ = (
                px[ipart, icell, jcell, kcell],
                py[ipart, icell, jcell, kcell],
                pz[ipart, icell, jcell, kcell],
            )
            if !any(isnan, pᵢ)
                px[ipart, icell, jcell, kcell], py[ipart, icell, jcell, kcell], pz[ipart, icell, jcell, kcell] = _advection_RK2_edges(
                    pᵢ, V, grid, dxi, clamped_limits, dt, (icell, jcell, kcell), α
                )
            end
        end
    end

    return nothing
end

"""
    y ← y + h*( (1-1/2/α)*f(t,y) + (1/2/α) * f(t, y+α*h*f(t,y)) )
    α = 0.5 ==> midpoint
    α = 1 ==> Heun
    α = 2/3 ==> Ralston
"""
function _advection_RK2_edges(
    p0::NTuple{N,T},
    V::NTuple{N,AbstractArray{T,N}},
    grid_vi,
    dxi,
    clamped_limits,
    dt,
    idx::NTuple,
    α,
) where {T,N}
    _α = inv(α)
    ValN = Val(N)

    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        Base.@_inline_meta
        # _grid2particle_xcell_edge(flip(p0, i), grid_vi[i], flip(dxi, i), V[i], flip(idx, i))
        _grid2particle_xcell_edge(p0, grid_vi[i], dxi, V[i], idx)
    end

    # advect α*dt
    p1 = ntuple(ValN) do i
        Base.@_inline_meta
        muladd(vp0[i] * α, dt, p0[i])
        # clamp(xtmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    # interpolate velocity to new location
    vp1 = ntuple(ValN) do i
        Base.@_inline_meta
        _grid2particle_xcell_edge(p1, grid_vi[i], dxi, V[i], idx)
        # _grid2particle_xcell_edge(flip(p1, i), grid_vi[i], flip(dxi, i), V[i], flip(idx, i))
    end

    # final advection
    pf = ntuple(ValN) do i
        Base.@_inline_meta
        if α == 0.5
            @muladd p0[i] + dt * vp1[i]
        else
            @muladd p0[i] + dt * ((1.0 - 0.5 * _α) * vp0[i] + 0.5 * _α * vp1[i])
        end
        # clamp(ptmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    return pf
end

@inline function flip(x::NTuple{2,T}, i) where {T}
    i == 1 && return x
    i == 2 && return (x[2], x[1])
end

@inline function flip(x::NTuple{3,T}, i) where {T}
    i == 1 && return x
    i == 2 && return (x[2], x[1], x[3])
    i == 3 && return (x[3], x[2], x[1])
end

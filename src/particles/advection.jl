function advection_RK2!(particles::Particles, V, grid::NTuple{2,T}, dt, α) where {T}
    # unpack 
    (; coords, index, max_xcell) = particles
    px, py = coords
    # compute some basic stuff
    dxi = compute_dx(grid)
    grid_lims = extrema.(grid)
    clamped_limits = clamp_grid_lims(grid_lims, dxi)
    _, nx, ny = size(px)
    xci_augmented = augment_lazy_grid(grid, dxi)
    # launch parallel advection kernel
    @parallel (1:max_xcell, 1:nx, 1:ny) advection_RK2!(
        px, py, V, index, grid, xci_augmented, clamped_limits, dxi, dt, α
    )

    return nothing
end

@parallel_indices (ipart, icell, jcell) function advection_RK2!(
    px,
    py,
    V::NTuple{2,T1},
    index::AbstractArray,
    grid,
    xci_augmented,
    clamped_limits,
    dxi,
    dt,
    α,
) where {T1}
    if i ≤ size(px, 2) && j ≤ size(px, 3) && index[ipart, icell, jcell]
        pᵢ = (px[ipart, icell, jcell], py[ipart, icell, jcell])
        if !any(isnan, pᵢ)
            px[ipart, icell, jcell], py[ipart, icell, jcell] = _advection_RK2(
                pᵢ, V, grid, xci_augmented, dxi, clamped_limits, dt, i, j; α=α
            )
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
function _advection_RK2(
    p0::NTuple{N,T},
    v0::NTuple{N,AbstractArray{T,N}},
    grid,
    xci_augmented,
    dxi,
    clamped_limits,
    dt,
    icell,
    jcell;
    α=0.5,
) where {T,N}
    _α = inv(α)
    ValN = Val(N)

    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        _grid2particle_xcell_centered(p0, grid, xci_augmented, dxi, v0[i], icell, jcell)
    end

    # advect α*dt
    p1 = ntuple(ValN) do i
        xtmp = p0[i] + vp0[i] * α * dt
        clamp(xtmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    # interpolate velocity to new location
    vp1 = ntuple(ValN) do i
        _grid2particle_xcell_centered(p1, grid, xci_augmented, dxi, v0[i], icell, jcell)
    end

    # final advection
    pf = ntuple(ValN) do i
        ptmp = if α == 0.5
            @muladd p0[i] + dt * vp1[i]
        else
            @muladd p0[i] + dt * ((1.0 - 0.5 * _α) * vp0[i] + 0.5 * _α * vp1[i])
        end
        clamp(ptmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    return pf
end

# ADVECTION WHEN VELOCITIES ARE LOCATED AT THE GRID NODES

function advection_RK2_vertex!(particles::Particles, V, grid::NTuple{2,T}, dt; α=0.5) where {T}
    # unpack 
    (; coords, index, max_xcell) = particles
    px, py = coords
    # compute some basic stuff
    dxi = compute_dx(grid)
    grid_lims = extrema.(grid)
    clamped_limits = clamp_grid_lims(grid_lims, dxi)
    _, nx, ny = size(px)
    # launch parallel advection kernel
    @parallel (1:max_xcell, 1:nx-1, 1:ny-1) advection_RK2_vertex!(
        px, py, V, index, grid, clamped_limits, dxi, dt, α
    )

    return nothing
end

@parallel_indices (ipart, icell, jcell) function advection_RK2_vertex!(
    px,
    py,
    V::NTuple{2,T1},
    index::AbstractArray,
    grid,
    clamped_limits,
    dxi,
    dt,
    α,
) where {T1}
    if icell ≤ size(px, 2) && jcell ≤ size(px, 3) && index[ipart, icell, jcell]
        pᵢ = (px[ipart, icell, jcell], py[ipart, icell, jcell])
        if !any(isnan, pᵢ)
            px[ipart, icell, jcell], py[ipart, icell, jcell] = _advection_RK2_vertex(
                pᵢ, V, grid, dxi, clamped_limits, dt, icell, jcell, α
            )
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
function _advection_RK2_vertex(
    p0::NTuple{N,T},
    v0::NTuple{N,AbstractArray{T,N}},
    grid,
    dxi,
    clamped_limits,
    dt,
    icell,
    jcell,
    α,
) where {T,N}
    _α = inv(α)
    ValN = Val(N)

    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        _grid2particle_xvertex(p0, grid, dxi, v0[i], (icell, jcell))
    end

    # advect α*dt
    p1 = ntuple(ValN) do i
        xtmp = p0[i] + vp0[i] * α * dt
        clamp(xtmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    # interpolate velocity to new location
    vp1 = ntuple(ValN) do i
        _grid2particle_xvertex(p1, grid, dxi, v0[i], (icell, jcell))
    end

    # final advection
    pf = ntuple(ValN) do i
        ptmp = if α == 0.5
            @muladd p0[i] + dt * vp1[i]
        else
            @muladd p0[i] + dt * ((1.0 - 0.5 * _α) * vp0[i] + 0.5 * _α * vp1[i])
        end
        clamp(ptmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    return pf
end


# ADVECTION FOR A GENERIC PARTICLES CLOUD

function advection_RK2_vertex!(coords::NTuple{2, T}, V, grid::NTuple{2,F}, dt; α=0.5) where {T, F}
    # unpack 
    px, py = coords
    # compute some basic stuff
    dxi = compute_dx(grid)
    grid_lims = extrema.(grid)
    clamped_limits = clamp_grid_lims(grid_lims, dxi)
    # launch parallel advection kernel
    np = length(px)
    @parallel (1:np) advection_RK2_vertex!(
        px, py, V, grid, clamped_limits, dxi, dt, α
    )

    return nothing
end

@parallel_indices (ipart) function advection_RK2_vertex!(
    px,
    py,
    V::NTuple{2,T},
    grid::NTuple{2,F},
    clamped_limits,
    dxi,
    dt,
    α,
) where {T, F}
    if ipart ≤ length(px)
        px[ipart], py[ipart] = _advection_RK2_vertex(
            (px[ipart], py[ipart]), V, grid, dxi, clamped_limits, dt, α
        )
    end
    return nothing
end

"""
    y ← y + h*( (1-1/2/α)*f(t,y) + (1/2/α) * f(t, y+α*h*f(t,y)) )
    α = 0.5 ==> midpoint
    α = 1 ==> Heun
    α = 2/3 ==> Ralston
"""
function _advection_RK2_vertex(
    p0::NTuple{N,T},
    v0::NTuple{N,AbstractArray{T,N}},
    grid,
    dxi,
    clamped_limits,
    dt,
    α,
) where {T,N}
    _α = inv(α)
    ValN = Val(N)
    origin = minimum.(grid)

    # interpolate velocity to current location
    vp0 = ntuple(ValN) do i
        _grid2particle(p0, origin, grid, dxi, v0[i])
    end

    # advect α*dt
    p1 = ntuple(ValN) do i
        xtmp = p0[i] + vp0[i] * α * dt
        clamp(xtmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    # interpolate velocity to new location
    vp1 = ntuple(ValN) do i
        _grid2particle(p1, origin,  grid, dxi, v0[i])
    end

    # final advection
    pf = ntuple(ValN) do i
        ptmp = if α == 0.5
            @muladd p0[i] + dt * vp1[i]
        else
            @muladd p0[i] + dt * ((1.0 - 0.5 * _α) * vp0[i] + 0.5 * _α * vp1[i])
        end
        clamp(ptmp, clamped_limits[i][1], clamped_limits[i][2])
    end

    return pf
end
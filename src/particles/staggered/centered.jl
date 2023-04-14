
function int2part!(Fp, F, F0, particles::Particles, grid::NTuple{2,T}; α=1.0) where {T}
    (; coords, index, max_xcell) = particles
    dxi = compute_dx(grid)
    px, py = coords
    _, nx, ny = size(px)

    xci_augmented = augment_lazy_grid(grid, dxi)

    @parallel (1:max_xcell, 1:nx, 1:ny) int2part!(
        Fp, F, F0, px, py, index, grid, xci_augmented, dxi, α
    )

    return nothing
end

@parallel_indices (ipart, i, j) function int2part!(
    Fp, F, F0, px, py, index, grid, xci_augmented, dxi, α
)
    if i ≤ size(px, 2) && j ≤ size(px, 3) && index[ipart, i, j]
        pᵢ = (px[ipart, i, j], py[ipart, i, j])
        _int2part!(pᵢ, Fp, F, F0, grid, xci_augmented, dxi, i, j, α)
    end

    return nothing
end

function _int2part!(
    p::NTuple,
    Fp::AbstractArray{T,N},
    F::AbstractArray,
    F0::AbstractArray,
    grid,
    xci_augmented,
    dxi,
    icell,
    jcell,
    α,
) where {T,N}
    max_xcell = size(Fp, 1)

    for ip in 1:max_xcell
        # interpolate field to current particle
        F_p = _grid2particle_xcell_centered(p, grid, xci_augmented, dxi, F, icell, jcell)
        F0_p = _grid2particle_xcell_centered(p, grid, xci_augmented, dxi, F0, icell, jcell)
        F_flip = Fp[ip, icell, jcell] + F_p - F0_p
        Fp[ip, icell, jcell] = F_p * α + F_flip * (1 - α)
    end
end

############

function int2part_vertex!(
    Fp, F, F0, particles::Particles, grid::NTuple{2,T}; α=1.0
) where {T}
    (; coords, index, max_xcell) = particles
    dxi = compute_dx(grid)
    px, py = coords
    nx, ny = size(F)

    @parallel (1:max_xcell, 1:nx, 1:ny) int2part_vertex!(
        Fp, F, F0, px, py, index, grid, dxi, α
    )

    return nothing
end

@parallel_indices (ipart, i, j) function int2part_vertex!(
    Fp, F, F0, px, py, index, grid, dxi, α
)
    if i < size(F, 1) && j < size(F, 2) && index[ipart, i, j]
        pᵢ = (px[ipart, i, j], py[ipart, i, j])
        _int2part_vertex!(pᵢ, Fp, F, F0, grid, dxi, i, j, α)
    end

    return nothing
end

function _int2part_vertex!(
    p::NTuple,
    Fp::AbstractArray{T,N},
    F::AbstractArray,
    F0::AbstractArray,
    grid,
    dxi,
    icell,
    jcell,
    α,
) where {T,N}
    max_xcell = size(Fp, 1)

    for ip in 1:max_xcell
        # interpolate field to current particle
        F_p = _grid2particle_xvertex(p, grid, dxi, F, icell, jcell)
        F0_p = _grid2particle_xvertex(p, grid, dxi, F0, icell, jcell)
        F_flip = Fp[ip, icell, jcell] + F_p - F0_p
        Fp[ip, icell, jcell] = F_p * α + F_flip * (1 - α)
    end
end

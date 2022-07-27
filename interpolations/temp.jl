function gather_temperature_xvertex!(
    F::AbstractArray{_T,2}, Fp::AbstractArray, ρCₚp::AbstractArray, xi, particle_coords
) where _T
    dxi = (xi[1][2] - xi[1][1], xi[2][2] - xi[2][1])
    nx, ny = size(F)
    Threads.@threads for jnode in 1:ny
        for inode in 1:nx
            _gather_temperature_xvertex!(F, Fp, ρCₚp, inode, jnode, xi, particle_coords, dxi)
        end
    end
end

@inbounds function _gather_temperature_xvertex!(F, Fp, ρCₚp, inode, jnode, xi, p, dxi)
    px, py = p # particle coordinates
    nx, ny = size(F)
    xvertex = (xi[1][inode], xi[2][jnode]) # cell lower-left coordinates
    ω, ωxF = 0.0, 0.0 # init weights
    max_xcell = size(px, 1) # max particles per cell

    # iterate over cells around i-th node
    for joffset in -1:0
        jvertex = joffset + jnode
        for ioffset in -1:0
            ivertex = ioffset + inode
            # make sure we stay within the grid
            if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny)
                # iterate over cell
                @inbounds for i in 1:max_xcell
                    p_i = (px[i, ivertex, jvertex], py[i, ivertex, jvertex])
                    ρCₚp_i = ρCₚp[i, ivertex, jvertex]
                    # ignore lines below for unused allocations
                    isnan(p_i[1]) && continue
                    ω_i = bilinear_weight(xvertex, p_i, dxi) * ρCₚp_i
                    @show ρCₚp_i
                    ω += ω_i
                    ωxF += ω_i * Fp[i, ivertex, jvertex]
                end
            end
        end
    end


    return F[inode, jnode] = ωxF / ω
end

function load_benchmark_data(filename)
    params = matread(filename)
    return params["Vxp"], params["Vyp"]
end

function save_timestep!(fname, p, t)
    matwrite(
        fname, Dict("pX" => Array(p[1]), "pY" => Array(p[2]), "time" => t); compress=true
    )
    return nothing
end

function foo(i)
    if i < 10
        return "0000$i"

    elseif 10 ≤ i ≤ 99
        return "000$i"

    elseif 100 ≤ i ≤ 999
        return "00$i"

    elseif 1000 ≤ i ≤ 9999
        return "0$i"
    end
end

function plot(x, y, T, particles, pT, it)
    pX, pY = Array.(particles.coords)
    pidx = Array(particles.index)
    ii = findall(x -> x == true, pidx)

    T = T[2:(end - 1), 2:(end - 1)]
    cmap = :batlow

    f = Figure(; resolution=(900, 450))
    ax1 = Axis(f[1, 1])
    scatter!(ax1, pX[ii], pY[ii]; color=Array(pT[ii]), colorrange=(0, 1), colormap=cmap)

    ax2 = Axis(f[1, 2])
    hm = heatmap!(ax2, x, y, Array(T); colorrange=(0, 1), colormap=cmap)
    Colorbar(f[1, 3], hm)

    hideydecorations!(ax2)
    linkaxes!(ax1, ax2)
    for ax in (ax1, ax2)
        xlims!(ax, 0, 10)
        ylims!(ax, 0, 10)
    end

    fi = foo(it)
    fname = joinpath("imgs", "fig_$(fi).png")
    save(fname, f)

    return f
end

# function shuffle_particles!(particles, grid, dxi, nxi, args)
#     nx, ny = nxi
#     # unpack
#     (; coords, index, inject, max_xcell, min_xcell) = particles
#     nx, ny = nxi
#     px, py = coords

#     offsets = ( 
#         (1, 0, 0), 
#         (1, 0, 1), 
#         (1, 1, 0), 
#         (2, 0, 0), 
#     )
#     # TODO add offset to indices to avoid race conditions
#     N=2
#     n_i = ceil(Int, nx*(1/N))
#     n_j = ceil(Int, ny*(1/N))

#     # Q = Set{Tuple{Int64,Int64}}()
#     # for j in 1:nx-1, i in 1:ny-1
#     #     push!(Q, (i,j))
#     # end
#     # c = 0
#     # color = 0
#     # I = zeros(Int, nx-1, ny-1)
#     for offset_i in offsets
#         # offset, offset_x, offset_y = offset_i
#         # color+=1
#         for icell in 1:n_i, jcell in 1:n_j
#             # i = offset + 2*(icell-1) + offset_x
#             # j = offset + 2*(jcell-1) + offset_y 
#             # println("($i, $j)")
#             # if i < 40 && j < 40
#             #     c+=1
#             #     I[i,j] = color 
#             # end
#             shuffle_particles!(px, py, grid, dxi, nxi, index, inject, max_xcell, min_xcell, offset_i, icell, jcell, args)
#         end
#     end

#     if (px,py) != particles.coords
#         copyto!(particles.coords[1], px)
#         copyto!(particles.coords[2], py)
#     end
# end

# function shuffle_particles!(px, py, grid, dxi::NTuple{2,T}, nxi, index, inject, max_xcell, min_xcell, offset_i, icell, jcell, args) where T
#     nx, ny = nxi
#     offset, offset_x, offset_y = offset_i
#     icell = offset + 2*(icell-1) + offset_x
#     jcell = offset + 2*(jcell-1) + offset_y 

#     if (icell ≤ nx-1) && (jcell ≤ ny-1)
#         _shuffle_particles!(px, py, grid, dxi, nxi, index, inject, max_xcell, min_xcell, icell, jcell, args)
#     end
#     return nothing
# end

# offsets = ( 
#     (1, 0, 0), 
#     (1, 0, 1), 
#     (1, 1, 0), 
#     (2, 0, 0), 
# )
# # TODO add offset to indices to avoid race conditions
# N=2
# n_i = ceil(Int, nx*(1/N))
# n_j = ceil(Int, ny*(1/N))
# color = 0
# I = zeros(Int, nx-1, ny-1)
# for offset_i in offsets
#     offset, offset_x, offset_y = offset_i
#     color+=1
#     for icell in 1:n_i, jcell in 1:n_j
#         i = offset + 2*(icell-1) + offset_x
#         j = offset + 2*(jcell-1) + offset_y 
#         println("($i, $j)")
#         if i < nx && j < ny
#             I[i,j] = color 
#         end
#     end
# end

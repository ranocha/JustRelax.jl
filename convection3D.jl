ENV["PS_PACKAGE"] = :Threads
using Pkg;
Pkg.activate(".");
using JustRelax
using Printf, LinearAlgebra
using GLMakie
# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 3)
environment!(model)

# using JustPIC
pth = "C:\\Users\\albert\\Desktop\\JustPIC.jl"
using MuladdMacro
using ParallelStencil.FiniteDifferences3D

using StencilInterpolations

include(joinpath(pth, "src/particles.jl"))
include(joinpath(pth, "src/utils.jl"))
include(joinpath(pth, "src/advection.jl"))
include(joinpath(pth, "src/injection.jl"))
include(joinpath(pth, "src/shuffle_vertex.jl"))
include(joinpath(pth, "src/staggered/centered.jl"))
include(joinpath(pth, "src/staggered/velocity.jl"))
include(joinpath(pth, "src/data.jl"))

function it2str(it)
    it < 10 && return "000$it"
    it < 100 && return "00$it"
    it < 1000 && return "0$it"
    it < 10000 && return "$it"
end

function plot(F,n)
    fig = Figure(; resolution=(1600, 1200))
    ax = Axis(fig[1, 1])
    hm = heatmap!(ax, xvi[1], xvi[3], Array(F[:, n,:]); colormap=:batlow)
    Colorbar(fig[1,2], hm)
    fig
end

function cleanse!(particles::Particles, args, xvi::NTuple{N,T}) where {N,T}
    index, coords = particles.index, particles.coords 
    grid_lims = extrema.(xvi)
    px, py, pz = coords
    rng = ntuple(i->1:length(xvi[i])-1,Val(N))
    @parallel rng _cleanse!(px, py, pz, index, args, grid_lims)
end

@parallel_indices (i,j,k) function _cleanse!(px, py, pz, index, args, grid_lims)
    for ipart in axes(px, 1)
        if !(grid_lims[1][1] ≤ px[ipart,i,j,k] ≤  grid_lims[1][2]) || !(grid_lims[2][1] ≤ py[ipart,i,j,k] ≤  grid_lims[2][2]) || !(grid_lims[3][1] ≤ pz[ipart,i,j,k] ≤  grid_lims[3][2])
            index[ipart,i,j,k] = false
            px[ipart,i,j,k] = NaN
            py[ipart,i,j,k] = NaN
            pz[ipart,i,j,k] = NaN
            for n in eachindex(args)
                args[n][ipart,i,j,k] = NaN
            end
        end
    end
    return nothing
end

@parallel_indices (i, j) function init_T!(T, ΔT, w, ix, iy, dx, dy, lx, ly)
    T[i] =
        ΔT *
        exp(-(((ix[i] - 1) * dx - 0.5 * lx) / w)^2 - (((iy[j] - 1) * dy - 0.5 * ly) / w)^2)
    return nothing
end

@parallel function viscosity!(η, T)
    @all(η) = exp(23.03 / (@all(T) + 1.0) - 23.03 * 0.5)
    return nothing
end

function compute_dt(S::StokesArrays, di::NTuple{N,T}, dt_diff) where {N,T}
    return compute_dt(S.V, di, dt_diff)
end


function compute_dt(V::Velocity, di::NTuple{3,T}, dt_diff) where {T}
    return compute_dt(V.Vx, V.Vy, V.Vz, di[1], di[2], di[3], dt_diff)
end

function compute_dt(Vx, Vy, Vz, dx, dy, dz, dt_diff)
    dt_adv = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 8.1
    return min(dt_diff, dt_adv)
end

@parallel function add_dTdt(T, ΔT)
    @inn(T) = @inn(T) + @inn(ΔT)
    return nothing
end

@parallel function update_buoyancy!(fy, T, ρ0gα)
    @all(fy) = ρ0gα .* @all(T)
    return nothing
end

@parallel_indices (i, j, k) function init_linear_T(T, y)
    T[i, j, k] = clamp(-y[k] * (1 + 0.05 * rand()), 0.0, 1.0)
    return nothing
end

@parallel_indices (i, j, k) function init_block_T(T, x, y, z)

    # if (4 ≤ x[i] ≤ 4) && (0.4 ≤ y[j] ≤ 0.6) && (-0.6 ≤ z[k] ≤ -0.4)
    if (4-0.1 ≤ x[i] ≤ 4+0.1) && (0.4 ≤ y[j] ≤ 0.6) && (-0.52 ≤ z[k] ≤ -0.48)
        T[i, j, k] = 0.0
    else
        T[i, j, k] = 1.0
    end

    return nothing
end

function twoxtwo_particles3D(nxcell, max_xcell, min_xcell, x, y, z, dx, dy, dz, nx, ny, nz)
    nx -= 1
    ny -= 1
    nz -= 1
    ncells = nx * ny * nz
    np = max_xcell * ncells
    dx_2 = dx * 0.5
    dy_2 = dy * 0.5
    dz_2 = dz * 0.5
    px, py, pz = ntuple(_ -> fill(NaN, max_xcell, nx, ny, nz), Val(3))
    # min_xcell = ceil(Int, nxcell / 2)
    # min_xcell = 4

    # index = zeros(UInt32, np)
    inject = falses(nx, ny, nz)
    index = falses(max_xcell, nx, ny, nz)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        # vertex of the cell
        x0, y0, z0 = x[i], y[j], z[k]
        # fill index array
        for l in 1:nxcell
            # px[l, i, j, k] = x0 + dx_2 * (1.0 + 0.8 * (rand() - 0.5))
            # py[l, i, j, k] = y0 + dy_2 * (1.0 + 0.8 * (rand() - 0.5))
            # pz[l, i, j, k] = z0 + dz_2 * (1.0 + 0.8 * (rand() - 0.5))
            px[l, i, j, k] = x0 + dx*rand()
            py[l, i, j, k] = y0 + dy*rand()
            pz[l, i, j, k] = z0 + dz*rand()
            index[l, i, j, k] = true
        end
    end

    if ENV["PS_PACKAGE"] === "CUDA"
        pxi = CuArray.((px, py))
        return Particles(
            pxi,
            CuArray(index),
            CuArray(inject),
            nxcell,
            max_xcell,
            min_xcell,
            np,
            (nx, ny, nz),
        )

    else
        return Particles(
            (px, py, pz), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny, nz)
        )
    end
end

function velocity_grids(xvi::NTuple{3,T}, di) where {T}
    dx, dy, dz = di
    xvx = (xvi[1][1] - dx / 2):dx:(xvi[1][end] + dx / 2)
    yvy = (xvi[2][1] - dy / 2):dy:(xvi[2][end] + dy / 2)
    zvz = (xvi[3][1] - dz / 2):dz:(xvi[3][end] + dz / 2)
    grid_vx = (xvx, xvi[2], xvi[3])
    grid_vy = (xvi[1], yvy, xvi[3])
    grid_vz = (xvi[1],  xvi[3], zvz)

    return grid_vx, grid_vy, grid_vz
end

function velocity(s::StokesArrays)
    (;Vx,Vy,Vz) = s.V;
    return √(maximum(Vx)^2+maximum(Vy)^2+maximum(Vz)^2)
end

nx = 64
ny = 8
nz = 32
lx = 3e0
ly = 1e0
lz = 1e0
ar = 8
init_MPI=true
finalize_MPI=false

function thermal_convection3D(; nx=64, ny=64, nz=64, ar=3, ly=1e0, lz=1e0, init_MPI=true, finalize_MPI=false)

    # Physical domain
    ni = (nx, ny, nz)
    lx = ar * ly
    li = (lx, ly, lz) # domain length in x- and y-
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=init_MPI)...) # init MPI
    di = @. li / (nx_g()-1, ny_g()-1, nz_g()-1) # grid step in x- and -y
    max_li = max(li...)
    xci, xvi = lazy_grid(di, li; origin=(0.0, 0.0, -lz)) # nodes at the center and vertices of the cells

    # Physical parameters
    η0 = 1.0                # viscosity, Pa*s
    κ = 1.0                # heat diffusivity, m^2/s
    ΔT = 1.0                # initial temperature perturbation K
    # Physics - nondim numbers
    Ra = 1e7                # Raleigh number = ρ0*g*α*ΔT*ly^3/η0/κ
    Pra = 1e3                # Prandtl number = η0/ρ0/DcT
    g = 1
    # Physics - dimentionally dependent parameters
    w = 5e-2 * ly            # initial perturbation standard deviation, m
    ρ0gα = Ra * η0 * κ / ΔT / ly^3    # thermal expansion
    dη_dT = 1e-10 / ΔT           # viscosity's temperature dependence
    dt_diff = 1.0 / 6.1 * min(di...)^3 / κ      # diffusive CFL timestep limiter
    dt = dt_diff # physical time step

    # Thermal diffusion ----------------------------------
    # allocate and define initial geotherm
    thermal = ThermalArrays(ni)
    # thermal.T .= PTArray([-xvi[2][j]*(1 + 0.01*rand()) for i in axes(thermal.T, 1), j in axes(thermal.T, 2)])
    # clamp!(thermal.T, 0.0, 1.0)
    @parallel (1:nx, 1:ny, 1:nz) init_linear_T(thermal.T, xvi[3])
    # @parallel (1:nx, 1:ny, 1:nz) init_block_T(thermal.T, xvi...)
    @parallel assign!(thermal.Told, thermal.T)

    # physical parameters
    ρ0 = 1 / Pra * η0 / κ
    ρ = @fill(ρ0, ni...)
    Cp = @fill(1.0, ni...)
    ρCp = @. Cp * ρ
    K = ρCp .* κ
    thermal_parameters = ThermalParameters(K, ρCp)
    # PT coefficients
    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li; CFL=0.5 / √3)
    thermal_bc = (flux_x=true, flux_y=true, flux_z=false)
    # ----------------------------------------------------

    # Stokes ---------------------------------------------
    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di; CFL=0.9 / √3, ϵ=1e-4)
    G = Inf
    ## Setup-specific parameters and fields
    η = @zeros(ni...) # viscosity field
    @parallel viscosity!(η, thermal.T)
    fy = (
        @zeros(ni...),
        @zeros(ni...),
        ρ0gα .* thermal.T
    )
    ## Boundary conditions
    freeslip = (freeslip_x=true, freeslip_y=true, freeslip_z=true)
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 8, 10, 6
    particles = twoxtwo_particles3D(
        nxcell, max_xcell, min_xcell, xvi[1], xvi[2], xvi[3], di[1], di[2], di[3], nx, ny, nz
    )
    # velocity grids
    grid_vx, grid_vy, grid_vz = velocity_grids(xvi, di)
    # temperature
    pT = similar(particles.coords[1])
    particle_args = (pT,)
    # ρCₚp = similar(pT)
    # grid2particle_xvertex!(ρCₚp, xvi, ρCp, particles.coords)

    # gathering_xvertex!(thermal.T, pT, xvi, particles.coords)
    # particle_args = (pT, ρCₚp)
    # ----------------------------------------------------

    # # Initialize animation -------------------------------
    # fig = Figure(; resolution=(1600, 1200))
    # ax = Axis(fig[1, 1])
    # hm = heatmap!(ax, xvi[1], xvi[3], Array(thermal.T[:, ny÷2,:]); colormap=:batlow)
    # # hm = heatmap!(ax, xvi[1], xvi[3], Array(stokes.V.Vz[:, ny÷2,:]); colormap=:inferno)
    # Colorbar(fig[1,2], hm)
    # px0, py0, pz0 = deepcopy(particles.coords)

    # Physical time loop
    it = 0
    nt = 10
    local t = 0.0
    local iters
    # while it ≤ 100
    for it in 1:nt
        # Stokes solver ---------------
        iters = solve!(
            stokes, 
            pt_stokes, 
            ni,
            di, 
            li, 
            max_li, 
            freeslip, 
            fy,
            η,
            G,
            dt,
            igg;
            iterMax=10e3,
            b_width=(4, 4, 4),
        )
        # ------------------------------

        @show dt = compute_dt(stokes, di, dt_diff)
        @show velocity(stokes)

        # Thermal solver ---------------
        # grid2particle_xvertex!(ρCₚp, xvi, ρCp,  particles.coords)
        # _gather_temperature_xvertex!(thermal.T, ρCₚp, pT, xvi,  particles.coords)

        pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li; ϵ=1e-4, CFL=5e-2 / √3)
        iters = solve!(
            thermal,
            pt_thermal,
            thermal_parameters,
            thermal_bc,
            ni,
            di,
            igg,
            dt;
            iterMax=100e3,
            nout=10,
            verbose=false,
        )
        # ------------------------------

        # Advection --------------------
        # interpolate fields from grid vertices to particles
        grid2particle_xvertex!(pT, xvi, thermal.T, particles.coords)
        # int2part_vertex!(pT, thermal.T, thermal.Told, particles, xvi)
        # advect particles in space
        V = (stokes.V.Vx, stokes.V.Vy, stokes.V.Vy)
        advection_RK2_edges!(particles, V, grid_vx, grid_vy, grid_vz, dt, 0.5)

        # ii = findall(particles.index);
        # a = (px0, py0, pz0) .- particles.coords
        # max_displ = maximum(@. √(( a[1][ii]^2 + a[2][ii]^2 + a[3][ii]^2)))
        # @show max_displ
        # # any(di.< max_displ) && break

        # advect particles in memory
        @edit shuffle_particles_vertex!(particles, xvi, particle_args)
        cleanse!(particles, particle_args, xvi)
        foo(particles, xvi)

        # check if we need to inject particles
        @show inject = check_injection(particles)
        inject && inject_particles!(particles, particle_args, (thermal.T,), xvi)
        # interpolate fields from particle to grid vertices
        gathering_xvertex!(thermal.T, pT, xvi, particles.coords)
        # gather_temperature_xvertex!(thermal.T, pT, ρCₚp, xvi, particles.coords)

        # update Temperature field
        # @parallel add_dTdt(thermal.T, thermal.ΔT)
        # thermal.T[:, :, 1] .= 1.0
        # thermal.T[:, :, end] .= 0.0
        # ------------------------------
        # Update viscosity
        @parallel viscosity!(η, thermal.T)

        # Update buoyancy
        @parallel update_buoyancy!(fy[3], thermal.T, ρ0gα)

        # @show it += 1
        # t += dt

        # copyto!(px0, particles.coords[1])
        # copyto!(py0, particles.coords[2])
        # copyto!(pz0, particles.coords[3])

        if it % 1 == 0
            # hm[3] = Array(thermal.T[:, ny÷2,:]);
            # hm[3] = Array(stokes.V.Vz[:, ny÷2,:]);
            ii=findall(particles.index);
            Px,_,Pz = particles.coords;
            pxx = Px[ii];
            pzz = Pz[ii];
            # f,ax,h=heatmap(xvi[1], xvi[3], stokes.V.Vz[:,ny÷2,:])
            f,ax,h=heatmap(xvi[1], xvi[3], thermal.T[:,ny÷2,:])
            scatter!(pxx,pzz,color=:red, markersize=10)
            # save("figs_3d/fig_$(it2str(it)).png", fig)
            save("figs_3d/fig_$(it2str(it)).png", f)
        end
        # f
    end
        
    finalize_global_grid(; finalize_MPI=finalize_MPI)

    # return (ni=ni, xci=xci, li=li, di=di), thermal, iters
    # return fig
end

nx = 64
ny = 8
nz = 32
lx = 3e0
ly = 1e0
ly = 1e0
ar = 8

# # @time fig=thermal_convection2D(; nx=nx, ny=ny, ar=ar, ly=ly)

# ii=findall(particles.index);
# pxx,pyy,pzz = particles.coords;
# pxx = pxx[ii];
# pyy = pyy[ii];
# pzz = pzz[ii];
# f,ax,h=heatmap(xvi[1], xvi[3], thermal.T[:,ny÷2,:])
# scatter!(pxx,pzz,color=pT[ii], markersize=25)
# scatter!(pxx,pzz,color=:red, markersize=10)
# scatter!(x,z, markersize=10)
# scatter(pxx,pyy,pzz,color=pT[ii], markersize=50)

# x = [x for x in xvi[1], y in xvi[2], z in xvi[3]][:];
# y = [y for x in xvi[1], y in xvi[2], z in xvi[3]][:];
# z = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:];

# scatter(pxx,pzz, markersize=10)

# # plot(thermal.T, ny)


# import StencilInterpolations: grid_size, bilinear_weight

# # foo!(thermal.T, pT, xvi, particles.coords)

# function foo!(
#     F::AbstractArray, Fp::AbstractArray, xi::NTuple{3, T}, particle_coords
# ) where {T}
#     dxi = grid_size(xi)
#     Threads.@threads for knode in axes(F,3)
#         for jnode in axes(F,2), inode in axes(F,1)
#             _foo!(F, Fp, inode, jnode, knode, xi, particle_coords, dxi)
#         end
#     end
# end

# @inbounds function _foo!(F, Fp, inode, jnode, knode, xi::NTuple{3, T}, p, dxi) where T
#     px, py, pz = p # particle coordinates
#     nx, ny, nz = size(F)
#     xvertex = (xi[1][inode], xi[2][jnode],  xi[3][knode]) # cell lower-left coordinates
#     ω, ωxF = 0.0, 0.0 # init weights
#     max_xcell = size(px, 1) # max particles per cell

#     # iterate over cells around i-th node
#     for koffset in -1:0
#         kvertex = koffset + knode
#         for joffset in -1:0
#             jvertex = joffset + jnode
#             for ioffset in -1:0
#                 ivertex = ioffset + inode
#                 # make sure we stay within the grid
#                 if (1 ≤ ivertex < nx) && (1 ≤ jvertex < ny) && (1 ≤ kvertex < nz)
#                     # iterate over cell
#                     @inbounds for i in 1:max_xcell
#                         p_i = (
#                             px[i, ivertex, jvertex, kvertex],
#                             py[i, ivertex, jvertex, kvertex], 
#                             pz[i, ivertex, jvertex, kvertex]
#                         )
#                         # ignore lines below for unused allocations
#                         isnan(p_i[1]) && continue
#                         ω_i = bilinear_weight(xvertex, p_i, dxi)
#                         ω += ω_i
#                         ωxF += ω_i * Fp[i, ivertex, jvertex, kvertex]
#                         @show ω_i,  ivertex, jvertex, kvertex
#                     end
#                 end
#             end
#         end
#     end

#     return F[inode, jnode, knode] = ωxF / ω
# end


function foo(particles, xvi)   
    pxx,pyy,pzz=particles.coords;
    n = 0
    for k in axes(pxx,4), j in axes(pxx,3), i in axes(pxx,2),ip in axes(pxx,1)
        if (i < nx) && (j < ny) && (k < nz) && particles.index[ip,i,j,k]
            p_i = (
                pxx[ip,i,j,k],
                pyy[ip,i,j,k],
                pzz[ip,i,j,k],
            )

            xv = (
                xvi[1][i],
                xvi[2][j],
                xvi[3][k],
            )

            if !isincell(p_i, xv, di) 
                n+=1 
                println("ATENCIO......")
                @show ip,i,j,k
                println(".............")
                return ip,i,j,k
            end
        end

    end
end




ii=findall(particles.index);
pxx,pyy,pzz = particles.coords;
p_x = pxx[Ip...];
p_y = pyy[Ip...];
p_z = pzz[Ip...];
# f,ax,h=heatmap(xvi[1], xvi[3], thermal.T[:,ny÷2,:])
# scatter!(pxx,pzz,color=pT[ii], markersize=25)
# scatter!(pxx,pzz,color=:red, markersize=10)
# scatter!(x,z, markersize=10)
# scatter(pxx,pyy,pzz,color=pT[ii], markersize=50)

x = [x for x in xvi[1], y in xvi[2], z in xvi[3]][:];
y = [y for x in xvi[1], y in xvi[2], z in xvi[3]][:];
z = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:];

scatter(x,z, markersize=10)
scatter!((p_x,p_z), markersize=10, color=:red)
scatter!((xvi[1][I[1]], xvi[3][I[3]]), markersize=10, color=:orange)

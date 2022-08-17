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
include("rheology/Viscosity.jl")

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

function check_lost(particles, xvi)
    pxx,pyy,pzz=particles.coords;
    n = 0
    for k in axes(pxx,4), j in axes(pxx,3), i in axes(pxx,2), ip in axes(pxx,1)
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
            end
        end

    end
    n
end

function find_lost(particles, xvi, nlost)

    I = Vector{NTuple{4,Int64}}(undef, nlost)
    pxx,pyy,pzz=particles.coords;
    n = 0
    for k in axes(pxx,4), j in axes(pxx,3), i in axes(pxx,2), ip in axes(pxx,1)
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
                I[n] = (ip,i,j,k)
            end
        end

    end
    I
end

function compute_dt(S::StokesArrays, di::NTuple{N,T}) where {N,T}
    return compute_dt(S.V, di)
end

function compute_dt(V::Velocity, di::NTuple{3,T}) where {T}
    return compute_dt(V.Vx, V.Vy, V.Vz, di[1], di[2], di[3])
end

function compute_dt(Vx, Vy, Vz, dx, dy, dz)
    return min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy)), dz / maximum(abs.(Vz))) / 6.1
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

@inline function incube(x, y, z, depth, width) 
    low_lim = depth-width
    up_lim = depth+width
    (low_lim ≤ x ≤ up_lim) && (low_lim ≤ y ≤ up_lim) && (low_lim ≤ z ≤ up_lim) && return true
    return false
end

@parallel_indices (i, j, k) function getphase!(phase, x, y, z, block_geometry)
    if incube(x[i], y[j], z[k], block_geometry.z, block_geometry.w*0.5)
        phase[i, j, k] = 2 # inclusion phase
    end
    return nothing
end

@parallel_indices (i, j, k) function property2grid!(F, phase, Fi)
    F[i, j, k] = Fi[phase[i, j, k]]
    return nothing
end

function property2grid(phase, Fi)
    ni = size(phase)
    F = @zeros(ni...)

    @parallel (1:ni[1], 1:ni[2], 1:ni[3]) property2grid!(F, phase, Fi)

    return F
end

@parallel_indices (i, j, k) function _geoparams2grid!(F, phase, args)
    F.val[i, j, k] = compute_viscosity(F, phase[i, j, k], tupleindex(args, i, j, k))
    return nothing
end

function geoparams2grid!(F, phase, args)
    ni = size(phase)
    @parallel (1:ni[1], 1:ni[2], 1:ni[3]) _geoparams2grid!(F, phase, args)
    return F
end

@parallel function compute_density!(ρ, T, ρ0, α)
    @all(ρ) = ρ0*(1.0 - α*@all(T))
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
            px[l, i, j, k] = x0 + dx_2 * (1.0 + 0.8 * (rand() - 0.5))
            py[l, i, j, k] = y0 + dy_2 * (1.0 + 0.8 * (rand() - 0.5))
            pz[l, i, j, k] = z0 + dz_2 * (1.0 + 0.8 * (rand() - 0.5))
            # px[l, i, j, k] = x0 + dx*rand()
            # py[l, i, j, k] = y0 + dy*rand()
            # pz[l, i, j, k] = z0 + dz*rand()
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

PTtype(::Type{Array{T,N}}) where {T,N} = Array
if isdefined(Main, :CUDA)
    PTtype(::Type{CuArray{T,N}}) where {T,N} = CuArray
end

nx = 32
ny = 32
nz = 32
lx = 1e0
ly = 1e0
lz = 1e0
ar = 8
init_MPI=false
finalize_MPI=false

function thermal_convection3D(; nx=64, ny=64, nz=64, ar=3, ly=1e0, lz=1e0, init_MPI=true, finalize_MPI=false)

    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny, nz) # number of nodes in x- and y-
    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=init_MPI)...) # init MPI
    li = (lx, ly, lz)  # domain length in x- and y-
    di = @. li / (nx_g()-1, ny_g()-1, nz_g()-1) # grid step in x- and -y
    max_li = max(li...)
    xci, xvi = lazy_grid(di, li) # nodes at the center and vertices of the cells
    depth = [maximum(xci[3]).-z for x in xvi[1], y in xvi[2], z in xvi[3]]

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    dt = 1   # physical time step

    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, ViscoElastic)
    stokes.P .= PTtype(PTArray)(ones(eltype(stokes.P)))
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di; Re=6π, CFL=  0.5 / √3)

    ## Setup-specific parameters and fields
    G = 1.0 # elastic shear modulus
    dt = typemax(Float64)
    
    # phase array (every material phase is associated to an integer)
    block_geometry =(z=0.5, w=0.3)
    phasev = PTtype(PTArray)(ones(Int64, ni...)) # 1 = matrix phase
    @parallel (1:ni[1], 1:ni[2], 1:ni[3]) getphase!(
        phasev, xvi[1], xvi[2], xvi[3], block_geometry
    )

    local_max_li_phase = (max_li, block_geometry.w)
    # local max length on the grid
    local_max_li = property2grid(phasec, local_max_li_phase)

    ## Viscosity
    # custom struct
    visc_params_background = StagCreep(1e0, 1e0, 0e0)
    visc_params_sinker = StagCreep(1e0, log(1e1), 0e0)
    η = Viscosity(
        @allocate(ni...),
        (visc_params_background, visc_params_sinker),
        (Elasticity(1e0), Elasticity(1e0)),
    )

    # arguments for rheological laws
    T = @zeros(ni...)
    T[phasev.==1] .= 1.0
    args = (
        (T, depth), # inputs for LinearViscous
        dt, # inputs for Elasticity
    )

    # compute effective viscosity on the grid
    geoparams2grid!(η, phasev, args)

    ## Density
    ρ0, α = 1.0, 1.0
    ρ = @zeros(ni...)
    @parallel compute_density!(ρ, T, ρ0, α)

    ## Boundary conditions
    freeslip = (freeslip_x=true, freeslip_y=true, freeslip_z=true)

    ## Body forces
    g = (0, 0, -1)
    fy = ntuple(i -> @fill(g[i], ni...) .* ρ, Val(3))

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 18, 18*2, 12
    particles = twoxtwo_particles3D(
        nxcell, max_xcell, min_xcell, xvi[1], xvi[2], xvi[3], di[1], di[2], di[3], nx, ny, nz
    )
    # velocity grids
    grid_vx, grid_vy, grid_vz = velocity_grids(xvi, di)
    # temperature
    pT = similar(particles.coords[1])
    pPhase = similar(pT)
    particle_args = (pT,pPhase)

    ## Time loop
    t = 0.0
    # println_mpi("Starting solver")
    # I = (1, (@. ni÷2)...)
    # pI = [getindex.(particles.coords,I...)]
    # Physical time loop
    it = 0
    nt = 25
    local t = 0.0
    local iters
    # while it ≤ 100

    @show check_lost(particles, xvi)
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
            η.val,
            G,
            dt,
            igg;
            iterMax=10e3,
            b_width=(4, 4, 4),
        )
        # ------------------------------

        @show dt = compute_dt(stokes, di)
        # @show velocity(stokes)

        # Thermal solver ---------------
        # pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li; ϵ=1e-4, CFL=5e-2 / √3)
        # iters = solve!(
        #     thermal,
        #     pt_thermal,
        #     thermal_parameters,
        #     thermal_bc,
        #     ni,
        #     di,
        #     igg,
        #     dt;
        #     iterMax=100e3,
        #     nout=10,
        #     verbose=false,
        # )
        # ------------------------------

        # Advection --------------------
        # interpolate fields from grid vertices to particles
        for (pargsi, argsi) in zip(particle_args, (T, phasev))
            grid2particle_xvertex!(pargsi, xvi, argsi, particles.coords)
        end

        any(extrema(pT[findall(particles.index)]) .< 0) && break

        # grid2particle_xvertex!(pT, xvi, T, particles.coords)
        # grid2particle_xvertex!(pPhase, xvi, phasev, particles.coords)
        @. pPhase = round(pPhase)
        # advect particles in space
        V = (stokes.V.Vx, stokes.V.Vy, stokes.V.Vy)
        advection_RK2_edges!(particles, V, grid_vx, grid_vy, grid_vz, dt, 0.5)
    
        # advect particles in memory
        cleanse!(particles, particle_args, xvi)
        shuffle_particles_vertex!(particles, xvi, particle_args)
        
        println("--------------------------------") 
        nlost = (check_lost(particles, xvi))
        println("$nlost lost particles") 
        nlost > 0 && break
        println("--------------------------------") 
    
        # check if we need to inject particles
        @show inject = check_injection(particles)
        inject && inject_particles!(particles, particle_args, (T, phasev), xvi)
        # interpolate fields from particle to grid vertices
        gathering_xvertex!(T, pT, xvi, particles.coords)
        any(extrema(T) .< 0) && break

        clamp!(T, 0.0, 1.0)

        # thermal.T[:,:,1] .= 1.0
        # thermal.T[:,:,end] .= 0.0
        # ------------------------------

        # Update viscosity
        args[1][1].=T
        geoparams2grid!(η, phasev, args)

        # # Update buoyancy
        @parallel compute_density!(ρ, T, ρ0, α)
        fy[3] .= @fill(g[3], ni...) .* ρ

        # @show it += 1
        # t += dt

        # if it % 50 == 0
        #     # hm[3] = Array(thermal.T[:, ny÷2,:]);
        #     # hm[3] = Array(stokes.V.Vz[:, ny÷2,:]);
            ii=findall(particles.index);
            Px,Py,Pz = particles.coords;
            pxx = Px[ii];
            pyy = Py[ii];
            pzz = Pz[ii];
        #     # f,ax,h=heatmap(xci[1], xvi[3], stokes.V.Vz[:,ny÷2,:])
            # f,ax,h=heatmap(xvi[1], xvi[3], T[:,ny÷2,:])
            # f,ax,h=scatter(pxx,pzz,color=pT[ii], markersize=10, colormap=:batlow)
            f,ax,h=scatter(pxx,pyy,pzz,color=pT[ii], markersize=10, colormap=:batlow)
        #     # save("figs_3d/fig_$(it2str(it)).png", fig)
            save("figs_3d/fig_$(it2str(it)).png", f)
        # end
        # f

    end
        
    finalize_global_grid(; finalize_MPI=finalize_MPI)

    # return (ni=ni, xci=xci, li=li, di=di), thermal, iters
    # return fig
end

nx = 32
ny = 32
nz = 32
lx = 1e0
ly = 1e0
ly = 1e0
ar = 8
init_MPI=false
finalize_MPI=false

# Save julia setup 
# vtk_grid("Sinker", xvi[1], xvi[2], xvi[3]) do vtk
#     vtk["T"] = thermal.T
#     vtk["Vx"] = stokes.V.Vx[1:ni[1], 1:ni[2], 1:ni[3]]
#     vtk["Vy"] = stokes.V.Vy[1:ni[1], 1:ni[2], 1:ni[3]]
#     vtk["Vz"] = stokes.V.Vz[1:ni[1], 1:ni[2], 1:ni[3]]
# end

thermal_convection3D(; nx=nx, ny=ny, nz=nz, ar=3, ly=ly, lz=lz, init_MPI=init_MPI, finalize_MPI=finalize_MPI)

ii,jj,kk = 17,11,9
I = 17,11,9
for i in 0:1, j in 0:1, k in 0:1
    I,J,K = i+ii,j+jj,k+kk
    println("$I, $J, $K $(T[I,J,K])")
end


@inline function field_corners(F::AbstractArray{T,3}, idx::NTuple{3,Integer}) where {T}
    idx_x, idx_y, idx_z = idx
    return (
        F[idx_x,     idx_y,     idx_z],   # v000
        F[idx_x + 1, idx_y,     idx_z],   # v100
        F[idx_x,     idx_y,     idx_z + 1], # v001
        F[idx_x + 1, idx_y,     idx_z + 1], # v101
        F[idx_x,     idx_y + 1, idx_z],   # v010
        F[idx_x + 1, idx_y + 1, idx_z],   # v110
        F[idx_x,     idx_y + 1, idx_z + 1], # v011
        F[idx_x + 1, idx_y + 1, idx_z + 1], # v111
    )
end

import StencilInterpolations: particle2tuple,field_corners,normalize_coordinates,ndlinear

function foo!(Fp::Array, xvi, F::Array{T,3}, particle_coords) where {T}
    # cell dimensions
    dxi = grid_size(xvi)
    nx, ny, nz = length.(xvi)
    max_xcell = size(particle_coords[1], 1)
    Threads.@threads for knode in 1:(nz - 1)
        for jnode in 1:(ny - 1), inode in 1:(nx - 1)
            _foo!(
                Fp, particle_coords, xvi, dxi, F, max_xcell, (inode, jnode, knode)
            )
        end
    end
end

function _foo!(
    Fp::Array, p::NTuple, xvi::NTuple, dxi::NTuple, F::Array, max_xcell, idx 
)

    @inline function particle2tuple(ip::Integer, idx::NTuple{N,T}) where {N,T}
        return ntuple(i -> p[i][ip, idx...], Val(N))
    end

    # F at the cell corners
    Fi = field_corners(F, idx)

    for i in 1:max_xcell
        # check that the particle is inside the grid
        # isinside(p, xi)

        p_i = particle2tuple(i, idx)

        p_i = ntuple(ix -> p[ix][i, idx...], Val(N))

        any(isnan, p_i) && continue

        # normalize particle coordinates
        ti = normalize_coordinates(p_i, xvi, dxi, idx)

        # Interpolate field F onto particle
        # Fp[i, idx...] = ndlinear(ti, Fi)
        @show p_i i ndlinear(ti, Fi)
    end
end

foo!(pT, xvi, T, particles.coords)

ntuple(i -> xvi[i][idx[i]], Val(3))

pxx,pyy,pzz=particles.coords
p=particles.coords
(
                pxx[ip,i,j,k],
                pyy[ip,i,j,k],
                pzz[ip,i,j,k],
            )

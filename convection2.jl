ENV["PS_PACKAGE"] = :CUDA
using Pkg;
Pkg.activate(".");
using JustRelax
using Printf, LinearAlgebra
using GLMakie
# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
environment!(model)

# include("/home/albert/Desktop/JustPIC.jl/src/JustPIC.jl")
using JustPIC

function it2str(it)
    it < 10 && return "000$it"
    it < 100 && return "00$it"
    it < 1000 && return "0$it"
    it < 10000 && return "$it"
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

function compute_dt(S::StokesArrays, di::NTuple{2,T}, dt_diff) where {T}
    return compute_dt(S.V, di, dt_diff)
end
function compute_dt(V::Velocity, di::NTuple{2,T}, dt_diff) where {T}
    return compute_dt(V.Vx, V.Vy, di[1], di[2], dt_diff)
end

function compute_dt(Vx, Vy, dx, dy, dt_diff)
    dt_adv = min(dx / maximum(abs.(Vx)), dy / maximum(abs.(Vy))) / 2.1
    return min(dt_diff, dt_adv)
end

@parallel function add_dTdt(T, ΔT)
    @inn(T) = @inn(T) + @inn(ΔT)
    return nothing
end

@parallel function update_buoyancy!(fy, T, ρ0gα)
    @all(fy) = -ρ0gα .* @all(T)
    return nothing
end

@parallel_indices (i,j) function init_linear_T(T, y)
    T[i,j] = clamp(-y[j]*(1 + 0.05*rand()), 0.0, 1.0)
    return 
end


@parallel (1:nx,1:ny) init_linear_T(thermal.T, xvi[2])

function twoxtwo_particles2D(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
    nx -= 1
    ny -= 1
    ncells = nx * ny
    np = max_xcell * ncells
    dx_2 = dx * 0.5
    dy_2 = dy * 0.5
    px, py = ntuple(_ -> fill(NaN, max_xcell, nx, ny), Val(2))
    # min_xcell = ceil(Int, nxcell / 2)
    # min_xcell = 4

    # index = zeros(UInt32, np)
    inject = falses(nx, ny)
    index = falses(max_xcell, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        # center of the cell
        x0, y0 = x[i], y[j]
        # fill index array
        for l in 1:nxcell
            px[l, i, j] = x0 + dx_2 * (1.0 + 0.8 * (rand() - 0.5))
            py[l, i, j] = y0 + dy_2 * (1.0 + 0.8 * (rand() - 0.5))
            index[l, i, j] = true
        end
    end

    if PS_PACKAGE === :CUDA
        pxi = CuArray.((px, py))
        return Particles(
            pxi, CuArray(index), CuArray(inject), nxcell, max_xcell, min_xcell, np, (nx, ny)
        )

    else
        return Particles((px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny))
    end
end

function velocity_grids(xvi::NTuple{2,T}, di) where {T}
    dx, dy = di
    # yvx = xvi[2][1]-dy/2:dy:xvi[2][end]+dy/2
    # xvy = xvi[1][1]-dx/2:dx:xvi[1][end]+dx/2
    # grid_vx = (xvi[1], yvx)
    # grid_vy = (xvy, xvi[2])

    xvx = (xvi[1][1] - dx / 2):dx:(xvi[1][end] + dx / 2)
    yvy = (xvi[2][1] - dy / 2):dy:(xvi[2][end] + dy / 2)
    grid_vx = (xvx, xvi[2])
    grid_vy = (xvi[1], yvy)

    return grid_vx, grid_vy
end

nx = 287
ny = 95
lx = 3e0
ly = 1e0
ar = 8

function thermal_convection2D(; nx=64, ny=64, ar=3, ly=1e0)

    # Physical domain
    ni = (nx, ny)
    lx = ar * ly
    li = (lx, ly) # domain length in x- and y-
    di = @. li / (ni - 1) # grid step in x- and -y
    max_li = max(li...)
    xci, xvi = lazy_grid(di, li; origin=(0.0, -ly)) # nodes at the center and vertices of the cells

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
    dt_diff = 1.0 / 4.1 * min(di...)^2 / κ      # diffusive CFL timestep limiter
    dt = dt_diff # physical time step

    # Thermal diffusion ----------------------------------
    # allocate and define initial geotherm
    thermal = ThermalArrays(ni)
    thermal.T .= [-xvi[2][j]*(1 + 0.01*rand()) for i in axes(thermal.T, 1), j in axes(thermal.T, 2)]
    clamp!(thermal.T, 0.0, 1.0)

    @parallel (1:nx,1:ny) init_linear_T(thermal.T, xvi[2])


    @parallel assign!(thermal.Told, thermal.T)
    # physical parameters
    ρ0 = 1 / Pra * η0 / κ
    ρ = @fill(ρ0, ni...)
    Cp = @fill(1.0, ni...)
    ρCp = @. Cp * ρ
    K = ρCp .* κ
    thermal_parameters = ThermalParameters(K, ρCp)
    # PT coefficients
    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li; CFL=0.5 / √2)
    thermal_bc = (flux_x=true, flux_y=false)
    # ----------------------------------------------------

    # Stokes ---------------------------------------------
    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, Viscous)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di; ϵ=1e-4)
    ## Setup-specific parameters and fields
    η = @zeros(ni...) # viscosity field
    @parallel viscosity!(η, thermal.T)
    fy = -ρ0gα .* thermal.T
    ## Boundary conditions
    freeslip = (freeslip_x=true, freeslip_y=true)
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 8, 10, 6
    particles = twoxtwo_particles2D(
        nxcell, max_xcell, min_xcell, xvi[1], xvi[2], di[1], di[2], nx, ny
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xvi, di)
    # temperature
    pT = similar(particles.coords[1])
    ρCₚp = similar(pT)
    grid2particle_xvertex!(ρCₚp, xvi, ρCp, particles.coords)

    # gathering_xvertex!(thermal.T, pT, xvi, particles.coords)
    particle_args = (pT, ρCₚp)
    # ----------------------------------------------------

    # Initialize animation -------------------------------
    fig = Figure(; resolution=(1600, 1200))
    ax = Axis(fig[1, 1])
    hm = heatmap!(ax, xvi[1], xvi[2], thermal.T; colormap=:inferno)

    # Physical time loop
    local t = 0.0
    it = 0
    nt = 3000
    local iters
    # while it ≤ 100
    for it in 1:nt
        # Stokes solver ---------------
        iters = solve!(stokes, pt_stokes, di, li, max_li, freeslip, fy, η; iterMax=10e3)

        @show dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Thermal solver ---------------
        # grid2particle_xvertex!(ρCₚp, xvi, ρCp,  particles.coords)
        # _gather_temperature_xvertex!(thermal.T, ρCₚp, pT, xvi,  particles.coords)

        pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li; ϵ=1e-4, CFL=5e-2 / √2)
        iters = solve!(
            thermal,
            pt_thermal,
            thermal_parameters,
            thermal_bc,
            ni,
            di,
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
        V = (stokes.V.Vx, stokes.V.Vy)
        advection_RK2_edges!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles_vertex!(particles, xvi, particle_args)
        # check if we need to inject particles
        @show inject = check_injection(particles)
        inject && inject_particles!(particles, particle_args, (thermal.T,), xvi)
        # interpolate fields from particle to grid vertices
        gathering_xvertex!(thermal.T, pT, xvi, particles.coords)
        # gather_temperature_xvertex!(thermal.T, pT, ρCₚp, xvi, particles.coords)

        # update Temperature field
        # @parallel add_dTdt(thermal.T, thermal.ΔT)
        thermal.T[:, 1] .= 1.0
        thermal.T[:, end] .= 0.0
        # ------------------------------
        # Update viscosity
        @parallel viscosity!(η, thermal.T)

        # Update buoyancy
        @parallel update_buoyancy!(fy, thermal.T, ρ0gα)

        # @show it += 1
        # t += dt

        if it % 10 == 0
            hm[3] = thermal.T
            save("figs2/fig_$(it2str(it)).png", fig)
        end

    end

    return (ni=ni, xci=xci, li=li, di=di), thermal, iters
end


nx = 287
ny = 95
lx = 3e0
ly = 1e0
ar = 8

# thermal_convection2D(; nx=nx, ny=ny, ar=ar, ly=ly)
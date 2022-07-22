ENV["PS_PACKAGE"] = :Threads

using JustRelax 
using Printf, LinearAlgebra

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model) 

# include("/home/albert/Desktop/JustPIC.jl/src/JustPIC.jl")
using JustPIC

@parallel_indices (i, j) function init_T!(T, ΔT, w, ix, iy, dx, dy, lx, ly)
    T[i] = ΔT*exp(-(((ix[i]-1)*dx-0.5*lx)/w)^2 -(((iy[j]-1)*dy-0.5*ly)/w)^2) 
    return nothing
end

@parallel function viscosity!(η, T, ΔT, dη_dT)
    @all(η) = η0*(1.0 - dη_dT*(@all(T) + ΔT*0.5))
    return
end

compute_dt(S::StokesArrays, di::NTuple{2,T}, dt_diff) where T = compute_dt(S.V, di, dt_diff)
compute_dt(V::Velocity, di::NTuple{2,T}, dt_diff) where T = compute_dt(V.Vx, V.Vy, di[1], di[2], dt_diff)

function compute_dt(Vx, Vy, dx, dy, dt_diff)
    dt_adv = min(dx/maximum(abs.(Vx)), dy/maximum(abs.(Vy))) / 2.1
    return  min(dt_diff, dt_adv)
end

@parallel function add_dTdt(T, ΔT)
    @inn(T) = @inn(T) + @inn(ΔT)
    return nothing
end

function twoxtwo_particles2D(nxcell, max_xcell, x, y, dx, dy, nx, ny)
    ncells = nx * ny
    np = max_xcell * ncells
    dx_2 = dx * 0.5
    dy_2 = dy * 0.5
    px, py = ntuple(_ -> fill(NaN, max_xcell, nx, ny), Val(2))
    min_xcell = ceil(Int, nxcell / 2)
    min_xcell = 4

    # index = zeros(UInt32, np)
    inject = falses(nx, ny)
    index = falses(max_xcell, nx, ny)
    idx = 1
    @inbounds for j in 1:ny, i in 1:nx
        # center of the cell
        x0, y0 = x[i], y[j]
        # index of first particle in cell
        # add 4 new particles in a 2x2 manner + some small random perturbation
        px[idx, i, j] = x0 - 0.25 * dx_2 # * (1.0 + 0.15*(rand() - 0.5))
        px[idx + 1, i, j] = x0 + 0.25 * dx_2 # * (1.0 + 0.15*(rand() - 0.5))
        px[idx + 2, i, j] = x0 - 0.25 * dx_2 # * (1.0 + 0.15*(rand() - 0.5))
        px[idx + 3, i, j] = x0 + 0.25 * dx_2 # * (1.0 + 0.15*(rand() - 0.5))
        py[idx, i, j] = y0 - 0.25 * dy_2 # * (1.0 + 0.15*(rand() - 0.5))
        py[idx + 1, i, j] = y0 - 0.25 * dy_2 # * (1.0 + 0.15*(rand() - 0.5))
        py[idx + 2, i, j] = y0 + 0.25 * dy_2 # * (1.0 + 0.15*(rand() - 0.5))
        py[idx + 3, i, j] = y0 + 0.25 * dy_2 # * (1.0 + 0.15*(rand() - 0.5))
        # fill index array
        for l in 1:nxcell
            # px[l, i, j] = x0 + dx/3*(1.0 + (rand() - 0.5))
            # py[l, i, j] = y0 + dy/3*(1.0 + (rand() - 0.5))
            index[l, i, j] = true
        end
    end

    # if PS_PACKAGE === :CUDA
    #     pxi = CuArray.((px, py))
    #     return Particles(
    #         pxi, CuArray(index), CuArray(inject), nxcell, max_xcell, min_xcell, np, (nx, ny)
    #     )

    # else
    return Particles(
            (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
        )
    # end
end

function velocity_grids(xvi::NTuple{2, T}, di) where {T}
    dx, dy = di
    yvx = xvi[2][1]-dy/2:dy:xvi[2][end]+dy/2
    xvy = xvi[1][1]-dx/2:dx:xvi[1][end]+dx/2
    grid_vx = (xvi[1], yvx)
    grid_vy = (xvy, xvi[2])

    return grid_vx, grid_vy
end
        
nx=287
ny=95
lx=3e0
ly=1e0

function thermal_convection2D(; nx=64, ny=64, lx=3e0, ly=1e0)
   
    # Physical domain
    ni = (nx, ny)
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / (ni-1) # grid step in x- and -y
    max_li = max(li...)
    xci, xvi= lazy_grid(di, li; origin=(-lx/2, -ly/2)) # nodes at the center and vertices of the cells

    # Physical parameters
    η0        = 1.0                # viscosity, Pa*s
    κ         = 1.0                # heat diffusivity, m^2/s
    ΔT        = 1.0                # initial temperature perturbation K
    # Physics - nondim numbers
    Ra        = 1e7                # Raleigh number = ρ0*g*α*ΔT*ly^3/η0/κ
    Pra       = 1e3                # Prandtl number = η0/ρ0/DcT
    ar        = 3                  # aspect ratio
    g         = 1
    # Physics - dimentionally dependent parameters
    lx        = ar*ly              # domain extend, m
    w         = 1e-2*ly            # initial perturbation standard deviation, m
    ρ0gα      = Ra*η0*κ/ΔT/ly^3    # thermal expansion
    dη_dT     = 1e-10/ΔT           # viscosity's temperature dependence
    dt_diff   = 1.0/4.1*min(di...)^2/κ      # diffusive CFL timestep limiter
    dt        = dt_diff # physical time step
    
    # Thermal diffusion ----------------------------------
    # general thermal arrays
    thermal = ThermalArrays(ni)
    
    # physical parameters
    ρ0 = 1/Pra*η0/κ
    ρ = @fill(ρ0, ni...)
    Cp = @fill(1.0, ni...)
    ρCp = @. Cp * ρ
    K = ρCp.*κ
    thermal_parameters = ThermalParameters(K, ρCp)

    pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li, CFL= 0.1 / √2)
    thermal_bc = (flux_x=true, flux_y=false)

    # @parallel (1:nx, 1:ny) init_T!(thermal.T, ΔT, w, 1:nx, 1:ny, di..., lx, ly)

    thermal.T .= [ΔT*exp(-(((ix-1)*di[1]-0.5*lx)/w)^2 -(((iy-1)*di[2]-0.5ly)/w)^2) for ix=1:size(thermal.T,1), iy=1:size(thermal.T,2)]
    thermal.T[:,1  ] .=  ΔT/2.0
    thermal.T[:,end] .= -ΔT/2.0
    
    @parallel assign!(thermal.Told, thermal.T)
    # ----------------------------------------------------

    # Stokes ---------------------------------------------
    ## Allocate arrays needed for every Stokes problem
    # general stokes arrays
    stokes = StokesArrays(ni, Viscous)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di; ϵ = 1e-4)

    ## Setup-specific parameters and fields
    η = @zeros(ni...) # viscosity field
    @parallel viscosity!(η, thermal.T, ΔT, dη_dT)
    fy = -ρ0gα.* thermal.T

    ## Boundary conditions
    freeslip = (freeslip_x=true, freeslip_y=true)
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell = 4, 6
    particles = twoxtwo_particles2D(nxcell, max_xcell, xci[1], xci[2], di[1], di[1], nx-1, ny-1)
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xvi, di)
    # temperature
    pT = similar(particles.coords[1])
    grid2particle_xvertex!(pT, xvi, thermal.T,  particles.coords)

    # gathering_xvertex!(thermal.T, pT, xvi, particles.coords)
    particle_args = (pT,)
    # ----------------------------------------------------

    # Physical time loop
    t = 0.0
    it = 0
    nt = 10
    local iters
    while it < nt
        # Stokes solver ---------------
        iters = solve!(
            stokes, 
            pt_stokes, 
            di, 
            li, 
            max_li,
            freeslip, 
            fy, 
            η; 
            iterMax=10_000
        )

        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------
        
        # Thermal solver ---------------
        pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li; ϵ = 1e-4, CFL= 0.1 / √2)
        iters = solve!(
            thermal,
            pt_thermal,
            thermal_parameters,
            thermal_bc,
            ni,
            di,
            dt;
            iterMax=10_000,
            nout=10,
            verbose=false,
        )
        # ------------------------------

        # Advection --------------------
        # interpolate fields from grid vertices to particles
        grid2particle_xvertex!(pT, xvi, thermal.ΔT,  particles.coords)
        # advect particles in space
        V = (stokes.V.Vx, stokes.V.Vy)
        advection_RK2_edges!(particles, V, grid_vx, grid_vy, dt, 2/3) 
        # advect particles in memory
        shuffle_particles_vertex!(particles, xvi, particle_args)
        # interpolate fields from particle to grid vertices
        gathering_xvertex!(thermal.ΔT, pT, xvi,  particles.coords)
        # check if we need to inject particles
        check_injection(particles) && inject_particles!(particles, args, (thermal.ΔT,), grid)
        # update Temperature field
        @parallel add_dTdt(thermal.T, thermal.ΔT)
    
        # ------------------------------

        # Update viscosity
        @parallel viscosity!(η, thermal.T, ΔT, dη_dT)
        thermal.T[:,1  ] .=  ΔT/2.0
        thermal.T[:,end] .= -ΔT/2.0
        
        # Update buoyancy
        fy = -ρ0gα.* thermal.T

        it += 1
        t += dt

        # @show extrema(stokes.V.Vy)
        # heatmap(xci[1], xci[2], thermal.T , colormap=:vik, colorrange=(-0.5, 0.5))

    end

    return (ni=ni, xci=xci, li=li, di=di), thermal, iters
end

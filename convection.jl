ENV["PS_PACKAGE"] = :Threads

using JustRelax
using Printf, LinearAlgebra

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

include("C:\\Users\\albert\\Desktop\\JustPIC.jl\\src\\JustPIC.jl")
using .JustPIC

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
    pt_stokes = PTStokesCoeffs(ni, di)

    ## Setup-specific parameters and fields
    η = @zeros(ni...) # viscosity field
    @parallel viscosity!(η, thermal.T, ΔT, dη_dT)
    fy = ρ0gα.* thermal.T

    ## Boundary conditions
    freeslip = (freeslip_x=true, freeslip_y=true)
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell = 4 # initial number of particles per cell
    max_xcell = 8 # max number of particles per cell
    min_xcell = 2 # min number of particles per cell
    particles = init_particles(nxcell, max_xcell, min_xcell, xvi..., di..., ni...)
    pT = similar(particles.coords[1])
    grid2particle!(pT, xvi, thermal.T, particles.coords)
    particle_args = (pT,)
    # ----------------------------------------------------

    # Physical time loop
    t = 0.0
    it = 0
    nt = 5
    local iters
    while it < nt
        # Stokes solver
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
        pt_thermal = PTThermalCoeffs(K, ρCp, dt, di, li, CFL= 0.1 / √2)

        # Thermal solver
        iters = solve!(
            thermal,
            pt_thermal,
            thermal_parameters,
            thermal_bc,
            ni,
            di,
            dt;
            iterMax=10_000,
            nout=1,
            verbose=false,
        )

        # Update viscosity
        @parallel viscosity!(η, thermal.T, ΔT, dη_dT)
        # Update buoyancy
        fy = ρ .* g.* thermal.T

        # Particles advection
        grid2particle!(pT, xvi, thermal.ΔT, particles.coords)

        # advect particles in space
        advection_RK2!(particles, (stokes.V.Vx, stokes.V.Vy), xci, di, dt, 0.5)

        # advect particles in memory
        shuffle_particles!(particles, xvi, di, ni, particle_args)

        particle2node!(thermal.ΔT, pT, xvi, particles)
        
        @parallel add_dTdt(thermal.T, thermal.ΔT)
        
        check_injection(particles.inject) && inject_particles!(particles, xvi, ni, di)

        # gathering!(thermal.ΔT, pT, xci, particles.coords, particles.upper_buffer, particles.lower_buffer)

        it += 1
        t += dt
    end

    return (ni=ni, xci=xci, li=li, di=di), thermal, iters
end
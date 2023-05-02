using JustRelax

# needs this branch of GeoParams , uncomment line below to install it
# using Pkg; Pkg.add(url="https://github.com/JuliaGeodynamics/GeoParams.jl"; rev="adm-arrhenius_dim")

# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions

# PARTICLES #################################################
using MuladdMacro
using ParallelStencil.FiniteDifferences2D

using StencilInterpolations

include("src/particles/particles.jl")
include("src/particles/utils.jl")
include("src/particles/advection.jl")
include("src/particles/injection.jl")
include("src/particles/shuffle_vertex.jl")
include("src/particles/staggered/centered.jl")
include("src/particles/staggered/velocity.jl")
include("src/particles/data.jl")

function twoxtwo_particles2D(nxcell, max_xcell, min_xcell, x, y, dx, dy, nx, ny)
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

    if ENV["PS_PACKAGE"] === "CUDA"
        pxi = CuArray.((px, py))
        return Particles(
            pxi, CuArray(index), CuArray(inject), nxcell, max_xcell, min_xcell, np, (nx, ny)
        )

    else
        return Particles(
            (px, py), index, inject, nxcell, max_xcell, min_xcell, np, (nx, ny)
        )
    end
end

# function velocity_grids(xvi::NTuple{2,T}, di) where {T}
#     dx, dy = di
#     # yvx = xvi[2][1]-dy/2:dy:xvi[2][end]+dy/2
#     # xvy = xvi[1][1]-dx/2:dx:xvi[1][end]+dx/2
#     # grid_vx = (xvi[1], yvx)
#     # grid_vy = (xvy, xvi[2])

#     xvx = (xvi[1][1] - dx / 2):dx:(xvi[1][end] + dx / 2)
#     yvy = (xvi[2][1] - dy / 2):dy:(xvi[2][end] + dy / 2)
#     grid_vx = (xvx, xvi[2])
#     grid_vy = (xvi[1], yvy)

#     return grid_vx, grid_vy
# end

function velocity_grids(xci, xvi, di)
    dx, dy = di
    # yvx = xvi[2][1]-dy/2:dy:xvi[2][end]+dy/2
    # xvy = xvi[1][1]-dx/2:dx:xvi[1][end]+dx/2
    # grid_vx = (xvi[1], yvx)
    # grid_vy = (xvy, xvi[2])

    # yVx = (xci[2][1] - dx):dx:(xci[2][end] + dx)
    yVx = LinRange(xci[2][1] - dx, xci[2][end] + dx, length(xci[2])+2)
    xVy = LinRange(xci[1][1] - dy, xci[1][end] + dy, length(xci[1])+2)
    grid_vx = (xvi[1], yVx)
    grid_vy = (xVy, xvi[2])

    return grid_vx, grid_vy
end
#############################################################

# function to compute strain rate (compulsory)
@inline function custom_εII(a::CustomRheology, TauII; args...)
    η = custom_viscosity(a; args...)
    return TauII / η * 0.5
end

# function to compute deviatoric stress (compulsory)
@inline function custom_τII(a::CustomRheology, EpsII; args...)
    η = custom_viscosity(a; args...)
    return 2.0 * η * EpsII
end

# helper function (optional)
@inline function custom_viscosity(a::CustomRheology; P=0.0, T=273.0, depth=0.0, kwargs...)
    (; η0, Ea, Va, T0, R, cutoff) = a.args
    η = η0 * exp((Ea + P * Va) / (R * T) - Ea / (R * T0))
    # correction = (depth ≤ 660e3) + (2740e3 ≥ depth > 660e3) * 1e1  + (depth > 2740e3) * 1e-2
    correction = (depth ≤ 660e3) + (2740e3 ≥ depth > 660e3) * 1e1  + (depth > 2700e3) * 1e-1
    η = clamp(η * correction, cutoff...)
end

# HELPER FUNCTIONS ---------------------------------------------------------------
# visco-elasto-plastic with GeoParams
@parallel_indices (i, j) function compute_viscosity_gp!(η, args, MatParam)

    # convinience closure
    @inline av(T)     = (T[i + 1, j] + T[i + 2, j] + T[i + 1, j + 1] + T[i + 2, j + 1]) * 0.25

    @inbounds begin
        args_ij       = (; dt = args.dt, P = (args.P[i, j]), depth = abs(args.depth[j]), T=av(args.T), τII_old=0.0)
        εij_p         = 1.0, 1.0, (1.0, 1.0, 1.0, 1.0)
        τij_p_o       = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        phases        = 1, 1, (1,1,1,1) # for now hard-coded for a single phase
        # # update stress and effective viscosity
        _, _, η[i, j] = compute_τij(MatParam, εij_p, args_ij, τij_p_o, phases)
    end
    
    return nothing
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = @all(ρg)*(-@all_j(z))
    return nothing
end

# Half-space-cooling model
@parallel_indices (i, j) function init_T!(T, z, κ, Tm, Tp, Tmin, Tmax)
    yr      = 3600*24*365.25
    dTdz    = (Tm-Tp)/2890e3
    zᵢ      = abs(z[j])
    Tᵢ      = Tp + dTdz*(zᵢ)
    time    = 100e6 * yr
    Ths     = Tmin + (Tm -Tmin) * erf((zᵢ)*0.5/(κ*time)^0.5)
    T[i, j] = min(Tᵢ, Ths)
    # Tᵢ      = min(Tᵢ, Ths)
    # time    = 10e6 * yr #6e9 * yr
    # # Ths     = Tmax - (Tmax + Tm) * erf((-minimum(z)-zᵢ)*0.5/(κ*time*5)^0.5)
    # T[i, j] = max(Tᵢ, Ths)
    return 
end


function elliptical_perturbation!(T, δT, xc, yc, r, xvi)

    @parallel_indices (i, j) function _elliptical_perturbation!(T, δT, xc, yc, r, x, y)
        @inbounds if (((x[i]-xc ))^2 + ((y[j] - yc))^2) ≤ r^2
            T[i, j] *= δT/100 + 1
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, δT, xc, yc, r, xvi...)
end

function random_perturbation!(T, δT, xbox, ybox, xvi)

    @parallel_indices (i, j) function _random_perturbation!(T, δT, xbox, ybox, x, y)
        @inbounds if (xbox[1] ≤ x[i] ≤ xbox[2]) && (abs(ybox[1]) ≤ abs(y[j]) ≤ abs(ybox[2]))
            δTi = δT * (rand() -  0.5) # random perturbation within ±δT [%]
            T[i, j] *= δTi/100 + 1
        end
        return nothing
    end
    
    @parallel (@idx size(T)) _random_perturbation!(T, δT, xbox, ybox, xvi...)
end

# --------------------------------------------------------------------------------

@parallel_indices (i, j) function compute_ρg!(ρg, rheology, args)
   
    i1, j1 = i + 1, j + 1
    i2 = i + 2
    @inline av(T) = 0.25 * (T[i1,j] + T[i2,j] + T[i1,j1] + T[i2,j1]) - 273.0

    @inbounds ρg[i, j] = -compute_density(rheology, (; T = av(args.T), P=args.P[i, j])) * compute_gravity(rheology.Gravity[1])

    return nothing
end

Rayleigh_number(ρ, α, ΔT, κ, η0) = ρ * 9.81 * α * ΔT * 2890e3^3 * inv(κ * η0) 

function thermal_convection2D(; ar=8, ny=16, nx=ny*8, figdir="figs2D")

    # Physical domain ------------------------------------
    ly       = 2890e3
    lx       = ly * ar
    origin   = 0.0, -ly                         # origin coordinates
    ni       = nx, ny                           # number of cells
    li       = lx, ly                           # domain length in x- and y-
    di       = @. li / ni                       # grid step in x- and -y
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # create rheology struct
    # v_args = (; η0=5e20, Ea=100e3, Va=1.6e-6, T0=1.6e3, R=8.3145, cutoff=(1e16, 1e25))
    v_args = (; η0=5e20, Ea=200e3, Va=2.6e-6, T0=1.6e3, R=8.3145, cutoff=(1e16, 1e25))
    # v_args = (; η0=5e20, Ea=370e3, Va=3.65e-6, T0=1.6e3, R=8.3145, cutoff=(1e18, 1e25))
    # v_args = (; η0=1.2e21, Ea=35e3, Va=0.0, T0=1.6e3, R=8.3145, cutoff=(1e16, 1e25))
    creep = CustomRheology(custom_εII, custom_τII, v_args)

    # Physical properties using GeoParams ----------------
    η_reg     = 1e8
    G0        = 80e9    # shear modulus
    cohesion  = 30e6*0.0
    friction  = asind(0.01)
    # friction  = 30.0
    pl        = DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    # pl        = DruckerPrager(; C = 30e6, ϕ=friction, Ψ=0.0) # non-regularized plasticity
    el        = SetConstantElasticity(; G=G0, ν=0.45)                             # elastic spring
    β         = inv(get_Kb(el))
    # creep     = ArrheniusType2(; η0 = 1e22, T0=1600, Ea=100e3, Va=1.0e-6)       # Arrhenius-like (T-dependant) viscosity
    # creep     = LinearViscous(; η = 1e22)       # Arrhenius-like (T-dependant) viscosity

    # Define rheolgy struct
    # rheology = SetMaterialParams(;
    #     Name              = "Mantle",
    #     Phase             = 1,
    #     Density           = PT_Density(; ρ0=3.5e3, β=β, T0=0.0, α = 1.5e-5),
    #     HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
    #     Conductivity      = ConstantConductivity(; k=3.0),
    #     CompositeRheology = CompositeRheology((creep, el)),
    #     Elasticity        = el,
    #     Gravity           = ConstantGravity(; g=-9.81),
    # )
    rheology = SetMaterialParams(;
        Name              = "Mantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3.1e3, β=β, T0=0.0, α = 1.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((creep, el, )),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=-9.81),
    )
    rheology_depth = SetMaterialParams(;
        Name              = "Mantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3.5e3, β=β, T0=0.0, α = 1.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((creep, el, pl)),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=-9.81),
    )
    # rheology_depth    = SetMaterialParams(;
    #     Name              = "Mantle",
    #     Phase             = 1,
    #     Density           = PT_Density(; ρ0=3.5e3, β=0.0, T0=0.0, α = 1.5e-5),
    #     HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
    #     Conductivity      = ConstantConductivity(; k=3.0),
    #     CompositeRheology = CompositeRheology((creep, el, pl)),
    #     Elasticity        = el,
    #     Gravity           = ConstantGravity(; g=-9.81),
    # )
    # heat diffusivity
    κ            = (rheology.Conductivity[1].k / (rheology.HeatCapacity[1].cp * rheology.Density[1].ρ0)).val
    dt = dt_diff = 0.5 * min(di...)^2 / κ / 2.01 # diffusive CFL timestep limiter
    # Ra = Rayleigh_number(rheology.Density[1].ρ0.val, rheology.Density[1].α.val, 3e3-300, κ, creep.η) 
    # ----------------------------------------------------
    
    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(; 
        # no_flux     = (left = false, right = false, top = false, bot = false), 
        # periodicity = (left = true, right = true, top = false, bot = false),
        no_flux     = (left = true, right = true, top = false, bot = false), 
        periodicity = (left = false, right = false, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    adiabat     = 0.3 # adiabatic gradient
    Tp          = 1900
    Tm          = Tp + adiabat * 2890
    Tmin, Tmax  = 300.0, 3.5e3
    # thermal.T  .= 1600.0
    @parallel init_T!(thermal.T, xvi[2], κ, Tm, Tp, Tmin, Tmax)
    thermal_bcs!(thermal.T, thermal_bc)
    # Elliptical temperature anomaly 
    δT          = 5.0              # thermal perturbation (in %)
    # random_perturbation!(thermal.T, δT, (lx*1/8, lx*7/8), (-2000e3, -2600e3), xvi)
    random_perturbation!(thermal.T, δT, (lx*1/8, lx*7/8), (-0, -Inf), xvi)
    # δT          = 10.0              # thermal perturbation (in %)
    # xc, yc      = 0.5*lx, -0.75*ly  # origin of thermal anomaly
    # r           = 150e3             # radius of perturbation
    # elliptical_perturbation!(thermal.T, δT, xc, yc, r, xvi)

    # yv = [y for x in xvi[1], y in xvi[2]]./2890e3
    # xv = [x for x in xvi[1], y in xvi[2]]./2890e3
    # thermal.T[2:end-1,:] .+= PTArray(@. exp(-(10*(xv-4)^2 + 80*(yv + 0.75)^2)) * 50)
    @views thermal.T[:, 1]   .= Tmax
    @views thermal.T[:, end] .= Tmin
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(ni, ViscoElastic)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL = 1.0 / √2.1)
    # Buoyancy forces
    ρg              = @zeros(ni...), @zeros(ni...)
    for _ in 1:2
        @parallel (@idx ni) compute_ρg!(ρg[2], rheology, (T=thermal.T, P=stokes.P))
        @parallel init_P!(stokes.P, ρg[2], xci[2])
    end

    # Rheology
    η               = @ones(ni...)
    args_ηv         = (; T = thermal.T, P = stokes.P, depth = xci[2], dt = Inf)
    @parallel (@idx ni) compute_viscosity_gp!(η, args_ηv, (rheology,))
    η_vep           = deepcopy(η)
    dt_elasticity   = Inf
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(; 
        # free_slip = (left=false, right=false, top=true, bot=true),
        # periodicity = (left = true, right = true, top = false, bot = false),
        free_slip   = (left = true, right=true, top=true, bot=true),
        periodicity = (left = false, right = false, top = false, bot = false),
    )
    # ----------------------------------------------------

     # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 8, 10, 6
    particles = twoxtwo_particles2D(
        nxcell, max_xcell, min_xcell, xvi[1], xvi[2], di[1], di[2], nx, ny
    )
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT = similar(particles.coords[1])
    ρCₚp = similar(pT)
    ρCp = @fill(ρg[2][1]/9.81*1200, ni.+1...)
    
    grid2particle_xvertex!(ρCₚp, xvi, ρCp, particles.coords)
 
    # gathering_xvertex!(thermal.T, pT, xvi, particles.coords)
    particle_args = (pT, ρCₚp)
    # ----------------------------------------------------
    
    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles
    fig0 = let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y =  [y for x in xci[1], y in xci[2]][:]
        fig = Figure(resolution = (1200, 900))
        ax1 = Axis(fig[1,1], aspect = 2/3, title = "T")
        ax2 = Axis(fig[1,2], aspect = 2/3, title = "log10(η)")
        lines!(ax1, Array(thermal.T[2:end-1,:][:]), Yv./1e3)
        lines!(ax2, Array(log10.(η[:])), Y./1e3)
        ylims!(ax1, -2890, 0)
        ylims!(ax2, -2890, 0)
        hideydecorations!(ax2)
        fig
        save( joinpath(figdir, "initial_profile.png"), fig)
    end

    # Time loop
    t, it = 0.0, 0
    nt    = 2_000
    local iters
    # while it < nt
    while (t/(1e6 * 3600 * 24 *365.25)) < 4.5e3
        # Update buoyancy and viscosity -
        args_ηv = (; T = thermal.T, P = stokes.P, depth = xci[2], dt=Inf)
        @parallel (@idx ni) compute_viscosity_gp!(η, args_ηv, (rheology,))
        @parallel (@idx ni) compute_ρg!(ρg[2], rheology, (T=thermal.T, P=stokes.P))
        # ------------------------------
 
        # Stokes solver ----------------
        args_η = (; T = thermal.T, P = stokes.P, depth = xci[2], dt=dt)
        λ, iters = solve!(
            stokes,
            thermal,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            args_η,
            # it > 3 ?  (; linear=rheology_depth,) : (; linear=rheology,), # do a few initial time-steps without plasticity to improve convergence
            # (; linear=rheology, plastic=rheology), # do a few initial time-steps without plasticity to improve convergence
            (; linear=rheology_depth, ), # do a few initial time-steps without plasticity to improve convergence
            # (; linear=rheology, plastic=rheology_depth), # do a few initial time-steps without plasticity to improve convergence
            # rheology, # d/o a few initial time-steps without plasticity to improve convergence
            dt,
            iterMax=250e3,
            nout=1e3,
        );

        @show sum(λ)

        println("starting non linear iterations")
        dt = compute_dt(stokes, di, dt_diff) * 1
        # ------------------------------

        # Thermal solver ---------------
        args_T = (; P=stokes.P)
        solve!(
            thermal,
            thermal_bc,
            stokes,
            rheology,
            args_T,
            di,
            dt 
        )
        # ------------------------------

        # Advection --------------------
        # interpolate fields from grid vertices to particles
        grid2particle_xvertex!(pT, xvi, thermal.T, particles.coords)
        # int2part_vertex!(pT, thermal.T, thermal.Told, particles, xvi)
        # advect particles in space
        V = (stokes.V.Vx, stokes.V.Vy)
        advection_RK2!(particles, V, grid_vx, grid_vy, dt, 2 / 3)
        # advect particles in memory
        shuffle_particles_vertex!(particles, xvi, particle_args)
        # check if we need to inject particles
        @show inject = check_injection(particles)
        inject && inject_particles!(particles, particle_args, (thermal.T,), xvi)
        # interpolate fields from particle to grid vertices
        gathering_xvertex!(thermal.T[2:end-1, :], pT, xvi, particles.coords)
        # gather_temperature_xvertex!(thermal.T, pT, ρCₚp, xvi, particles.coords)

        @show it += 1
        t += dt

        # Plotting ---------------------
        if it == 1 || rem(it, 50) == 0
            fig = Figure(resolution = (1000, 1000), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[3,1], aspect = ar, title = "τII [MPa]")
            # ax4 = Axis(fig[4,1], aspect = ar, title = "ρ [kg/m3]")
            # ax4 = Axis(fig[4,1], aspect = ar, title = "τII - τy [Mpa]")
            ax4 = Axis(fig[4,1], aspect = ar, title = "log10(η)")
            h1 = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1].*1e-3, xvi[2].*1e-3, Array(stokes.V.Vy[2:end-1,:]) , colormap=:batlow)
            h3 = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(stokes.τ.II.*1e-6) , colormap=:batlow) 
            h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:batlow)
            # h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(abs.(ρg[2]./9.81)) , colormap=:batlow)
            # h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(@.(stokes.P * friction  + cohesion - stokes.τ.II)/1e6) , colormap=:batlow)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)

            Colorbar(fig[1,2], h1, height=100)
            Colorbar(fig[2,2], h2, height=100)
            Colorbar(fig[3,2], h3, height=100)
            Colorbar(fig[4,2], h4, height=100)
           
            fig
            save( joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

function run()
    figdir = "figs2D"
    ar     = 8 # aspect ratio
    n      = 32
    nx     = n*ar - 2
    ny     = n - 2

    thermal_convection2D(; figdir=figdir, ar=ar,nx=nx, ny=ny);
end

# @time run()

using JustRelax

# needs this branch of GeoParams , uncomment line below to install it
# using Pkg; Pkg.add(url="https://github.com/JuliaGeodynamics/GeoParams.jl"; rev="adm-arrhenius_dim")

# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions

# HELPER FUNCTIONS ---------------------------------------------------------------
@parallel_indices (i, j) function computeViscosity!(η, v, args)

    @inline av(T) = 0.25* (T[i+1,j] + T[i+2,j] + T[i+1,j+1] + T[i+2,j+1])

    @inbounds η[i, j] = computeViscosity_εII(v, 1.0, (; T = av(args.T), P=args.P[i, j], depth=abs(args.depth[j])))

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
@parallel_indices (i, j) function init_T!(T, z, k, Tm, Tp, Tmin, Tmax)
    yr      = 3600*24*365.25
    dTdz    = Tm-Tp
    zᵢ      = abs(z[j])
    Tᵢ      = Tp + dTdz*(zᵢ)
    time    = 100e6 * yr
    Ths     = Tmin + (Tm -Tmin) * erf((zᵢ)*0.5/(k*time)^0.5)
    Tᵢ      = min(Tᵢ, Ths)
    time    = 100e6 * yr #6e9 * yr
    Ths     = Tmax - (Tmax + Tm) * erf((-minimum(z)-zᵢ)*0.5/(k*time*5)^0.5)
    T[i, j] = Tᵢ
    T[i, j] = max(Tᵢ, Ths)

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
# --------------------------------------------------------------------------------

@parallel_indices (i, j) function compute_ρg!(ρg, rheology, args)

    @inline av(T) = 0.25* (T[i+1,j] + T[i+2,j] + T[i+1,j+1] + T[i+2,j+1])

    @inbounds ρg[i, j] = compute_density(rheology, (; T = av(args.T), P=args.P[i, j])) * _compute_gravity(rheology)

    return nothing
end
_compute_gravity(v::MaterialParams) = compute_gravity(v.Gravity[1])

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

    # Physical properties using GeoParams ----------------
    η_reg     = 1e16
    G0        = Inf                                                             # shear modulus
    cohesion  = 30e6
    pl        = DruckerPrager_regularised(; C = cohesion, ϕ=30.0, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    el        = SetConstantElasticity(; G=G0, ν=0.5)                             # elastic spring
    creep     = ArrheniusType2(; η0 = 1e22, T0=1600, Ea=100e3, Va=1.0e-6)       # Arrhenius-like (T-dependant) viscosity
    # Define rheolgy struct
    rheology = SetMaterialParams(;
        Name              = "Mantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3300, β=0.0, α = 3e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((creep, el)),
        Elasticity        = SetConstantElasticity(; G=G0, ν=0.5),
        Gravity           = ConstantGravity(; g=9.81),
    )
    rheology_depth    = SetMaterialParams(;
        Name              = "Mantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3300, β=0.0, α = 3e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((creep, el, pl)),
        Elasticity        = SetConstantElasticity(; G=G0, ν=0.5),
        Gravity           = ConstantGravity(; g=9.81),
    )
    # heat diffusivity
    κ            = (rheology.Conductivity[1].k / (rheology.HeatCapacity[1].cp * rheology.Density[1].ρ0)).val
    dt = dt_diff = 0.5 / 4.1 * min(di...)^2 / κ # diffusive CFL timestep limiter
    # ----------------------------------------------------
    
    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(; 
        no_flux     = (left = false, right = false, top = false, bot = false), 
        periodicity = (left = true, right = true, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    k           = 3/4500/1200
    Tm, Tp      = 1900, 1600
    Tmin, Tmax  = 300.0, 3e3
    @parallel init_T!(thermal.T, xvi[2], k, Tm, Tp, Tmin, Tmax)
    thermal_bcs!(thermal.T, thermal_bc)
    # Elliptical temperature anomaly 
    xc, yc      =  0.5*lx, -0.75*ly  # origin of thermal anomaly
    δT          = 10.0              # thermal perturbation (in %)
    r           =  150e3             # radius of perturbation
    elliptical_perturbation!(thermal.T, δT, xc, yc, r, xvi)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(ni, ViscoElastic)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-5,  CFL=1 / √2.1)
    # Buoyancy forces
    ρg              = @zeros(ni...), @zeros(ni...)
    @parallel (@idx ni) compute_ρg!(ρg[2], rheology, (T=thermal.T, P=stokes.P))
    @parallel init_P!(stokes.P, ρg[2], xci[2])
    # Rheology
    η               = @ones(ni...)
    args_η          = (; T = thermal.T, P = stokes.P, depth = xci[2], dt=Inf)
    @parallel (@idx ni) computeViscosity!(η, rheology.CompositeRheology[1], args_η) # init viscosity field
    η_vep           = deepcopy(η)
    dt_elasticity   = Inf
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(; 
        free_slip = (left=false, right=false, top=true, bot=true),
        periodicity = (left = true, right = true, top = false, bot = false),
    )
    # ----------------------------------------------------

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it = 0.0, 0
    nt    = 500
    v     = rheology.CompositeRheology[1]
    local iters
    while it < nt

        # Update buoyancy and viscosity -
        @parallel (@idx ni) computeViscosity!(η, v, args_η)
        @parallel (@idx ni) compute_ρg!(ρg[2], rheology, (T=thermal.T, P=stokes.P))
        # ------------------------------

        # Stokes solver ----------------
        args_η = (; T = thermal.T, P = stokes.P, depth = xci[2], dt=Inf)
        iters = solve!(
            stokes,
            thermal,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            η_vep,
            args_η,
            it > 3 ? rheology_depth : rheology, # do a few initial time-steps without plasticity to improve convergence
            dt_elasticity,
            iterMax=150e3,
            nout=500,
        )
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Thermal solver ---------------
        args_T = (; T=thermal.T, P=stokes.P)
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

        @show it += 1
        t += dt

        # Plotting ---------------------
        if it == 1 || rem(it, 25) == 0
            fig = Figure(resolution = (900, 1400), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy")
            ax3 = Axis(fig[3,1], aspect = ar, title = "τII")
            ax4 = Axis(fig[4,1], aspect = ar, title = "η")
            h1 = heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1].*1e-3, xvi[2].*1e-3, Array(stokes.V.Vy[2:end-1,:]) , colormap=:batlow)
            h3 = heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(stokes.τ.II) , colormap=:romaO) 
            h4 = heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:batlow)
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

figdir = "figs2D"
ar     = 8 # aspect ratio
n      = 32
nx     = n*ar - 2
ny     = n - 2

thermal_convection2D(; figdir=figdir, ar=ar,nx=nx, ny=ny);


#  # Compute some constant stuff
#  _dx, _dy = inv.(di)
#  nx, ny = size(thermal.T)

#  # solve heat diffusion
#  @parallel assign!(thermal.Told, thermal.T)
 
#  environment!(model)
#  @parallel (1:(nx - 1), 1:(ny - 1)) JustRelax.ThermalDiffusion2D.compute_flux!(
#      thermal.qTx, thermal.qTy, thermal.T, rheology, args_T, _dx, _dy
#  )

#  @parallel JustRelax.ThermalDiffusion2D.advect_T!(
#      thermal.dT_dt,
#      thermal.qTx,
#      thermal.qTy,
#      thermal.T,
#      stokes.V.Vx,
#      stokes.V.Vy,
#      _dx,
#      _dy,
#  )
#  @parallel update_T!(thermal.T, thermal.dT_dt, dt)
#  thermal_bcs!(thermal.T, thermal_bc)

#  @. thermal.ΔT = thermal.T - thermal.Told
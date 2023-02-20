using JustRelax
# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 2)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions

# HELPER FUNCTIONS ---------------------------------------------------------------
@parallel function update_buoyancy!(fz, T, ρ0gα)
    @all(fz) = ρ0gα * @all(T)
    return nothing
end

@parallel_indices (i, j) function computeViscosity!(η, v, args)

    @inline av(T) = 0.25* (T[i,j] + T[i+1,j] + T[i,j+1] + T[i+1,j+1])

    # @inbounds η[i, j] = computeViscosity_εII(v, 1.0, (; T = av(args.T), P=args.P[i, j]))
    @inbounds η[i, j] = computeViscosity_εII(v, 1.0, (; T = av(args.T), P=args.P[i, j], depth=abs(args.depth[j])))

    return nothing
end

import ParallelStencil.INDICES
const idx_j = INDICES[2]
macro all_j(A)
    esc(:($A[$idx_j]))
end

@parallel function foo!(P, ρg, z)
    @all(P) = @all(ρg)*@all_j(z)
    return nothing
end

# Half-space-cooling model
@parallel_indices (i, j) function init_T!(T, z, k, Tm, Tp, Tmin, Tmax)
    yr          = 3600*24*365.25
    dTdz = Tm-Tp
    @inbounds zᵢ = abs(z[j])
    Tᵢ          = Tp + dTdz*(zᵢ)
    time        = 500e6 * yr
    Ths         = Tmin + (Tm -Tmin) * erf((zᵢ)*0.5/(k*time)^0.5)
    Tᵢ          = min(Tᵢ, Ths)
    time        = 500e6 * yr #6e9 * yr
    Ths         = Tmax - (Tmax + Tm) * erf((maximum(z)-zᵢ)*0.5/(k*time*5)^0.5)
    @inbounds T[i, j] =  max(Tᵢ, Ths)

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
    @inbounds ρg[i, j] = compute_density(rheology, ntuple_idx(args, i, j)) * _compute_gravity(rheology)
    return nothing
end
_compute_gravity(v::MaterialParams) =  compute_gravity(v.Gravity[1])

function thermal_convection2D(; ar=8, ny=16, nx=ny*8, figdir="figs2D")

    # Physical domain ------------------------------------
    CharUnits = GEO_units(; viscosity=1e23, length=2890km, temperature=1000K)
    ly       = 2890e3
    lx       = ly * ar
    origin   = 0.0, -ly                         # origin coordinates
    ni       = nx, ny                           # number of cells
    li       = lx, ly                           # domain length in x- and y-
    di       = @. li / ni                       # grid step in x- and -y
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    η_reg     = 1e20
    G0        = Inf                                                             # shear modulus
    cohesion  = 30e6
    pl        = DruckerPrager_regularised(; C = cohesion, ϕ=30.0, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    el        = SetConstantElasticity(; G=G0, ν=0.5)                             # elastic spring
    creep     = ArrheniusType2(; η0 = 5e20, T0=1600, Ea=100e3, Va=1.0e-6)       # Arrhenius-like (T-dependant) viscosity
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
    κ            = (rheology.Conductivity[1].k / (rheology.HeatCapacity[1].cp * rheology.Density[1].ρ0)).val
    # heat diffusivity
    dt = dt_diff = 0.5 / 6.1 * min(di...)^2 / κ # diffusive CFL timestep limiter
    # ----------------------------------------------------
    
    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(; 
        no_flux     = (left = true, right = true, top = false, bot = false), 
        periodicity = (left = false, right = false, top = false, bot = false),
    )
    # initialize thermal profile - Half space cooling
    k           = 3/4500/1200
    Tm, Tp      = 1900, 1600
    Tmin, Tmax  = 3e2, 1.9e3#4e3
    @parallel init_T!(thermal.T, xvi[2], k, Tm, Tp, Tmin, Tmax)
    # Elliptical temperature anomaly 
    xc, yc      =  0.5*lx, -0.75*ly  # origin of thermal anomaly
    δT          = 10.0              # thermal perturbation (in %)
    r           =  150e3             # radius of perturbation
    elliptical_perturbation!(thermal.T, δT, xc, yc, r, xvi)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(ni, ViscoElastic)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-6,  CFL=1 / √2.1)
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
        free_slip = (left=true, right=true, top=true, bot=true), 
    )
    # ----------------------------------------------------

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # ----------------------------------------------------

    # Time loop
    t, it = 0.0, 0
    nt    = 150
    local iters
    while it < nt

        # Update buoyancy and viscosity -
        @parallel (@idx ni) computeViscosity!(η, rheology.CompositeRheology[1], args_η)
        @parallel (@idx ni) compute_ρg!(ρg[2], rheology, (T=thermal.T, P=stokes.P))
        # @parallel update_buoyancy!(ρg[2], thermal.T, -Ra)
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
            dt,
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
        if it == 1 || rem(it, 1) == 0
            fig = Figure(resolution = (900, 1600), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy")
            ax3 = Axis(fig[3,1], aspect = ar, title = "τII")
            ax4 = Axis(fig[4,1], aspect = ar, title = "η")
            h1 = heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1], xvi[2], Array(stokes.V.Vy[2:end-1,:]) , colormap=:batlow)
            h3 = heatmap!(ax3, xci[1], xci[2], Array(stokes.τ.II) , colormap=:romaO) 
            h4 = heatmap!(ax4, xci[1], xci[2], Array(log10.(η_vep)) , colormap=:batlow)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[3,2], h3)
            Colorbar(fig[4,2], h4)
            fig
            save( joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

figdir = "figs2D"
ar     = 3 # aspect ratio
n      = 128
nx     = n*ar - 2
ny     = n - 2

thermal_convection2D(; figdir=figdir, ar=ar,nx=nx, ny=ny);
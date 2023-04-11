using JustRelax

# needs this branch of GeoParams , uncomment line below to install it
# using Pkg; Pkg.add(url="https://github.com/JuliaGeodynamics/GeoParams.jl"; rev="adm-arrhenius_dim")

# setup ParallelStencil.jl environment
model = PS_Setup(:gpu, Float64, 3)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions

# function to compute strain rate (compulsory)
@inline function custom_εII(a::CustomRheology, TauII; args...)
    η = custom_viscosity(a; args...)
    return (TauII / η) * 0.5
end

# function to compute deviatoric stress (compulsory)
@inline function custom_τII(a::CustomRheology, EpsII; args...)
    η = custom_viscosity(a; args...)
    return 2.0 * (η * EpsII)
end

# helper function (optional)
@inline function custom_viscosity(a::CustomRheology; T=273.0, P=0.0, depth=0.0, kwargs...)
    (; η0, Ea, Va, T0, R, cutoff) = a.args
    # T += 273 # to Kelvin
    η = η0 * exp((Ea + P * Va) / (R * T) - Ea / (R * T0))
    correction = (depth ≤ 660e3) + (depth > 660e3) * 1e1
    # correction = 1e0
    return clamp(η * correction, cutoff...)
end

# HELPER FUNCTIONS ---------------------------------------------------------------
# visco-elasto-plastic with GeoParams
@parallel_indices (i, j, k) function compute_viscosity_gp!(η, args, MatParam)

    # convinience closure
    Base.@propagate_inbounds @inline av(T) = 0.125* (
        T[i, j, k  ] + T[i+1, j, k  ] + T[i, j+1, k  ] + T[i+1, j+1, k  ] +
        T[i, j, k+1] + T[i+1, j, k+1] + T[i, j+1, k+1] + T[i+1, j+1, k+1]
    )

    _ones = 1.0, 1.0, 1.0, 1.0
    _zeros = 0.0, 0.0, 0.0, 0.0
    @inbounds begin
        args_ij  = (; dt = args.dt, P = (args.P[i, j, k]), depth = abs(args.depth[k]), T=av(args.T), τII_old=0.0)
        εij_p    = 1.0, 1.0, 1.0, _ones, _ones, _ones
        τij_p_o  = 0.0, 0.0, 0.0, _zeros, _zeros, _zeros
        phases   = 1, 1, 1, (1,1,1,1), (1,1,1,1), (1,1,1,1) # for now hard-coded for a single phase
        # # update stress and effective viscosity
        _, _, η[i, j, k] = compute_τij(MatParam, εij_p, args_ij, τij_p_o, phases)
    end
    
    return nothing
end

import ParallelStencil.INDICES
const idx_k = INDICES[3]
macro all_k(A)
    esc(:($A[$idx_k]))
end

@parallel function init_P!(P, ρg, z)
    @all(P) = -@all(ρg)*(abs(@all_k(z)))
    return nothing
end

# Half-space-cooling model
@parallel_indices (i, j, k) function init_T!(T, z, κ, Tm, Tp, Tmin, Tmax)
    year       = 3600*24*365.25
    dTdz       = (Tm-Tp)/2890e3
    zᵢ         = abs(z[k])
    Tᵢ         = Tp + dTdz*(zᵢ)
    time       = 500e6 * year
    Ths        = Tmin + (Tm - Tmin) * erf((zᵢ)*0.5/(κ*time)^0.5)
    Tᵢ         = min(Tᵢ, Ths)
    time       = 500e6 * year
    Ths        = Tmax - (Tmax + Tm) * erf((-minimum(z)-zᵢ)*0.5/(κ*time*5)^0.5)
    T[i, j, k] = max(Tᵢ, Ths)
    return 
end

function elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi)

    @parallel_indices (i, j, k) function _elliptical_perturbation!(T, δT, xc, yc, zc, r, x, y, z)
        @inbounds if (((x[i]-xc ))^2 + ((y[j] - yc))^2 + ((z[k] - zc))^2) ≤ r^2
            T[i,j,k] *= δT/100 + 1
        end
        return nothing
    end

    @parallel _elliptical_perturbation!(T, δT, xc, yc, zc, r, xvi...)

end
# --------------------------------------------------------------------------------

@parallel_indices (i, j, k) function compute_ρg!(ρg, rheology, args)

    # convinience closure
    Base.@propagate_inbounds @inline av(T) = 0.125* (
        T[i, j, k  ] + T[i+1, j, k  ] + T[i, j+1, k  ] + T[i+1, j+1, k  ] +
        T[i, j, k+1] + T[i+1, j, k+1] + T[i, j+1, k+1] + T[i+1, j+1, k+1]
    )
    
    @inbounds ρg[i, j, k] = -compute_density(rheology, (; T = av(args.T), P=args.P[i, j, k])) * _gravity(rheology)

    return nothing
end
_gravity(v::MaterialParams) = compute_gravity(v.Gravity[1])

function thermal_convection2D(; ar=8, nz=16, ny=nz*8, nx=nz*8, figdir="figs2D")

    # Physical domain ------------------------------------
    lz       = 2890e3
    lx = ly  = lz * ar
    origin   = 0.0, 0.0, -lz                        # origin coordinates
    ni       = nx, ny, nz                           # number of cells
    li       = lx, ly, lz                           # domain length in x- and y-
    di       = @. li / ni                           # grid step in x- and -y
    xci, xvi = lazy_grid(di, li, ni; origin=origin) # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Init MPI -------------------------------------------
    igg  = IGG(
        init_global_grid(nx, ny, nz; init_MPI=false)...
    )
    ni_v = (nx-2)*igg.dims[1], (ny-2)*igg.dims[2], (nz-2)*igg.dims[3]
    # ----------------------------------------------------
        
    # Physical properties using GeoParams ----------------
    η_reg     = 1e16
    G0        = 70e9                                                             # shear modulus
    cohesion  = 30e6
    pl        = DruckerPrager_regularised(; C = cohesion, ϕ=30.0, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
    el        = SetConstantElasticity(; G=G0, ν=0.5)                             # elastic spring
    # creep_var = (; η0=5e20, Ea=100e3, Va=1.6e-6, T0=1.6e3, R=8.3145, cutoff=(1e16, 1e25))
    creep_var = (; η0=5e20, Ea=200e3, Va=2.6e-6, T0=1.6e3, R=8.3145, cutoff=(1e16, 1e25))
    creep     = CustomRheology(custom_εII, custom_τII, creep_var)
    # creep = LinearViscous(; η=1e21)

    # Define rheolgy struct
    rheology = SetMaterialParams(;
        Name              = "Mantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=4000, β=0.0, α = 3e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((creep, el)),
        Elasticity        = SetConstantElasticity(; G=G0, ν=0.5),
        Gravity           = ConstantGravity(; g=9.81),
    )
    rheology_depth    = SetMaterialParams(;
        Name              = "Mantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=4000, β=0.0, α = 3e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((creep, el, pl)),
        Elasticity        = SetConstantElasticity(; G=G0, ν=0.5),
        Gravity           = ConstantGravity(; g=9.81),
    )
    # heat diffusivity
    κ            = (rheology.Conductivity[1].k / (rheology.HeatCapacity[1].cp * rheology.Density[1].ρ0)).val
    dt = dt_diff = 1.0 / (3.1*(2.0 / min(di...)^2)) / κ # diffusive CFL timestep limiter
    # dt = Inf # diffusive CFL timestep limiter
    # ----------------------------------------------------
    
    # TEMPERATURE PROFILE --------------------------------
    thermal    = ThermalArrays(ni)
    thermal_bc = TemperatureBoundaryConditions(; 
        no_flux     = (left = true, right = true, top = false, bot = false, front=true, back=true), 
        periodicity = (left = false, right = false, top = false, bot = false, front=false, back=false),
    )
    # initialize thermal profile - Half space cooling
    k           = 3/4500/1200
    Tm, Tp      = 1900, 1600
    Tmin, Tmax  = 300.0, 4e3
    @parallel init_T!(thermal.T, xvi[3], k, Tm, Tp, Tmin, Tmax)
    # thermal.T  .= 1e3
    thermal_bcs!(thermal.T, thermal_bc)
    # Elliptical temperature anomaly 
    xc, yc, zc  =  0.5*lx, 0.5*ly,-0.75*lz  # origin of thermal anomaly
    δT          = 10.0              # thermal perturbation (in %)
    r           =  150e3             # radius of perturbation
    elliptical_perturbation!(thermal.T, δT, xc, yc, zc, r, xvi)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes          = StokesArrays(ni, ViscoElastic)
    pt_stokes       = PTStokesCoeffs(li, di; ϵ=1e-4,  CFL=1 / √3.1)
    # Buoyancy forces
    ρg              = @zeros(ni...), @zeros(ni...),  @zeros(ni...)
    @parallel (@idx ni) compute_ρg!(ρg[3], rheology, (T=thermal.T, P=stokes.P))
    @parallel init_P!(stokes.P, ρg[3], xci[3])
    # Rheology
    η               = @ones(ni...)
    args_ηv          = (; T = thermal.T, P = stokes.P, depth = xci[3], dt = Inf)
    @parallel (@idx ni) compute_viscosity_gp!(η, args_ηv, (rheology,))
    η_vep           = deepcopy(η)
    dt_elasticity   = Inf
    # Boundary conditions
    flow_bcs = FlowBoundaryConditions(; 
        free_slip   = (left=true , right=true , top=true , bot=true , front=true , back=true ),
        no_slip     = (left=false, right=false, top=false, bot=false, front=false, back=false),
        periodicity = (left=false, right=false, top=false, bot=false, front=false, back=false),
    )
    # ----------------------------------------------------

    # IO ----- -------------------------------------------
    # if it does not exist, make folder where figures are stored
    !isdir(figdir) && mkpath(figdir)
    # mpi arrays
    Tc = @zeros(ni)
    Tg = zeros(ni_v...)
    ηg = zeros(ni_v...)
    # ----------------------------------------------------

    # Time loop
    t, it = 0.0, 0
    nt    = 15
    local iters
    while it < nt

        # Update buoyancy and viscosity -
        args_ηv = (; T = thermal.T, P = stokes.P, depth = xci[3], dt=Inf)
        @parallel (@idx ni) compute_viscosity_gp!(η, args_ηv, (rheology,))
        @parallel (@idx ni) compute_ρg!(ρg[3], rheology, (T=thermal.T, P=stokes.P))
        # ------------------------------

        # Stokes solver ----------------
        args_η = (; T = thermal.T, P = stokes.P, depth = xci[3], dt=Inf)
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
            igg;
            iterMax=150e3,
            nout=1e3,
        )
        dt = compute_dt(stokes, di, dt_diff)
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

        @show it += 1
        t += dt

        # Plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            @parallel (1:nx, 1:ny, 1:nz) vertex2center!(Tc, thermal.T)
            gather!(Array(Tc[2:end-1, 2:end-1, 2:end-1]), Tg)
            gather!(Array(η_vep[2:end-1, 2:end-1, 2:end-1]), ηg)

            fig = Figure(resolution = (900, 1800))
            ax1 = Axis(fig[1,1], aspect = ar, title = "T")
            ax2 = Axis(fig[2,1], aspect = ar, title = "η")
            ax3 = Axis(fig[3,1], aspect = ar, title = "τII")
            ax4 = Axis(fig[4,1], aspect = ar, title = "Vz")
            h1 = heatmap!(ax1, xci[1][2:end-1], xci[2][2:end-1], Array(Tg[nx÷2,:,:]) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1][2:end-1], xci[2][2:end-1], Array(log10.(ηg[nx÷2,:,:])) , colormap=:batlow)
            h3 = heatmap!(ax3, xci[1][2:end-1], xci[2][2:end-1], Array(stokes.τ.II[2:end-1,ny÷2,:]) , colormap=:batlow)
            h4 = heatmap!(ax4, xci[1][2:end-1], xvi[2][2:end-1], Array(stokes.V.Vz[2:end-1,ny÷2,:]) , colormap=:batlow)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[3,2], h3)
            Colorbar(fig[4,2], h4)
            fig
            save( joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    finalize_global_grid(; finalize_MPI=false)

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

figdir  = "figs3D"
ar      = 3 # aspect ratio
n       = 32
nx = ny = n*ar - 2
nz      = n - 2

thermal_convection2D(; figdir=figdir, ar=ar, nx=nx, ny=ny, nz=nz);
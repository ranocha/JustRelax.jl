using GeoParams, GLMakie, SpecialFunctions   

# Half-space-cooling model

function init_T!(T, z, κ, Tm, Tp, Tmin, Tmax)
    for i in eachindex(T)
        yr   = 3600*24*365.25
        dTdz = (Tm-Tp)/2890e3
        zᵢ   = abs(z[i])
        Tᵢ   = Tp + dTdz*(zᵢ)
        time = 10e6 * yr
        Tᵢ   = min(Tmin + (Tm -Tmin) * erf((zᵢ)*0.5/(κ*time)^0.5), Tᵢ)
        T[i] = Tᵢ
        time = 10e6 * yr 
        Ths  = Tmax - (Tmax + Tm) * erf((maximum(z)-zᵢ)*0.5/(κ*time*5)^0.5)
        T[i] = max(Tᵢ, Ths)
    end
    return 
end

function init_P!(P, depth, rheology, args)
   
    for i in eachindex(P)
        P[i] = - depth[i] * compute_density(rheology, ntuple_idx(args, i)) * compute_gravity(rheology.Gravity[1])
    end

    return nothing
end

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
    correction = (depth ≤ 660e3) + (depth > 660e3) * 1e1
    correction = (depth ≤ 660e3) + (2740e3 ≥ depth > 660e3) * 1e1  + (depth > 2740e3) * 1e-1
    η = clamp(η * correction, cutoff...)
    # correction = (depth ≤ 2700e3) + (depth > 2700e3) * 1e-1
    # η = clamp(η * correction, cutoff...)
end

function compute_viscosity_gp!(η, args, rheology)

    for i in eachindex(η)
        args_ij       = (; dt = dt, P = args.P[i], depth = abs(args.depth[i]), T=args.T[i], τII_old=0.0)
        εij_p         = 1.0, 1.0, (1.0, 1.0, 1.0, 1.0)
        τij_p_o       = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        phases        = 1, 1, (1,1,1,1) # for now hard-coded for a single phase
        # # update stress and effective viscosity
        _, _, η[i] = compute_τij((rheology,), εij_p, args_ij, τij_p_o, phases)
    end
    return nothing
end

function compute_viscosity!(η, args, rheology, ε)
    phase = rheology[1].Phase
    for i in eachindex(η)
        args_ij       = (; dt = args.dt, P = args.P[i], depth = args.depth[i], T=args.T[i], τII_old=0.0)
        εij_p         = ε, ε, (1.0, 1.0, 1.0, 1.0).*ε
        τij_p_o       = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        phases        = phase, phase, phase.*(1,1,1,1) # for now hard-coded for a single phase
        # # update stress and effective viscosity
        _, _, ηi = compute_τij(rheology, εij_p, args_ij, τij_p_o, phases)
        η[i] = clamp(ηi, 1e16, 1e25)
    end
    return nothing
end

function viscosity_profile(rheology; ε=1e-15)
    # allocate arrays
    n = 6600
    depth = LinRange(0, 660e3, n)
    T, P = zeros(n), zeros(n)

    # initialize thermal profile - Half space cooling
    adiabat     = 0.3 # adiabatic gradient
    Tp          = 1900
    Tm          = Tp + adiabat * 2890
    Tmin, Tmax  = 300.0, 3.5e3
    κ           = 7.142857142857143e-7
    init_T!(T, depth, κ, Tm, Tp, Tmin, Tmax)

    # Initialize pressure
    args = (; T=T, P=P)
    for _ in 1:5
        init_P!(P, depth, rheology, args)
    end

    η    = zeros(n)
    dt   = 100e3 * 3600 * 24 *365
    dt   = Inf
    args = (; T=T, P=P, depth=depth, dt=dt)
    compute_viscosity!(η, args, (rheology,), ε)
    
    return η, T, depth
end

function init_rheology(creep1, creep2)
    # Physical properties using GeoParams ----------------
    η_reg     = 1e8
    G0        = 80e9    # shear modulus
    cohesion  = 30e6*0.0
    friction  = asind(0.01)
    el        = SetConstantElasticity(; G=G0, ν=0.45)                             # elastic spring
    β         = inv(get_Kb(el))

    # Define rheolgy struct
    rheology = SetMaterialParams(;
        Name              = "Mantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3.5e3, β=β, T0=0.0, α = 1.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((creep1, creep2, el)),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=-9.81),
    )
end

v_args = (; η0=5e20, Ea=200e3, Va=2.6e-6, T0=1.6e3, R=8.3145, cutoff=(1e16, 1e25))
custom = CustomRheology(custom_εII, custom_τII, v_args)

dif1 = SetDiffusionCreep("Dry Anorthite | Rybacki et al. (2006)")
dif2 = SetDiffusionCreep("Wet Olivine | Mei & Kohlstedt (2000a)")
disl = SetDislocationCreep("Dry Olivine | Hirth & Kohlstedt (2003)")

rheology_cust = init_rheology(custom, custom)
rheology_dif = init_rheology(dif1, disl)

ηdif , T, depth = viscosity_profile(rheology_dif)
ηcust, T, depth = viscosity_profile(rheology_cust)
x=init_rheologies()

η, T, depth = viscosity_profile(x[3]; ε=1e-11)

f = Figure()
ax1 = Axis(f[1,1]) 
ax2 = Axis(f[1,2]) 
lines!(ax1, T, -depth./1e3, color=:black)
for xi in x
    η, = viscosity_profile(xi; ε=1e-20)
    lines!(ax2, log10.(η), -depth./1e3, label=join(xi.Name))
end
# scatter!(ax2, log10.(η), -depth./1e3, color=:black)
lines!(ax2, log10.(ηdif), -depth./1e3, color=:red)
# lines!(ax2, log10.(ηcust), -depth./1e3, color=:red)
ax1.xlabel = "T (K)"
ax1.ylabel = "deepth (km)"
axislegend(ax2, position=:lb)
f


lines!(ax1, Array(thermal.T[2:end-1,:][:]), Yv./1e3)
lines!(ax2, Array(log10.(η[:])), Y./1e3)

Name::NTuple{N,Char}
    n::GeoUnit{T,U1} # power-law exponent
    r::GeoUnit{T,U1} # exponent of water-fugacity
    A::GeoUnit{T,U2} # material specific rheological parameter
    E::GeoUnit{T,U3} # activation energy
    V::GeoUnit{T,U4} # activation volume

disl_upper_crust            = DislocationCreep(A=1e-3   , n=2.0, E=167e3, V=0.0 ,  r=0.0, R=8.3145)
disl_lower_crust            = DislocationCreep(A=3.27e-3, n=3.2, E=238e3, V=0.0 ,  r=0.0, R=8.3145)
disl_lithospheric_mantle    = DislocationCreep(A=1.1e-5 , n=3.5, E=530e3, V=17.0,  r=0.0, R=8.3145)
disl_sublithospheric_mantle = DislocationCreep(A=1.1e-5 , n=3.5, E=530e3, V=20.0,  r=0.0, R=8.3145)
diff_lithospheric_mantle    = DiffusionCreep(A=2.46e-5, n=1.0, E=375e3, V=10.0,  r=0.0, R=8.3145)
diff_sublithospheric_mantle = DiffusionCreep(A=2.46e-5, n=1.0, E=375e3, V=10.0,  r=0.0, R=8.3145)


# Physical properties using GeoParams ----------------
η_reg     = 1e16
G0        = 80e9    # shear modulus
cohesion  = 30e6
# friction  = asind(0.01)
friction  = 20.0
pl        = DruckerPrager_regularised(; C = cohesion, ϕ=friction, η_vp=η_reg, Ψ=0.0) # non-regularized plasticity
# pl        = DruckerPrager(; C = 30e6, ϕ=friction, Ψ=0.0) # non-regularized plasticity
el        = SetConstantElasticity(; G=G0, ν=0.5)                             # elastic spring
β         = inv(get_Kb(el))

# Define rheolgy struct
rheology = (
    SetMaterialParams(;
        Name              = "UpperCrust",
        Phase             = 1,
        Density           = PT_Density(; ρ0=2.8e3, β=β, T0=0.0, α = 2.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=2.7),
        CompositeRheology = CompositeRheology((disl_upper_crust, el, )),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=-9.81),
    ),
    SetMaterialParams(;
        Name              = "LowerCrust",
        Phase             = 2,
        Density           = PT_Density(; ρ0=2.9e3, β=β, T0=0.0, α = 2.5e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=2.7),
        CompositeRheology = CompositeRheology((disl_lower_crust, el, )),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=-9.81),
    )
    SetMaterialParams(;
        Name              = "LithosphericMantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3e3, β=β, T0=0.0, α = 3e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.0),
        CompositeRheology = CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, el, )),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=-9.81),
    ),
    SetMaterialParams(;
        Name              = "SubLithosphericMantle",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3e3, β=β, T0=0.0, α = 3e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.3),
        CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el, )),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=-9.81),
    ),
    SetMaterialParams(;
        Name              = "Plume",
        Phase             = 1,
        Density           = PT_Density(; ρ0=3e3, β=β, T0=0.0, α = 3e-5),
        HeatCapacity      = ConstantHeatCapacity(; cp=1.2e3),
        Conductivity      = ConstantConductivity(; k=3.3),
        CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el, )),
        Elasticity        = el,
        Gravity           = ConstantGravity(; g=-9.81),
    ),
)


using GeoParams, GLMakie, SpecialFunctions   

# Half-space-cooling model

function init_T!(T, z)
    for i in eachindex(T)
        zi   = abs(z[i])
        if zi ≤ 35e3
            dTdz = 600 / 35e3
            T[i] = dTdz * zi + 273

        elseif 120e3 ≥ zi > 35e3
            dTdz = 700 / 85e3
            T[i] = dTdz * (zi-35e3) + 600 + 273

        else
            T[i] = 1280 * 3e-5 * 9.81 * (zi-120e3) / 1200 + 1300 + 273
        end
    end
    return 
end

function init_P!(P, depth, rheology, args)
   
    for i in eachindex(P)
        P[i] = - depth[i] * compute_density(rheology, ntuple_idx(args, i)) * -9.81
    end

    return nothing
end

function compute_viscosity!(η, args, rheology, ε)
    for i in eachindex(η)

        rheology_i = if args.depth[i] < 17.5e3
            rheology[1]
        elseif 35e3 > args.depth[i] > 17.5e3
            rheology[2]
        elseif 120e3 > args.depth[i] > 35e3
            rheology[3]
        elseif args.depth[i] > 120e3
            rheology[4]
        end

        phase = rheology_i.Phase

        args_ij       = (; dt = Inf, P = args.P[i], depth = args.depth[i], T=args.T[i], τII_old=0.0)
        εij_p         = ε, ε, (1.0, 1.0, 1.0, 1.0).*ε
        τij_p_o       = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        phases        = phase, phase, phase.*(1,1,1,1) # for now hard-coded for a single phase
        # # update stress and effective viscosity
        _, _, ηi = compute_τij((rheology_i,), εij_p, args_ij, τij_p_o, phases)
        if i == 4000
            @show phase, ηi, args_ij
        end
        # _, _, ηi = compute_τij(rheology[3], εij_p, args_ij, τij_p_o, phases)
        η[i] = clamp(2*ηi, 1e16, 1e25)
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
    Tp          = 1400
    Tm          = Tp + adiabat * 2890
    Tmin, Tmax  = 300.0, 3.5e3
    κ           = 7.142857142857143e-7
    init_T!(T, depth)

    # Initialize pressure
    args = (; T=T.-273, P=P)
    for _ in 1:1
        init_P!(P, depth, rheology[end], args)
    end

    η    = zeros(n)
    dt   = 100e3 * 3600 * 24 *365
    dt   = Inf
    args = (; T=T, P=P, depth=depth, dt=dt)
    compute_viscosity!(η, args, rheology, ε)
    
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

x=init_rheologies()

η, T, depth = viscosity_profile(x; ε=1e-15)

f = Figure()
ax1 = Axis(f[1,1]) 
ax2 = Axis(f[1,2]) 
lines!(ax1, T, -depth./1e3, color=:black)
lines!(ax2, log10.(η), -depth./1e3, color=:black)

# for i in 1:6
#     η, = viscosity_profile(x[i]; ε=1e-15)
#     lines!(ax2, log10.(η), -depth./1e3, label=join(x[i].Name))
# end
# scatter!(ax2, log10.(η), -depth./1e3, color=:black)
# lines!(ax2, log10.(ηdif), -depth./1e3, color=:red)
# lines!(ax2, log10.(ηcust), -depth./1e3, color=:red)
ax1.xlabel = "T (K)"
ax1.ylabel = "deepth (km)"
# axislegend(ax2, position=:lb)
f


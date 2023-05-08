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

function init_P!(P, depth)
   
    for i in eachindex(P)
        P[i] = - depth[i] * 3.3e3 * 9.81
    end

    return nothing
end

function compute_viscosity!(η, args, rheology, ε)
    for i in eachindex(η)
        phase = if args.depth[i] < 17.5e3
            rheology[1]
        elseif 35e3 > args.depth[i] > 17.5e3
            rheology[2]
        elseif 120e3 > args.depth[i] > 35e3
            rheology[3]
        elseif args.depth[i] > 120e3
            rheology[4]
        end

        elements =  x[1].CompositeRheology[1].elements

        ηeff = 0.0
        for el_i in elements
            (; A, n, E, V,  r, R) = el_i
            ηi = 0.5 * A.val^(-1/n.val) * ε^((1-n.val)/n.val) * exp((E.val + args.P[i]*V.val)/(n.val*R.val*args.T[i]))

            ηeff += inv(ηi) 
        end
        η[i] = clamp(1/ηeff, 1e18, 1e26)

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
    args = (; T=T, P=P)
    for _ in 1:5
        init_P!(P, depth)
    end

    η    = zeros(n)
    dt   = 100e3 * 3600 * 24 *365
    dt   = Inf
    args = (; T=T, P=P, depth=depth, dt=dt)
    compute_viscosity!(η, args, rheology, ε)
    
    return η, T, depth
end


x=init_rheologies()

η, T, depth = viscosity_profile(x; ε=1e-15)

f = Figure()
ax1 = Axis(f[1,1]) 
ax2 = Axis(f[1,2]) 
lines!(ax1, T, -depth./1e3, color=:black)
lines!(ax2, log10.(η), -depth./1e3)
# for xi in x
#     η, = viscosity_profile(xi; ε=1e-20)
#     lines!(ax2, log10.(η), -depth./1e3, label=join(xi.Name))
# end
# scatter!(ax2, log10.(η), -depth./1e3, color=:black)
# lines!(ax2, log10.(ηdif), -depth./1e3, color=:red)
# lines!(ax2, log10.(ηcust), -depth./1e3, color=:red)
ax1.xlabel = "T (K)"
ax1.ylabel = "deepth (km)"
# axislegend(ax2, position=:lb)
f
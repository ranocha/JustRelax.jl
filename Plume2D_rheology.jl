# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies()
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
            HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
            Conductivity      = ConstantConductivity(; k=2.7),
            CompositeRheology = CompositeRheology((disl_upper_crust, el, pl)),
            Elasticity        = el,
            Gravity           = ConstantGravity(; g=-9.81),
        ),
        SetMaterialParams(;
            Name              = "LowerCrust",
            Phase             = 2,
            Density           = PT_Density(; ρ0=2.9e3, β=β, T0=0.0, α = 2.5e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=7.5e2),
            Conductivity      = ConstantConductivity(; k=2.7),
            CompositeRheology = CompositeRheology((disl_lower_crust, el, pl)),
            Elasticity        = el,
            Gravity           = ConstantGravity(; g=-9.81),
        ),
        SetMaterialParams(;
            Name              = "LithosphericMantle",
            Phase             = 3,
            Density           = PT_Density(; ρ0=3e3, β=β, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=3.0),
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle, diff_lithospheric_mantle, el, )),
            Elasticity        = el,
            Gravity           = ConstantGravity(; g=-9.81),
        ),
        SetMaterialParams(;
            Name              = "SubLithosphericMantle",
            Phase             = 4,
            Density           = PT_Density(; ρ0=3e3, β=β, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=3.3),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el, )),
            Elasticity        = el,
            Gravity           = ConstantGravity(; g=-9.81),
        ),
        SetMaterialParams(;
            Name              = "Plume",
            Phase             = 5,
            Density           = PT_Density(; ρ0=3e3, β=β, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=3.3),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el, )),
            Elasticity        = el,
            Gravity           = ConstantGravity(; g=-9.81),
        ),
        SetMaterialParams(;
            Name              = "WeakZone",
            Phase             = 6,
            Density           = PT_Density(; ρ0=3e3, β=β, T0=0.0, α = 3e-5),
            HeatCapacity      = ConstantHeatCapacity(; cp=1.25e3),
            Conductivity      = ConstantConductivity(; k=3.0),
            CompositeRheology = CompositeRheology((disl_sublithospheric_mantle, diff_sublithospheric_mantle, el, )),
            Elasticity        = el,
            Gravity           = ConstantGravity(; g=-9.81),
        ),
    )
end

function init_phases!(phases, particles::Particles; r=50e3)
    ni = size(phases, 2), size(phases, 3)

    @parallel_indices (i, j) function init_phases!(phases, px, py, index, r)
        @inbounds for ip in axes(phases,1)
            # quick escape
            index[ip, i, j] == 0 && continue

            x = px[ip, i, j]
            depth = -py[ip, i, j]        
            if depth ≤ 17.5e3
                phases[ip, i, j] = 1

            elseif 35e3 ≥ depth > 17.5e3
                phases[ip, i, j] = 2

            elseif 120e3 ≥ depth > 35e3
                phases[ip, i, j] = 3

                 # weak zone
                if abs(x - 400e3 ) ≤ r
                    phases[ip, i, j] = 6
                end

            elseif depth > 120e3
                phases[ip, i, j] = 4
            end

            # plume
            if (((x - 400e3 ))^2 + ((depth - 400e3))^2) ≤ r^2
                phases[ip, i, j] = 5
            end
        end
        return nothing
    end

    @parallel (@idx ni) init_phases!(pPhases, particles.coords[1], particles.coords[2], particles.index, r)
end

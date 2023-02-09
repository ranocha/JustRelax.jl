using JustRelax
# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

using Printf, LinearAlgebra, GeoParams, GLMakie, SpecialFunctions

@parallel function update_buoyancy!(fz, T, ρ0gα)
    @all(fz) = ρ0gα .* @all(T)
    return nothing
end

@inline function tupleindex(args::NamedTuple, I::Vararg{Int, N}) where N
    k = keys(args)
    v = getindex.(values(args), I...)
    return (; zip(k, v)...)
end

@parallel_indices (i, j) function computeViscosity!(η, v, args)

    @inline av(T) = 0.25* (T[i,j] + T[i+1,j] + T[i,j+1] + T[i+1,j+1])

    @inbounds η[i, j] = computeViscosity_εII(v, 1.0, (; T = av(args.T)))
    # @inbounds η[i, j] = computeViscosity_εII(v, 1.0, (; T = args.T[i, j]))

    return nothing
end

@parallel_indices (i, j) function init_T!(T, z, time, k, Tm, Tp, Tmin, Tmax)
    dTdz = Tm-Tp
    @inbounds zᵢ = z[j]
    Tᵢ = Tp + dTdz*(1-zᵢ)
    time       = 5e-4
    Ths = Tmin + (Tm -Tmin) * erf((1-zᵢ)*0.5/(k*time)^0.5)
    Tᵢ = min(Tᵢ, Ths)
    time       = 1e-3
    Ths = Tmax - (Tmax + Tm) * erf(zᵢ*0.5/(k*time*5)^0.5)
    @inbounds T[i, j] = max(Tᵢ, Ths)
    return 
end

function thermal_convection2D(; ar=8, ny=16, nx=ny*8)

    # This tests the MaterialParameters structure
    CharUnits = GEO_units(; viscosity=1e23, length=2900km, temperature=1000K)

    # Define a struct for a first phase
    rheology = SetMaterialParams(;
        Name="Rock",
        Phase=1,
        # Density=PT_Density(; ρ0=3000kg/m^3),
        Density=ConstantDensity(; ρ=4000kg/m^3),
        HeatCapacity=ConstantHeatCapacity(; cp=1250J/kg/K),
        Conductivity=ConstantConductivity(; k=5.0Watt/K/m),
        CreepLaws=LinearViscous(; η=1e23Pa*s),
        CharDim=CharUnits,
    )

    # Physical domain
    ly       = 2900km
    lx       = ly * ar
    lx_nd    = nondimensionalize(lx, CharUnits)
    ly_nd    = nondimensionalize(ly, CharUnits)
    Xo       = (0.0, 0.0)             # Origin coordinates
    ni       = (nx, ny)
    li       = (lx_nd, ly_nd)                # domain length in x- and y-
    di       = @. li / (ni)                  # grid step in x- and -y
    xci, xvi = lazy_grid(di, li; origin=Xo)  # nodes at the center and vertices of the cells

    # Physical parameters
    η0            = rheology.CreepLaws[1].η.val                                 # viscosity, Pa*s
    κ             = NumValue(
        rheology.Conductivity[1].k /
        (rheology.Density[1].ρ * rheology.HeatCapacity[1].cp),
    )                          # heat diffusivity, m^2/s
    ΔT            = nondimensionalize(1000K, CharUnits)                                 # initial temperature perturbation K
    ρ0            = rheology.Density[1].ρ.val
    g             = rheology.Gravity[1].g.val
    # α = rheology.Density[1].α.val
    α             = 0.03
    Cp0           = rheology.HeatCapacity[1].cp.val
    # Ra            = ρ0 * g * α * ΔT * ly_nd^3 / (η0* κ)
    Ra            = 1e6
    dt  = dt_diff = 0.5 / 4.1 * min(di...)^2 / κ      # diffusive CFL timestep limiter
    println("\n Ra-number is $Ra")
    
    # Thermal diffusion ----------------------------------
    # ρ   = @fill(ρ0, ni...)
    ρCp        = @fill(ρ0 * Cp0, ni...) 
    k          = @fill(rheology.Conductivity[1].k.val, ni...)
    thermal    = ThermalArrays(ni)

    # Stokes ---------------------------------------------
    ## Allocate arrays needed for every Stokes problem
    stokes    = StokesArrays(ni, ViscoElastic)
    pt_stokes = PTStokesCoeffs(ni, di; ϵ=1e-4,  CFL=1 / √2.1)

    ## Setup-specific parameters and fields -------------
    # Initial geotherm 
    time       = 1e-4
    k          = 1
    Tm         = 1900/2300
    Tp         = 1600/2300
    Tmin, Tmax = 0.12, 1.0
    @parallel init_T!(thermal.T, xvi[2], time, k, Tm, Tp, Tmin, Tmax)
    # Elliptical temperature anomaly 
    xc, yc                     = 0.5*lx_nd, 0.25*ly_nd
    ri                         = 0.1/2
    Elli                       = [ ((x-xc ))^2 + ((y - yc))^2 for x in xvi[1], y in xvi[2]]
    thermal.T[Elli .< ri.^2] .*= 1.1
    # thermal.T .*= 1 .+ rand(ni...).*0.05
    clamp!(thermal.T, Tmin, Tmax)
    @parallel assign!(thermal.Told, thermal.T)
    # Rheology 
    η = @ones(ni...)
    v = CompositeRheology( (ArrheniusType(),) )
    args_η = (;T=thermal.T)
    @parallel (1:nx, 1:ny) computeViscosity!(η, v, args_η) # init viscosity field
    dt_elasticity = Inf
    G = @fill(1.0, ni...) 
    Kb = @fill(Inf, ni...) 
    fy = -Ra .* thermal.T
    ρg = @zeros(ni...), fy

    ## Boundary conditions
    flow_bcs = FlowBoundaryConditions(; 
        free_slip = (left=false, right=false, top=true, bot=true), 
        periodicity = (left=true, right=true, top=false, bot=false)
    )
    thermal_bc = (flux_x=true, flux_y=false)

    # Physical time loop
    t   =  0.0
    it  =  0
    nt  =  300

    args = (;)
    local iters
    while it < nt

        # Stokes solver ----------------
        iters = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            η,
            G,
            Kb,
            dt,
            iterMax=150e3,
            nout=1e3,
        )

        dt = JustRelax.compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Thermal solver ---------------
        solve!(
            thermal,
            thermal_bc,
            stokes,
            rheology,
            args,
            di,
            dt
        )
        # ------------------------------

        # Update buoyancy and viscosity -
        @parallel (1:nx, 1:ny) computeViscosity!(η, v, args_η) # update viscosity field
        @parallel update_buoyancy!(ρg[2], thermal.T, -Ra)
        # ------------------------------

        @show it += 1
        t += dt

        if it == 1 || rem(it, 25) == 0
            fig = Figure(resolution = (900, 1600), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "T")
            ax2 = Axis(fig[2,1], aspect = ar, title = "Vy")
            ax3 = Axis(fig[3,1], aspect = ar, title = "τII")
            ax4 = Axis(fig[4,1], aspect = ar, title = "η")
            h1 = heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T) , colormap=:batlow)
            h2 = heatmap!(ax2, xci[1], xvi[2], Array(stokes.V.Vy[2:end-1,:]) , colormap=:batlow)
            # h3 = heatmap!(ax3, xci[1], xci[2], Array(stokes.P) , colormap=:romaO) #, colorrange = (0.0, σy))
            h3 = heatmap!(ax3, xvi[1], xci[2], Array(stokes.V.Vx[:,2:end-1]), colormap=:batlow) #, colorrange = (0.0, σy))
            h4 = heatmap!(ax4, xci[1], xci[2], Array(log10.(η)) , colormap=:batlow)
            Colorbar(fig[1,2], h1)
            Colorbar(fig[2,2], h2)
            Colorbar(fig[3,2], h3)
            Colorbar(fig[4,2], h4)
            fig
            save("figs2d/$(it).png", fig)
        end

    end

    return (ni=ni, xci=xci, li=li, di=di), thermal
end

ar = 8
n  = 16
nx = n*ar - 2
ny = n - 2

# thermal_convection2D(;ar=ar,nx=nx, ny=ny);

# X = [x for x in xvi[1], y in xvi[2], z in xvi[3]][:]
# Y = [y for x in xvi[1], y in xvi[2], z in xvi[3]][:]
# Z = [z for x in xvi[1], y in xvi[2], z in xvi[3]][:]
# scatter(thermal.T[:], Z)

V = @tuple stokes.V
τ_o = @tuple stokes.τ_o
_di = inv.(di)
@btime @parallel $(1:nx, 1:ny) JustRelax.Elasticity2D.rotate_stress!($V, $τ_o, $_di, $dt)

@parallel (1:nx, 1:ny) JustRelax.Elasticity2D.rotate_stress!(@tuple(stokes.V), @tuple(stokes.τ_o), _di, dt)

_tuple(V::Velocity{<:AbstractArray{T, 2}}) where T = V.Vx, V.Vy
_tuple(V::Velocity{<:AbstractArray{T, 3}}) where T = V.Vx, V.Vy, V.Vz
_tuple(A::SymmetricTensor{<:AbstractArray{T, 2}}) where T = A.xx, A.yy, A.xy_c
_tuple(A::SymmetricTensor{<:AbstractArray{T, 3}}) where T = A.xx, A.yy, A.zz, A.yz_c, A.xz_c, A.xy_c

macro tuple(A) 
    return quote _tuple($(esc(A))) end
end

# @code_warntype _tuple1(stokes.V)
@btime @tuple $stokes.V
@btime $stokes.V.Vx[$1]


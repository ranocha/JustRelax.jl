# include benchmark related functions
using JustRelax
using Printf, LinearAlgebra, GLMakie
using GeoParams

# setup ParallelStencil.jl environment
model = PS_Setup(:cpu, Float64, 2)
environment!(model)

function _viscosity!(η, xci, yci, rc, ηi, cx, cy)
    for i in 1:length(xci), j in 1:length(yci)
        if rc < sqrt((xci[i] - cx)^2 + (yci[j] - cy)^2)
            η[i, j] = ηi
        end
    end
end

function solvi_viscosity(ni, di, li, rc, η0, ηi)
    dx, dy = di
    lx, ly = li
    # cx, cy = lx/2, ly/2
    η = @fill(η0, ni...)
    # _viscosity!(η, xci, yci, rc, ηi, cx, cy)
    phase = Int.(@fill(1, ni...))
    Rad2 = [
        sqrt.(
            ((ix - 1) * dx + 0.5 * dx - 0.5 * lx)^2 +
            ((iy - 1) * dy + 0.5 * dy - 0.5 * ly)^2,
        ) for ix in 1:ni[1], iy in 1:ni[2]
    ]
    # Rad2 = [
    #     sqrt.(
    #         ((ix - 1) * dx + 0.5 * dx)^2 +
    #         ((iy - 1) * dy + 0.5 * dy)^2,
    #     ) for ix in 1:ni[1], iy in 1:ni[2]
    # ]
    η[Rad2 .< rc] .= ηi
    phase[Rad2 .< rc] .= 2
    return η, phase
end

# @parallel function correct_shear_stress!(
#     τxy::AbstractArray{T,2},
#     τxy_o::AbstractArray{T,2},
#     Gdτ::AbstractArray{T,2},
#     εxy::AbstractArray{T,2},
#     ηpt::AbstractArray{T,2},
#     G::AbstractArray{T,2},
#     dt::T
# ) where T
#     @all(τxy) = 
#         2.0 * (@all(εxy) + @all(τxy_o) * 0.5 * inv(@av(G) * dt) + @all(τxy) * 0.5 * inv(@av(Gdτ))) * @av(ηpt) 
#     return nothing
# end

@parallel function correct_shear_stress!(
    τxy::AbstractArray{T,2},
    τxy_o::AbstractArray{T,2},
    Gdτ::AbstractArray{T,2},
    εxy::AbstractArray{T,2},
    ηpt::AbstractArray{T,2},
    G::AbstractArray{T,2},
    dt::T
) where T
    @all(τxy) = 
        (2.0 * @all(εxy) + @all(τxy_o) * inv(@av(G) * dt) + @all(τxy) * inv(@av(Gdτ))) * @harm(ηpt) 
    return nothing
end

# ii = argmax(τxy)

# 2.0 * ((εxy[ii]) + (τxy_o[ii]) * 0.5 * inv((G)[ii] * dt) + (τxy[ii]) * 0.5 * inv((Gdτ[ii]))) * (ηpt[ii])

# ((τxy[ii]) + (τxy_o[ii]) * ((Gdτ[ii]) / ((G[ii]) * dt)) + 2.0 * (Gdτ[ii]) * (εxy[ii])) /
#         (1.0 + (Gdτ[ii]) / (η[ii]) + ((Gdτ[ii]) / ((G[ii]) * dt)))


function viscoelastic_viscosity(MatParam, G, phase, dt, I::Vararg{Int64,N}) where {N}
    return 1.0 / (
        1.0 / G[I...] * dt +
        1.0 / 2 / computeViscosity_τII(MatParam, 0.0, phase[I...], (;); cutoff=(-Inf, Inf))
    )
end

@parallel_indices (i, j) function viscoelastic_viscosity!(ηve, MatParam, G, phase, dt)
    ηve[i, j] = inv(
        inv(G[i, j] * dt) +
        inv(0.5*computeViscosity_τII(MatParam, 0.0, phase[i, j], (;); cutoff=(-Inf, Inf))),
    )
    return nothing
end

# WARNING: GEOPARAMS GIVES 2*η
@parallel_indices (i, j) function pt_viscosity!(ηpt, MatParam, G, Gdτ, phase, dt)
    ηpt[i, j] = inv(
        inv(G[i, j] * dt) +
        inv(0.5*computeViscosity_τII(MatParam, 0.0, phase[i, j], (;); cutoff=(-Inf, Inf))) +
        inv(Gdτ[i, j])
    )
    return nothing
end

@parallel_indices (i, j) function second_invariant!(CII, Cxx, Cyy, Cxy)
    normal_component(i, j) = @inbounds (Cxx[i, j], Cyy[i, j])
    function shear_component(i, j)
        @inbounds (Cxy[i - 1, j - 1], τxy[i, j], Cxy[i - 1, j], Cxy[i, j - 1])
    end

    CII[i, j] = second_invariant_staggered(normal_component(i, j), shear_component(i, j))

    return nothing
end

@parallel function center2vertex_inn!(Av, Ac)
    @inn(Av) = @av(Ac)
    return nothing
end

@parallel_indices (i,j) function c2v!(Av, Ac, nx, ny)
    # inner part
    if (1 < i < nx) && (1 < j < ny)
        Av[i,j] = 0.25 * (Ac[i-1,j-1] + Ac[i-1,j] + Ac[i,j-1] + Ac[i,j])

    # corners 
    elseif i == j == 1
        Av[i,j] = Ac[1, 1]

    elseif i == 1 && j == ny
        Av[i,j] = Ac[1, ny-1]

    elseif i == nx && j == 1
        Av[i,j] = Ac[nx-1, 1]

    elseif i == nx && j == ny
        Av[i,j] = Ac[nx-1, ny-1]

    # side walls
    elseif i == 1 && 1 < j < ny
        Av[i,j] = 0.5 * (Ac[i, j-1] + Ac[i, j])

    elseif i == nx && 1 < j < ny
        Av[i,j] = 0.5 * (Ac[i-1, j-1] + Ac[i-1, j])

    elseif 1 < i < ny && j == 1
        Av[i,j] = 0.5 * (Ac[i-1, j] + Ac[i, j])

    elseif 1 < i < ny && j == ny
        Av[i,j] = 0.5 * (Ac[i-1, j-1] + Ac[i, j-1])
    end    

    return nothing
end

function center2vertex!(Av, Ac)
    freeslip = (freeslip_x=true, freeslip_y=true)
    @parallel center2vertex_inn!(Av, Ac)
    return apply_free_slip!(freeslip, Av, Av)
end

@parallel function _vertex2center!(Ac, Av)
    @all(Ac) = @av(Av)
    return nothing
end

# @parallel function _vertex2center!(Ac, Av)
#     @all(Ac) = @harm(Av)
#     return nothing
# end

@inline vertex2center!(Ac, Av) = @parallel _vertex2center!(Ac, Av)

@parallel_indices (i, j) function viscoplastic_viscosity!(ηvp, η, λ, τII, εII)
    if λ[i, j] == 0.0
        ηvp[i, j] = η[i, j]
    else
        ηvp[i, j] = 0.5 * τII[i, j] / εII[i, j]
    end
    return nothing
end

@parallel_indices (i,j) function plasticity!(
    ηvp, τxx, τyy, τxy, λ, εxx, εyy, εxy, τxx_o, τyy_o, τxy_o, ηve, ηpt, P, phase, MatParam, G, Gdτ, dt, ηrg, C
)

    τxx[i, j], τyy[i, j], τxy[i, j], ηvp[i, j], λ[i,j]= pt_plasticity_kernel(
        τxx[i, j],
        τyy[i, j],
        τxy[i, j],
        εxx[i, j],
        εyy[i, j],
        εxy[i, j],
        τxx_o[i, j],
        τyy_o[i, j],
        τxy_o[i, j],
        λ[i,j],
        G[i,j],
        Gdτ[i,j],
        ηve[i, j],
        ηpt[i, j],
        P[i, j],
        phase[i,j],
        MatParam,
        dt,
        ηrg,
        C[i,j]
    )

    return nothing
end

function pt_plasticity_kernel(τxx, τyy, τxy, εxx, εyy, εxy, τxx_o, τyy_o, τxy_o, λ, G, Gdτ, ηve, ηpt, P, phase, MatParam, dt, ηrg, C)
    # trial strain
    εxx_trial = εxx + τxx_o * 0.5 * inv(G * dt) + τxx * 0.5 * inv(Gdτ)
    εyy_trial = εyy + τyy_o * 0.5 * inv(G * dt) + τyy * 0.5 * inv(Gdτ)
    εxy_trial = εxy + τxy_o * 0.5 * inv(G * dt) + τxy * 0.5 * inv(Gdτ)
    εII_trial = second_invariant(εxx_trial, εyy_trial, εxy_trial)
    # trial stress
    τxx = 2.0 * εxx_trial * ηpt
    τyy = 2.0 * εyy_trial * ηpt
    τxy = 2.0 * εxy_trial * ηpt
    τII = second_invariant(τxx, τyy, τxy)
    # yield function
    # F = compute_yieldfunction(MatParam, phase, (P=P, τII=τII)) - λ * ηrg
    F = τII - P * sind(30) - C * cosd(30) - λ * ηrg
    λ = 0.0
    # ηvep = ηve
    ηvep = ηpt
    if F > 0.0
        λ = F / (ηpt + ηrg/dt)
        ∂Q∂τxx, ∂Q∂τyy, ∂Q∂τxy = compute_plasticpotentialDerivative(
            MatParam, phase, (τxx, τyy, τxy)
        )
        # stress correction
        τxx = 2.0 * ηpt * (εxx_trial - λ * ∂Q∂τxx)
        τyy = 2.0 * ηpt * (εyy_trial - λ * ∂Q∂τyy)
        τxy = 2.0 * ηpt * (εxy_trial - 0.5 * λ * ∂Q∂τxy)
        ηvep = 0.5 * τII / εII_trial
    end
    # τII = second_invariant(τxx, τyy, τxy)
    return τxx, τyy, τxy, ηvep, λ
end

function plastic_strain(εvp::Number, τxx::Number, τyy::Number, τxy::Number, λ::Number, MatParam, phase, dt)
    τII = second_invariant(τxx, τyy, τxy)
    ∂Q∂τxx, ∂Q∂τyy, ∂Q∂τxy = compute_plasticpotentialDerivative(
        MatParam, phase, (τxx, τyy, τxy)
    )
    h = (2.0/(3.0*τII))*(∂Q∂τxx^2 + ∂Q∂τyy^2 + ∂Q∂τxy^2) * λ^2 # (λ*∂Q∂τᵀ) ⋅ (λ*∂Q∂τ)
    εvp += √h * dt
    return εvp
end

@parallel_indices (i, j) function plastic_strain!(εvp::AbstractArray{T, 2}, τxx::AbstractArray{T, 2}, τyy::AbstractArray{T, 2}, τxy::AbstractArray{T, 2}, λ::AbstractArray{T, 2}, MatParam, phase, dt) where T
    εvp[i,j] = plastic_strain(εvp[i,j], τxx[i,j], τyy[i,j], τxy[i,j], λ[i,j], MatParam, phase[i,j], dt)
    return 
end

# @parallel function softening!(C, εvp, Δ, μ, σ)
#     @all(C) = @all(C) - 0.5 * Δ * erf((μ - @all(εvp)) / σ) 
# end

@parallel function softening!(C::AbstractArray, C0::AbstractArray, λ::AbstractArray, h::AbstractArray, Cmin, dt)
    @all(C) = max(@all(C0) - dt * √(2.0/3.0) * @all(λ) * @all(h), Cmin * 0.5)
    return nothing
end

@parallel_indices (i, j) function plastic_iter_params!(
    dτ_Rho::AbstractArray,
    Gdτ::AbstractArray,
    ητ::AbstractArray,
    ηvp::AbstractArray,
    λ::AbstractArray,
    Vpdτ::T,
    G::AbstractArray,
    dt::M,
    Re::T,
    r::T,
    max_li::T,

) where {T,M}

    if λ[i,j] == 0.0 
        dτ_Rho[i,j] =
            Vpdτ * max_li / (Re * (one(T) / (one(T) / ητ[i,j] + one(T) / (G[i,j] * dt))))
    else
        dτ_Rho[i,j] =
            # Vpdτ * max_li / (Re * (one(T) / (one(T) / ηvp[i,j] + one(T) / (G[i,j] * dt))))
            Vpdτ * max_li / (Re * ηvp[i,j])
    end
    Gdτ[i,j] = Vpdτ^2 / (dτ_Rho[i,j] * (r + T(2.0)))
    return nothing
end

Δη = 1e-3
nx = 51
ny = 51
lx = 4e3
ly = 2e3
rc = 1e2
εbg = 1e0
Δε = 5e-5
ηrg = 1e-2

function solVi(ηrg; Δη=1e-3, nx=256 - 1, ny=256 - 1, lx=1e1, ly=1e1, rc=1e0, εbg=1e0)
    CharDim = GEO_units(; length=10km, viscosity=1e20Pa * s)        # Characteristic dimensions
    η0 = 1e21  # matrix viscosity
    ηi = 1e20  # matrix viscosity
    MatParam = (
        SetMaterialParams(;
            Name="Matrix",
            Phase=1,
            Density=PT_Density(; ρ0=3000kg / m^3, β=0.0 / Pa),
            CreepLaws=LinearViscous(; η=η0*Pa * s),
            Plasticity=DruckerPrager(; C=10e6Pa, ϕ=30NoUnits),
            # Plasticity=DruckerPrager(; C=(1.6/cosd(30))NoUnits),
            # CharDim=CharDim,
        ),
        SetMaterialParams(;
            Name="Inclusion",
            Phase=2,
            Density=PT_Density(; ρ0=3000kg / m^3, β=0.0 / Pa),
            CreepLaws=LinearViscous(; η=ηi*Pa * s),
            Plasticity=DruckerPrager(;  C=10e6Pa, ϕ=30NoUnits),
            # Plasticity=DruckerPrager(; C=(1.6/cosd(30))NoUnits),
            # CharDim=CharDim,
        ),
    )

    ## Spatial domain: This object represents a rectangular domain decomposed into a Cartesian product of cells
    # Here, we only explicitly store local sizes, but for some applications
    # concerned with strong scaling, it might make more sense to define global sizes,
    # independent of (MPI) parallelization
    ni = (nx, ny) # number of nodes in x- and y-
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / (ni) # grid step in x- and -y
    max_li = max(li...)
    nDim = length(ni) # domain dimension
    origin = (-lx/2, -ly)
    origin = (0.0,0.0)
    xci = Tuple([((di[i] / 2):di[i]:(li[i] - di[i] / 2)) .+ origin[i] for i in 1:nDim]) # nodes at the center of the cells
    xvi = Tuple([(0:di[i]:li[i]) .+ origin[i] for i in 1:nDim]) # nodes at the vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1   # physical time step

    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di; CFL = 0.75 * 0.9/sqrt(2))

    ## Setup-specific parameters and fields
    ξ = 4.0         # Maxwell relaxation time
    G = @fill(1e10, ni...)         # elastic shear modulus

    # ηi = η0 # inclusion viscosity
    η, phase = solvi_viscosity(ni, di, li, rc, η0, ηi) # viscosity field
    ηpt = @zeros(ni...)
    G[phase .== 2] .*= 2.5e9
    dt = η0 / (maximum(G) * ξ) 
    dt = 1e10

    ηvp = @fill(0.0, ni...) # viscoplastic viscosity
    ηve = @zeros(ni...) # viscoelastic viscosity
    @parallel (1:ni[1], 1:ni[2]) viscoelastic_viscosity!(ηve, MatParam, G, phase, dt)

    ## Plasticity 
    Ftrial = @zeros(ni...)
    λ = @zeros(ni...)
    τII = @ones(ni...)
    εII = @ones(ni...)

    ## Boundary conditions
    εbg = Δε/dt
    # εbg = 1e-14
    pureshear_bc!(stokes, di, li, εbg)
    # stokes.V.Vx .= [-εbg * x for x in xvi[1], y in xci[2]]
    # stokes.V.Vy .= [εbg * y for x in xci[1], y in xvi[2]]
    freeslip = (freeslip_x=true, freeslip_y=true)

    Ci = @fill(10e6, ni...)
    C0 = @fill(10e6, ni...)
    Cmin = 5e6
    h = @fill(2e1, ni...)

    ###
    # unpack
    _dt = inv(dt)
    dx, dy = di
    _dx, _dy = inv.(di)
    lx, ly = li
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    εxx, εyy, εxy = JustRelax.strain(stokes)
    τ, τ_o = JustRelax.stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry = stokes.R.Rx, stokes.R.Ry
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ,
    pt_stokes.dτ_Rho, pt_stokes.ϵ, pt_stokes.Re, pt_stokes.r,
    pt_stokes.Vpdτ
    _sqrt_leng_Rx = inv(√(length(Rx)))
    _sqrt_leng_Ry = inv(√(length(Ry)))
    _sqrt_leng_∇V = inv(√(length(∇V)))
    ###
    εxyv = @zeros(ni...)
    τxyv = @zeros(ni...)
    τxy_trialv = @zeros(ni...)
    τxy_ov = @zeros(ni...)
    εxy_effv = @zeros(ni...)

    # Physical time loop
    t = 0.0
    ρ = @zeros(ni...)

    iterMax = 500e3
    ϵ = 1e-6
    nout = 100
 
    evo_Txx = Float64[]
    ηvp .= ηve
    Gc = @zeros(ni.-1...)
    Gdτc = @zeros(ni.-1...)
    ηvp_c = @zeros(ni.-1...)
    vertex2center!(Gc, G)
    η_ec = @. Gc * dt
    ητ = similar(ηvp)
    idxs = (1:nx, 1:ny)
    t = 0.0
    evo_t = Float64[]
    ητ = deepcopy(η)
    ητ_vp = deepcopy(ηvp)

    for _ in 1:30
        JustRelax.Elasticity2D.update_τ_o!(stokes)
        @parallel idxs c2v!(τxy_ov, τxy_o, ni...)
        @parallel idxs pt_viscosity!(ηpt, MatParam, G, Gdτ, phase, dt)

        # ~preconditioner
        @parallel JustRelax.compute_maxloc!(ητ, η)
        @parallel JustRelax.compute_maxloc!(ητ_vp, ηvp)
        # PT numerical coefficients
        # @parallel JustRelax.Elasticity2D.elastic_iter_params!(
        #     dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li
        # )
        @parallel idxs plastic_iter_params!(
            dτ_Rho, Gdτ, ητ, ητ_vp, λ, Vpdτ, G, dt, Re, r, max_li
        )
        # errors
        err = 2 * ϵ
        iter = 0
        cont = 0

        # solver loop
        iter = 1
        C0 .= Ci
        @show extrema(C0)
        while err > ϵ && iter ≤ iterMax

            #### PT PLASTIC ITERATIONS #########################################
            @parallel softening!(Ci, C0, λ, h, Cmin, dt)
            @parallel JustRelax.Stokes2D.compute_strain_rate!(εxx, εyy, εxy, Vx, Vy, _dx, _dy)
            @parallel idxs c2v!(εxyv, εxy, ni...)
            @parallel JustRelax.Stokes2D.compute_P!(∇V, P, εxx, εyy, Gdτ, r)
            @parallel idxs plasticity!(
                ηvp, τxx, τyy, τxyv, λ, εxx, εyy, εxyv, τxx_o, τyy_o, τxy_ov, ηve, ηpt, P, phase, MatParam, G, Gdτ, dt, ηrg, Ci
            )
            # vertex2center!(ηvpc, ηvp)
            @parallel correct_shear_stress!(τxy, τxy_o, Gdτ, εxy, ηvp, G, dt)

            @parallel JustRelax.compute_maxloc!(ητ_vp, ηvp)
            @parallel idxs plastic_iter_params!(
                dτ_Rho, Gdτ, ητ, ητ_vp, λ, Vpdτ, G, dt, Re, r, max_li
            )
            @parallel JustRelax.Elasticity2D.compute_dV_elastic!(
                dVx, dVy, P, Rx, Ry, τxx, τyy, τxy, dτ_Rho, ρ, _dx, _dy
            )
            @parallel JustRelax.Stokes2D.compute_V!(Vx, Vy, dVx, dVy)
            ####################################################################

            # #### NORMAL PT NON-PLASTIC ITERATIONS ##############################
            # @parallel JustRelax.Stokes2D.compute_strain_rate!(εxx, εyy, εxy, Vx, Vy, _dx, _dy)
            # @parallel JustRelax.Stokes2D.compute_P!(∇V, P, εxx, εyy, Gdτ, r)
            # @parallel JustRelax.Elasticity2D.compute_τ!(
            #     τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Gdτ, εxx, εyy, εxy, η, G, dt
            # )
            # @parallel JustRelax.Elasticity2D.compute_dV_elastic!(
            #     dVx, dVy, P, Rx, Ry, τxx, τyy, τxy, dτ_Rho, ρ, _dx, _dy
            # )
            # @parallel JustRelax.Stokes2D.compute_V!(Vx, Vy, dVx, dVy)
            # ####################################################################

            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy)
            # @parallel JustRelax.compute_maxloc!(ητ, ηvp)
            # # PT numerical coefficients
            @parallel JustRelax.Elasticity2D.elastic_iter_params!(
                dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li
            )
                
            heatmap(xvi[1],xvi[2], τxx, colormap=:batlow)
                
            iter += 1
            if iter % nout == 0 && iter > 1
                cont += 1
                Vmin, Vmax = minimum(Vx), maximum(Vx)
                Pmin, Pmax = minimum(P), maximum(P)
                norm_Rx = norm(Rx) / (Pmax - Pmin) * lx * _sqrt_leng_Rx
                norm_Ry = norm(Ry) / (Pmax - Pmin) * lx * _sqrt_leng_Ry
                norm_∇V = norm(∇V) / (Vmax - Vmin) * lx * _sqrt_leng_∇V
                err = maximum((norm_Rx, norm_Ry, norm_∇V))
                if (err < ϵ) || (iter == iterMax)
                    @printf(
                        "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                        iter,
                        err,
                        norm_Rx,
                        norm_Ry,
                        norm_∇V,
                    )
                end
            end
            # @parallel JustRelax.compute_maxloc!(ητ, ηve)
            # apply_free_slip!(freeslip, ητ, ητ)
            # # PT numerical coefficients
            # @parallel JustRelax.Elasticity2D.elastic_iter_params!(
            #     dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li
            # )
            iter += 1

            # h=heatmap!(ax1, xvi[1],xvi[2], εII, colormap=:inferno)
            # scatter!(ax2, (t, maximum(τxx)), color=:black)
            # f
        end

        t += dt
        τII = @. sqrt(0.5*(τxx^2 + τyy^2) + τxyv^2)
        max_Txx = maximum(τII)
        # max_Txx = maximum(abs.(τxx))
        push!(evo_Txx, max_Txx)
        push!(evo_t, t)
        println("Maximum τxx = $(max_Txx*1e-6) MPa")
        @show maximum(λ)
        maximum(λ) != 0.0 && break
    end

    εII = @. sqrt(0.5*(εxx^2 + εyy^2) + εxyv^2)
    # f, ax, h = heatmap(xvi[1], xvi[2], τxx; colormap=:batlow)
    # Colorbar(f[1, 2], h)


    # f

    return evo_Txx, evo_t, (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, εII
        
    # return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end

Δη = 1e-3
nx = 62 * 2
ny = 31 * 2
lx = 4e3
ly = 2e3
rc = 1e2
εbg = 1e0
Δε = 5e-5
ηrg = 1e19

evo_Txx, evo_t, geometry, stokes, εII = solVi(ηrg; Δη=1e-3, nx=nx, ny=ny, lx=lx, ly=ly, rc=rc, εbg=εbg);
# f

fig = Figure(resolution = (1200, 1200))
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[2,1])
h=heatmap!(ax1, geometry.xvi[1], geometry.xvi[2], log10.(εII); colormap=:batlow)
# h=heatmap!(ax1, geometry.xvi[1], geometry.xvi[2], stokes.τ.xx; colormap=:batlow)
Colorbar(fig[1,2], h)
lines!(ax2, evo_t, @.(abs(evo_Txx*1e-6)), color=:black )
ax2.xlabel = "iters"
ax2.ylabel = "τxx (MPa)"
fig

# fig = Figure(resolution = (1200, 1200))
# ax1 = Axis(fig[1,1])
# ax2 = Axis(fig[2,1])
# f,ax,h=heatmap(geometry.xvi[1], geometry.xvi[2], stokes.τ.xx; colormap=:batlow)
# ax2.ylabel = "τxx (MPa)"
# fig

# f,ax,h=heatmap(geometry.xvi[1], geometry.xvi[2], stokes.τ.xx; colormap=:batlow)
# # f,ax,h=heatmap(geometry.xvi[1], geometry.xvi[2], η; colormap=:batlow)
# Colorbar(f[1,2], h)
# f

# f,ax,h=heatmap(geometry.xvi[1], geometry.xvi[2], stokes.V.Vy; colormap=:batlow)
# Colorbar(f[1,2], h)
# f

# G0 = 1e10
# μ0 = 1e22
# dt = 1e10
# # evo_t = cumsum(dt.*(1:length(evo_Txx)))
# εbg = Δε/dt
# sol = @. 2.0*εbg*μ0*(1.0-exp(-evo_t*G0/μ0))
# lines!(ax2, evo_t, 1e-6.*sol, color=:red)
# fig


# dt = 0.3e9
# evo_t = cumsum(dt.*(1:20))
# εbg = Δε/dt
# sol = @. 2.0*εbg*μ0*(1.0-exp(-evo_t*G0/μ0))
# lines(evo_t, 1e-6.*sol)

## UTILS

function stress(stokes::StokesArrays{ViscoElastic,A,B,C,D,nDim}) where {A,B,C,D,nDim}
    return stress(stokes.τ), stress(stokes.τ_o)
end

## DIMENSION AGNOSTIC ELASTIC KERNELS

@parallel function elastic_iter_params!(
    dτ_Rho::AbstractArray,
    Gdτ::AbstractArray,
    ητ::AbstractArray,
    Vpdτ::T,
    G::T,
    dt::M,
    Re::T,
    r::T,
    max_li::T,
) where {T,M}
    @all(dτ_Rho) = Vpdτ * max_li / Re / (one(T) / (one(T) / @all(ητ) + one(T) / (G * dt)))
    @all(Gdτ) = Vpdτ^2 / @all(dτ_Rho) / (r + T(2.0))
    return nothing
end

@parallel function elastic_iter_params!(
    dτ_Rho::AbstractArray,
    Gdτ::AbstractArray,
    ητ::AbstractArray,
    Vpdτ::T,
    G::AbstractArray,
    dt::M,
    Re::T,
    r::T,
    max_li::T,
) where {T,M}
    @all(dτ_Rho) =
        Vpdτ * max_li / Re / (one(T) / (one(T) / @all(ητ) + one(T) / (@all(G) * dt)))
    @all(Gdτ) = Vpdτ^2 / @all(dτ_Rho) / (r + T(2.0))
    return nothing
end

## 2D ELASTICITY MODULE

module Elasticity2D

using ..JustRelax
using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using GeoParams, LinearAlgebra, Printf

import JustRelax: stress, strain, elastic_iter_params!, PTArray, Velocity, SymmetricTensor
import JustRelax: Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic
import JustRelax: compute_maxloc!, solve!, @tuple

import ..Stokes2D: compute_P!, compute_V!, compute_strain_rate!

export solve!

include("StressRotation.jl")

## 2D ELASTIC KERNELS

function update_τ_o!(stokes::StokesArrays{ViscoElastic,A,B,C,D,2}) where {A,B,C,D}
    # τ, τ_o = stress(stokes)
    τxx, τyy, τxy, τxy_c = stokes.τ.xx, stokes.τ.yy, stokes.τ.xy, stokes.τ.xy_c
    τxx_o, τyy_o, τxy_o, τxy_o_c = stokes.τ_o.xx, stokes.τ_o.yy, stokes.τ_o.xy, stokes.τ_o.xy_c
    @parallel  update_τ_o!(τxx_o, τyy_o, τxy_o, τxy_o_c, τxx, τyy, τxy, τxy_c)
    return nothing
end

@parallel function update_τ_o!(τxx_o, τyy_o, τxy_o, τxy_o_c, τxx, τyy, τxy, τxy_c)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τxy_o) = @all(τxy)
    @all(τxy_o_c) = @all(τxy_c)
    return nothing
end

@parallel function compute_∇V!(∇V, Vx, Vy, _dx, _dy)
    @all(∇V) = @d_xi(Vx) * _dx + @d_yi(Vy) * _dy
    return nothing
end

@parallel function compute_strain_rate!(εxx, εyy, εxy, ∇V, Vx, Vy, _dx, _dy)
    @all(εxx) = @d_xi(Vx) * _dx - @all(∇V) / 3.0
    @all(εyy) = @d_yi(Vy) * _dy - @all(∇V) / 3.0
    @all(εxy) = 0.5 * (@d_ya(Vx) * _dy + @d_xa(Vy) * _dx)
    return nothing
end

# Continuity equation

## Incompressible 
@parallel function compute_P!(P, RP, ∇V, η, r, θ_dτ)
    @all(RP) = -@all(∇V)
    @all(P) = @all(P) + @all(RP) * r / θ_dτ * @all(η)
    return nothing
end

## Compressible 
@parallel function compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (@all(K) * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (@all(K) * dt))
    return nothing
end

## Compressible - GeoParams
@parallel function compute_P!(P, P_old, RP, ∇V, η, K::Number, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (K * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (K * dt))
    return nothing
end

@parallel function compute_V!(Vx, Vy, P, τxx, τyy, τxyv, ηdτ, ρgx, ρgy, ητ, _dx, _dy)
    @inn(Vx) =
        @inn(Vx) +
        (-@d_xa(P) * _dx + @d_xa(τxx) * _dx + @d_yi(τxyv) * _dy - @av_xa(ρgx)) * ηdτ /
        @harm_xa(ητ)
    @inn(Vy) =
        @inn(Vy) +
        (-@d_ya(P) * _dy + @d_ya(τyy) * _dy + @d_xi(τxyv) * _dx - @av_ya(ρgy)) * ηdτ /
        @harm_ya(ητ)
    return
end

## Compressible - GeoParams
@parallel function compute_P!(P, P_old, RP, ∇V, η, K::Number, dt, r, θ_dτ)
    @all(RP) = -@all(∇V) - (@all(P) - @all(P_old)) / (K * dt)
    @all(P) = @all(P) + @all(RP) / (1.0 / (r / θ_dτ * @all(η)) + 1.0 / (K * dt))
    return nothing
end

@parallel function compute_V!(Vx, Vy, P, τxx, τyy, τxyv, ηdτ, ρgx, ρgy, ητ, _dx, _dy)
    @inn(Vx) =
        @inn(Vx) +
        (-@d_xa(P) * _dx + @d_xa(τxx) * _dx + @d_yi(τxyv) * _dy - @av_xa(ρgx)) * ηdτ /
        @harm_xa(ητ)
    @inn(Vy) =
        @inn(Vy) +
        (-@d_ya(P) * _dy + @d_ya(τyy) * _dy + @d_xi(τxyv) * _dx - @av_ya(ρgy)) * ηdτ /
        @harm_ya(ητ)
    return nothing
end

@parallel_indices (i, j) function compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy)
    # Again, indices i, j are captured by the closure
    Base.@propagate_inbounds @inline d_xa(A) = (A[i + 1, j] - A[i, j]) * _dx
    Base.@propagate_inbounds @inline d_ya(A) = (A[i, j + 1] - A[i, j]) * _dy
    Base.@propagate_inbounds @inline d_xi(A) = (A[i + 1, j + 1] - A[i, j + 1]) * _dx
    Base.@propagate_inbounds @inline d_yi(A) = (A[i + 1, j + 1] - A[i + 1, j]) * _dy
    Base.@propagate_inbounds @inline av_xa(A) = (A[i + 1, j] + A[i, j]) * 0.5
    Base.@propagate_inbounds @inline av_ya(A) = (A[i, j + 1] + A[i, j]) * 0.5
   
    if i ≤ size(Rx, 1) && j ≤ size(Rx, 2)
        @inbounds Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
    end
    if i ≤ size(Ry, 1) && j ≤ size(Ry, 2)
        @inbounds Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
    end
    
    return nothing
end

@parallel_indices (i, j) function compute_V_Res!(
    Vx, Vy, Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, ητ, ηdτ, _dx, _dy
)

    # Again, indices i, j are captured by the closure
    Base.@propagate_inbounds @inline d_xa(A) = (A[i + 1, j] - A[i, j]) * _dx
    Base.@propagate_inbounds @inline d_ya(A) = (A[i, j + 1] - A[i, j]) * _dy
    Base.@propagate_inbounds @inline d_xi(A) = (A[i + 1, j + 1] - A[i, j + 1]) * _dx
    Base.@propagate_inbounds @inline d_yi(A) = (A[i + 1, j + 1] - A[i + 1, j]) * _dy
    Base.@propagate_inbounds @inline av_xa(A) = (A[i + 1, j] + A[i, j]) * 0.5
    Base.@propagate_inbounds @inline av_ya(A) = (A[i, j + 1] + A[i, j]) * 0.5

    if all((i, j) .≤ size(Rx))
        @inbounds R = Rx[i, j] = d_xa(τxx) + d_yi(τxy) - d_xa(P) - av_xa(ρgx)
        @inbounds Vx[i + 1, j + 1] += R * ηdτ / av_xa(ητ)
    end
    if all((i, j) .≤ size(Ry))
        @inbounds R = Ry[i, j] = d_ya(τyy) + d_xi(τxy) - d_ya(P) - av_ya(ρgy)
        @inbounds Vy[i + 1, j + 1] += R * ηdτ / av_ya(ητ)
    end
    return nothing
end

# Stress calculation

# viscous
@parallel function compute_τ!(τxx, τyy, τxy, εxx, εyy, εxy, η, θ_dτ)
    @all(τxx) = @all(τxx) + (-@all(τxx) + 2.0 * @all(η) * @all(εxx)) * 1.0 / (θ_dτ + 1.0)
    @all(τyy) = @all(τyy) + (-@all(τyy) + 2.0 * @all(η) * @all(εyy)) * 1.0 / (θ_dτ + 1.0)
    @inn(τxy) = @inn(τxy) + (-@inn(τxy) + 2.0 * @harm(η) * @inn(εxy)) * 1.0 / (θ_dτ + 1.0)
    return nothing
end

# visco-elastic
@parallel function compute_τ!(
    τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, η, G, θ_dτ, dt
)
    @all(τxx) =
        @all(τxx) +
        (
            -(@all(τxx) - @all(τxx_o)) * @all(η) / (@all(G) * dt) - @all(τxx) +
            2.0 * @all(η) * @all(εxx)
        ) * 1.0 / (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @all(τyy) =
        @all(τyy) +
        (
            -(@all(τyy) - @all(τyy_o)) * @all(η) / (@all(G) * dt) - @all(τyy) +
            2.0 * @all(η) * @all(εyy)
        ) * 1.0 / (θ_dτ + @all(η) / (@all(G) * dt) + 1.0)
    @inn(τxy) =
        @inn(τxy) +
        (
            -(@inn(τxy) - @inn(τxy_o)) * @av(η) / (@av(G) * dt) - @inn(τxy) +
            2.0 * @av(η) * @inn(εxy)
        ) * 1.0 / (θ_dτ + @harm(η) / (@av(G) * dt) + 1.0)

    return nothing
end

# visco-elasto-plastic with GeoParams
@parallel_indices (i, j) function compute_τ_gp!(
    τxx,
    τyy,
    τxy,
    τII,
    τxx_o,
    τyy_o,
    τxyv_o,
    εxx,
    εyy,
    εxyv,
    η,
    η_vep,
    args_η,
    T,
    MatParam,
    dt,
    θ_dτ
)

    nx, ny = size(η)

    # convinience closure
    @inline gather(A) = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1] 
    @inline av(T)     = (T[i + 1, j] + T[i + 2, j] + T[i + 1, j + 1] + T[i + 2, j + 1]) * 0.25
    @inline function maxloc(A)
        max(
            A[i, j],
            A[min(i+1, nx), j],
            A[max(i-1, 1), j],
            A[i, min(j+1, ny)],
            A[i, max(j-1, 1),],
        )
    end

    @inbounds begin
        _Gdt                = inv(get_G(MatParam[1]) * dt)
        ηij                 = η[i, j]
        # # numerics
        dτ_r                = 1.0 / (θ_dτ + maxloc(η) * _Gdt + 1.0) # original
        # # Setup up input for GeoParams.jl
        # args                = (; dt=dt, P = 1e6 * (1 - z[j]) , T=av(T), τII_old=0.0)
        args                = (; dt=dt, P = (args_η.P[i, j]), depth = abs(args_η.depth[j]), T=av(T), τII_old=0.0)
        εij_p               = εxx[i, j]+1e-25, εyy[i, j]+1e-25, gather(εxyv).+1e-25
        τij_p_o             = τxx_o[i,j], τyy_o[i,j], gather(τxyv_o)
        phases              = (1, 1, (1,1,1,1)) # for now hard-coded for a single phase
        # update stress and effective viscosity
        τij, τII[i, j], ηᵢ  = compute_τij(MatParam, εij_p, args, τij_p_o, phases)

        εxy_p   = 0.25 * sum(εij_p[3])
        τxy_p_o = 0.25 * sum(τij_p_o[3])
    
        # Viscous stress
        a           = 0.5 / ηij
        εxx_visc    = a * τij[1]
        εyy_visc    = a * τij[2]
        εxy_visc    = a * τij[3]
        # Elastic stress
        a           = 0.5 * _Gdt
        εxx_el      = a * (τij[1] - τij_p_o[1])
        εyy_el      = a * (τij[2] - τij_p_o[2])
        εxy_el      = a * (τij[3] - τxy_p_o)
        # Plastic stress
        εxx_pl      = εij_p[1] - εxx_visc - εxx_el
        εyy_pl      = εij_p[2] - εyy_visc - εyy_el
        εxy_pl      = εxy_p    - εxy_visc - εxy_el
        # @show εxx_pl
        τxx[i,j]   += dτ_r * (-(τxx[i,j]-τij_p_o[1]) *  ηij * _Gdt - τxx[i,j] + 2.0 * ηij * (εij_p[1] - εxx_pl) ) # NOTE: from GP Tij = 2*η_vep * εij
        τyy[i,j]   += dτ_r * (-(τyy[i,j]-τij_p_o[2]) *  ηij * _Gdt - τyy[i,j] + 2.0 * ηij * (εij_p[2] - εyy_pl) ) 
        τxy[i,j]   += dτ_r * (-(τxy[i,j]-τxy_p_o)    *  ηij * _Gdt - τxy[i,j] + 2.0 * ηij * (εxy_p    - 0.5*εxy_pl) ) 
        η_vep[i, j] = ηᵢ

        # # ηᵢ                  = clamp(ηᵢ, 1e0, 1e6)
        # # τxx[i, j]          += dτ_r * (-(τxx[i,j]) + τij[1] ) / ηᵢ # NOTE: from GP Tij = 2*η_vep * εij
        # # τyy[i, j]          += dτ_r * (-(τyy[i,j]) + τij[2] ) / ηᵢ 
        # # τxy[i, j]          += dτ_r * (-(τxy[i,j]) + τij[3] ) / ηᵢ 
        # τxx[i, j]          += dτ_r * (-(τxx[i,j] - τij_p_o[1]) * ηᵢ / Gdt - τxx[i,j] + τij[1] ) # NOTE: from GP Tij = 2*η_vep * εij
        # τyy[i, j]          += dτ_r * (-(τyy[i,j] - τij_p_o[2]) * ηᵢ / Gdt - τyy[i,j] + τij[2] ) 
        # τxy[i, j]          += dτ_r * (-(τxy[i,j] - sum(τij_p_o[3])/3 ) * ηᵢ / Gdt - τxy[i,j] + τij[3] ) 
        # η_vep[i, j]         = ηᵢ
    end
    
    return nothing
end


@inline isplastic(x::AbstractPlasticity) = true
@inline isplastic(x) = false
@inline plastic_params(v) = plastic_params(v.CompositeRheology[1].elements)

@generated function plastic_params(v::NTuple{N, Any}) where N
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> isplastic(v[i]) && return true, v[i].C.val, v[i].sinϕ.val, v[i].η_vp.val
        (false, 0.0, 0.0, 0.0)
    end
end

@parallel_indices (i, j) function compute_τ_new!(
    τxx,
    τyy,
    τxy,
    TII,
    τxx_old,
    τyy_old,
    τxyv_old,
    εxx,
    εyy,
    εxyv,
    P,
    η,
    η_vep,
    MatParam,
    dt,
    θ_dτ,
    λ0
)
    nx, ny = size(η)

    # convinience closure
    @inline Base.@propagate_inbounds gather(A) = A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1] 
    @inline Base.@propagate_inbounds av(A)     = (A[i + 1, j] + A[i + 2, j] + A[i + 1, j + 1] + A[i + 2, j + 1]) * 0.25
    @inline Base.@propagate_inbounds function maxloc(A)
        max(
            A[i, j],
            A[min(i+1, nx), j],
            A[max(i-1, 1), j],
            A[i, min(j+1, ny)],
            A[i, max(j-1, 1),],
        )
    end

    @inbounds begin
        _Gdt        = inv(get_G(MatParam[1]) * dt)
        ηij         = η[i, j]
        dτ_r        = inv(θ_dτ + ηij * _Gdt + 1.0) # original
        # cache tensors
        εij_p       = εxx[i, j], εyy[i, j], gather(εxyv)
        τij_p_o     = τxx_old[i,j], τyy_old[i,j], gather(τxyv_old) 
        τij         = τxx[i,j], τyy[i,j], τxy[i, j]

        εxy_p       = 0.25 * sum(εij_p[3])
        τxy_p_o     = 0.25 * sum(τij_p_o[3])

        # Stress increment
        dτxx      = dτ_r * (-(τij[1] - τij_p_o[1]) * ηij * _Gdt - τij[1] + 2.0 * ηij * (εij_p[1]))
        dτyy      = dτ_r * (-(τij[2] - τij_p_o[2]) * ηij * _Gdt - τij[2] + 2.0 * ηij * (εij_p[2])) 
        dτxy      = dτ_r * (-(τij[3] - τxy_p_o   ) * ηij * _Gdt - τij[3] + 2.0 * ηij * (εxy_p   )) 
        τII_trial = sqrt(0.5*((τij[1]+dτxx)^2 + (τij[2]+dτyy)^2) + (τij[3]+dτxy)^2)
        if τII_trial != 0.0
            # yield function
            is_pl, C, sinϕ, η_reg = plastic_params(MatParam[1])
            F            = τII_trial - C - P[i,j]*sinϕ
            λ = λ0[i,j]  = 0.8 * λ0[i,j] + 0.2 * (F>0.0) * F /(η[i, j] * 1 + η_reg) * is_pl
            λdQdτxx      = 0.5 * (τij[1] + dτxx) / τII_trial * λ
            λdQdτyy      = 0.5 * (τij[2] + dτyy) / τII_trial * λ
            λdQdτxy      = 0.5 * (τij[3] + dτxy) / τII_trial * λ
           
            # corrected stress
            dτxx_pl  = dτ_r * (-(τij[1] - τij_p_o[1]) * ηij * _Gdt - τij[1] + 2.0 * ηij * (εij_p[1] - λdQdτxx))
            dτyy_pl  = dτ_r * (-(τij[2] - τij_p_o[2]) * ηij * _Gdt - τij[2] + 2.0 * ηij * (εij_p[2] - λdQdτyy)) 
            dτxy_pl  = dτ_r * (-(τij[3] - τxy_p_o)    * ηij * _Gdt - τij[3] + 2.0 * ηij * (εxy_p    - λdQdτxy)) 
            τxx[i,j] += dτxx_pl
            τyy[i,j] += dτyy_pl
            τxy[i,j] += dτxy_pl
        else
            τxx[i,j] += dτxx
            τyy[i,j] += dτyy
            τxy[i,j] += dτxy
        end
        
        # visco-elastic strain rates
        εxx_ve     = εij_p[1] + 0.5 * τij_p_o[1] * _Gdt
        εyy_ve     = εij_p[2] + 0.5 * τij_p_o[2] * _Gdt
        εxy_ve     = εxy_p    + 0.5 * τxy_p_o    * _Gdt
        εII_ve     = sqrt(0.5*(εxx_ve^2 + εyy_ve^2) + εxy_ve^2)
        TII[i,j]   = sqrt(0.5*(τxx[i,j]^2 + τyy[i,j]^2) + τxy[i,j]^2)
        η_vep[i,j] = TII[i,j] * 0.5 / εII_ve

    end
    
    return nothing
end

## 2D VISCO-ELASTIC STOKES SOLVER 

# viscous solver
function JustRelax.solve!(
    stokes::StokesArrays{Viscous,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    freeslip,
    ρg,
    η,
    K,
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _dx, _dy = inv.(di)
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    εxx, εyy, εxy = strain(stokes)
    τ, _ = stress(stokes)
    τxx, τyy, τxy = τ
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry, RP = stokes.R.Rx, stokes.R.Ry, stokes.R.RP
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    nx, ny = size(P)

    ρgx, ρgy = ρg
    P_old = deepcopy(P)

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)

    # errors
    err = 2 * ϵ
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin

            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy)

            @parallel compute_∇V!(∇V, Vx, Vy, _dx, _dy)
            @parallel compute_strain_rate!(εxx, εyy, εxy, ∇V, Vx, Vy, _dx, _dy)
            @parallel compute_P!(P, P_old, RP, ∇V, η, K, dt, r, θ_dτ)
            @parallel compute_τ!(τxx, τyy, τxy, εxx, εyy, εxy, η, θ_dτ)
            @parallel compute_V!(Vx, Vy, P, τxx, τyy, τxy, ηdτ, ρgx, ρgy, ητ, _dx, _dy)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (1:nx, 1:ny) compute_Res!(
                Rx, Ry, P, τxx, τyy, τxy, ρgx, ρgy, _dx, _dy
            )
            Vmin, Vmax = extrema(Vx)
            Pmin, Pmax = extrema(P)
            push!(norm_Rx, norm(Rx) / (Pmax - Pmin) * lx / sqrt(length(Rx)))
            push!(norm_Ry, norm(Ry) / (Pmax - Pmin) * lx / sqrt(length(Ry)))
            push!(norm_∇V, norm(∇V) / (Vmax - Vmin) * lx / sqrt(length(∇V)))
            err = max(norm_Rx[end], norm_Ry[end], norm_∇V[end])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if (verbose && err > ϵ) || (iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
        end
    end

    update_τ_o!(stokes)

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

# visco-elastic solver
function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    η,
    G,
    K,
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _di = inv.(di)
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    nx, ny = size(stokes.P)
    P_old = deepcopy(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    apply_free_slip!((freeslip_x=true, freeslip_y=true), ητ, ητ)

    # errors
    err = 2 * ϵ
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, _di...)
            @parallel compute_strain_rate!(
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.xy,
                stokes.∇V,
                stokes.V.Vx,
                stokes.V.Vy,
                _di...,
            )
            @parallel compute_P!(stokes.P, P_old, stokes.R.RP, stokes.∇V, η, K, dt, r, θ_dτ)
            @parallel compute_τ!(
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                stokes.τ_o.xx,
                stokes.τ_o.yy,
                stokes.τ_o.xy,
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.xy,
                η,
                G,
                θ_dτ,
                dt,
            )
            @parallel compute_V!(
                stokes.V.Vx,
                stokes.V.Vy,
                stokes.P,
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                ηdτ,
                ρg...,
                ητ,
                _di...,
            )
            # free slip boundary conditions
            flow_bcs!(stokes, flow_bcs, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (1:nx, 1:ny) compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                stokes.P,
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                ρg...,
                _di...,
            )
            errs = ntuple(Val(3)) do i
                maximum(x->abs.(x), getfield(stokes.R, i))
            end
            err = maximum(errs)
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if (verbose) || (iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            end
        end
    end

    if -Inf < dt < Inf 
        update_τ_o!(stokes)
        @parallel (1:nx, 1:ny) rotate_stress!(@tuple(stokes.V), @tuple(stokes.τ_o), _di, dt)
    end

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end

# GeoParams: general (visco-elasto-plastic) solver

tupleize(v::MaterialParams) = (v,)
tupleize(v::Tuple) = v

@parallel function center2vertex!(vertex, center)
    @inn(vertex) = @av(center)
    return nothing
end


@parallel_indices (i, j) function maxloc!(B, A)

    nx, ny = size(A)

    @inline function harmloc(A)
        4 * (inv.((
            A[i, j],
            A[min(i+1, nx), j],
            A[max(i-1, 1), j],
            A[i, min(j+1, ny)],
            A[i, max(j-1, 1),],
        )) |> sum |> inv)
    end

    @inline function maxloc(A)
        max(
            A[i, j],
            A[min(i+1, nx), j],
            A[max(i-1, 1), j],
            A[i, min(j+1, ny)],
            A[i, max(j-1, 1),],
        )
    end

    @inline function minloc(A)
        min(
            A[i, j],
            A[min(i+1, nx), j],
            A[max(i-1, 1), j],
            A[i, min(j+1, ny)],
            A[i, max(j-1, 1),],
        )
    end

    B[i, j] =  maxloc(A)
    return 
end


function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    thermal::ThermalArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    flow_bcs,
    ρg,
    η,
    η_vep,
    args_η,
    MatParam,
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    _di = inv.(di)
    ϵ, r, θ_dτ, ηdτ = pt_stokes.ϵ, pt_stokes.r, pt_stokes.θ_dτ, pt_stokes.ηdτ
    ni = nx, ny = size(stokes.P)
    P_old = deepcopy(stokes.P)
    # z = LinRange(di[2]*0.5, 1.0-di[2]*0.5, ny)
    # ~preconditioner
    ητ = deepcopy(η)
    @parallel maxloc!(ητ, η)
    # @parallel compute_maxloc!(ητ, η)
    # apply_free_slip!((freeslip_x=true, freeslip_y=true), ητ, ητ)

    rheology = tupleize(MatParam.linear)
    Kb = get_Kb(MatParam.linear)
    # rheology = tupleize(MatParam)
    # Kb = get_Kb(MatParam)

    # errors
    err = 2 * ϵ
    iter = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    λ = @zeros(ni...)
    # solver loop
    wtime0 = 0.0
    # while iter < 1
    while iter < 2 || (err > ϵ && iter ≤ iterMax)

        wtime0 += @elapsed begin
            @parallel compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, _di...)
            @parallel compute_P!(
                stokes.P,
                P_old,
                stokes.R.RP,
                stokes.∇V,
                η,
                Kb,
                dt,
                r,
                θ_dτ,
            )

            # Update buoyancy and viscosity -
            # args_ηv = (; T = thermal.T, P = stokes.P, depth=args_η.depth, dt=Inf)
            # @parallel (@idx ni) compute_viscosity_gp!(η, args_ηv, rheology)
            @parallel (@idx ni) compute_ρg!(ρg[2], rheology[1], (T=thermal.T, P=stokes.P))
            # @parallel maxloc!(ητ, η)
            # η0 = deepcopy(η)

            @parallel compute_strain_rate!(
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.xy,
                stokes.∇V,
                @tuple(stokes.V)...,
                _di...,
            )

            @parallel (@idx ni) compute_τ_new!(
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy_c,
                stokes.τ.II,
                stokes.τ_o.xx,
                stokes.τ_o.yy,
                stokes.τ_o.xy,
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.xy,
                stokes.P,
                η,
                η_vep,
                rheology, # needs to be a tuple
                dt,
                θ_dτ,
                λ
            )
            # @show η == η0
            # @parallel (@idx ni) compute_τ_gp!(
            #     stokes.τ.xx,
            #     stokes.τ.yy,
            #     stokes.τ.xy_c,
            #     stokes.τ.II,
            #     stokes.τ_o.xx,
            #     stokes.τ_o.yy,
            #     stokes.τ_o.xy,
            #     stokes.ε.xx,
            #     stokes.ε.yy,
            #     stokes.ε.xy,
            #     η,
            #     η_vep,
            #     args_η,
            #     thermal.T,
            #     rheology, # needs to be a tuple
            #     dt,
            #     θ_dτ,
            # )

            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
            
            # @parallel maxloc!(ητ, η_vep)
            @parallel compute_V!(
                @tuple(stokes.V)...,
                stokes.P,
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                ηdτ,
                ρg...,
                ητ,
                _di...,
            )
            # apply boundary conditions boundary conditions
            flow_bcs!(stokes, flow_bcs, di)
            # ------------------------------

        end

        iter += 1
        if iter % nout == 0 && iter > 1
            @parallel (@idx ni) compute_Res!(
                stokes.R.Rx,
                stokes.R.Ry,
                stokes.P,
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.xy,
                ρg[1],
                ρg[2],
                _di...,
            )
            errs = ntuple(Val(3)) do i
                maximum(x->abs.(x), getfield(stokes.R, i))
            end
            err = maximum(errs)
            push!(norm_Rx, errs[1])
            push!(norm_Ry, errs[2])
            push!(norm_∇V, errs[3])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            # if (verbose && err > ϵ) || (iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[end],
                    norm_Ry[end],
                    norm_∇V[end]
                )
            # end
        end
    end

    # # non linear

    # println("starting non linear iterations")
    # rheology = tupleize(MatParam.plastic)
    # Kb = get_Kb(MatParam.plastic)

    #  # errors
    #  err = 2 * ϵ
    #  iter = 0
    #  err_evo1 = Float64[]
    #  err_evo2 = Float64[]
    #  norm_Rx = Float64[]
    #  norm_Ry = Float64[]
    #  norm_∇V = Float64[]
 
    #  # solver loop
    #  wtime0 = 0.0
    # @show iter, err

    # #  while iter < 0
    # while iter < 2 || (err > ϵ && iter ≤ iterMax)
    #      wtime0 += @elapsed begin
    #          @parallel compute_∇V!(stokes.∇V, stokes.V.Vx, stokes.V.Vy, _di...)
    #          @parallel compute_P!(
    #              stokes.P,
    #              P_old,
    #              stokes.R.RP,
    #              stokes.∇V,
    #              η,
    #              Kb,
    #              dt,
    #              r,
    #              θ_dτ,
    #          )
 
    #         # Update buoyancy and viscosity -
    #         args_ηv = (; T = thermal.T, P = stokes.P, depth=args_η.depth, dt=Inf)
    #         # @parallel (@idx ni) compute_viscosity_gp!(η, args_ηv, rheology)
    #         @parallel (@idx ni) compute_ρg!(ρg[2], rheology[1], (T=thermal.T, P=stokes.P))
    #         # @parallel maxloc!(ητ, η)
             
    #          @parallel compute_strain_rate!(
    #              stokes.ε.xx,
    #              stokes.ε.yy,
    #              stokes.ε.xy,
    #              stokes.∇V,
    #              @tuple(stokes.V)...,
    #              _di...,
    #          )
    #          @parallel (@idx ni) compute_τ_new!(
    #             stokes.τ.xx,
    #             stokes.τ.yy,
    #             stokes.τ.xy_c,
    #             stokes.τ.II,
    #             stokes.τ_o.xx,
    #             stokes.τ_o.yy,
    #             stokes.τ_o.xy,
    #             stokes.ε.xx,
    #             stokes.ε.yy,
    #             stokes.ε.xy,
    #             stokes.P,
    #             η,
    #             η_vep,
    #             rheology, # needs to be a tuple
    #             dt,
    #             θ_dτ,
    #             λ
    #         )
          
    #         # @parallel (@idx ni) compute_τ_gp!(
    #         #     stokes.τ.xx,
    #         #     stokes.τ.yy,
    #         #     stokes.τ.xy_c,
    #         #     stokes.τ.II,
    #         #     stokes.τ_o.xx,
    #         #     stokes.τ_o.yy,
    #         #     stokes.τ_o.xy,
    #         #     stokes.ε.xx,
    #         #     stokes.ε.yy,
    #         #     stokes.ε.xy,
    #         #     η,
    #         #     η_vep,
    #         #     args_η,
    #         #     thermal.T,
    #         #     rheology, # needs to be a tuple
    #         #     dt,
    #         #     θ_dτ,
    #         # )
    #          @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)
             
    #          # @parallel maxloc!(ητ, η_vep)
    #          @parallel compute_V!(
    #              @tuple(stokes.V)...,
    #              stokes.P,
    #              stokes.τ.xx,
    #              stokes.τ.yy,
    #              stokes.τ.xy,
    #              ηdτ,
    #              ρg...,
    #              ητ,
    #              _di...,
    #          )
    #          # apply boundary conditions boundary conditions
    #          flow_bcs!(stokes, flow_bcs, di)
 
    #          # ------------------------------
 
    #      end

    #      iter += 1

    #      if iter % nout == 0 && iter > 2
    #          @parallel (@idx ni) compute_Res!(
    #              stokes.R.Rx,
    #              stokes.R.Ry,
    #              stokes.P,
    #              stokes.τ.xx,
    #              stokes.τ.yy,
    #              stokes.τ.xy,
    #              ρg[1],
    #              ρg[2],
    #              _di...,
    #          )
    #          errs = ntuple(Val(3)) do i
    #              maximum(x->abs.(x), getfield(stokes.R, i))
    #          end
    #          err = maximum(errs)
    #          push!(norm_Rx, errs[1])
    #          push!(norm_Ry, errs[2])
    #          push!(norm_∇V, errs[3])
    #          push!(err_evo1, err)
    #          push!(err_evo2, iter)
    #          if (verbose && err > ϵ) || (iter == iterMax)
    #              @printf(
    #                  "JODER Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
    #                  iter,
    #                  err,
    #                  norm_Rx[end],
    #                  norm_Ry[end],
    #                  norm_∇V[end]
    #              )
    #          end
    #      end
    #  end

    # println("Non linear iterations done in $iter iterations")

    if -Inf < dt < Inf 
        update_τ_o!(stokes)
        @parallel (@idx ni) rotate_stress!(@tuple(stokes.V), @tuple(stokes.τ_o), _di, dt)
    end

    return λ, (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_∇V=norm_∇V,
    )
end


# HELPER FUNCTIONS ---------------------------------------------------------------
# visco-elasto-plastic with GeoParams
@parallel_indices (i, j) function compute_viscosity_gp!(η, args, MatParam)

    # convinience closure
    @inline av(T)     = (T[i + 1, j] + T[i + 2, j] + T[i + 1, j + 1] + T[i + 2, j + 1]) * 0.25

    @inbounds begin
        args_ij       = (; dt = args.dt, P = (args.P[i, j]), T=av(args.T), depth=abs(args.depth[j]), τII_old=0.0)
        εij_p         = 1.0, 1.0, (1.0, 1.0, 1.0, 1.0)
        τij_p_o       = 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
        phases        = 1, 1, (1,1,1,1) # for now hard-coded for a single phase
        # # update stress and effective viscosity
        _, _, η[i, j] = compute_τij(MatParam, εij_p, args_ij, τij_p_o, phases)
    end
    
    return nothing
end

@parallel_indices (i, j) function compute_ρg!(ρg, rheology, args)

    @inline av(T) = 0.25* (T[i+1,j] + T[i+2,j] + T[i+1,j+1] + T[i+2,j+1]) - 273.0

    @inbounds ρg[i, j] = -compute_density(rheology, (; T = av(args.T), P=args.P[i, j])) * compute_gravity(rheology.Gravity[1])

    return nothing
end

end # END OF MODULE
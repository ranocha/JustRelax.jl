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

## 3D ELASTICITY MODULE

module Elasticity3D

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using JustRelax
using CUDA
using LinearAlgebra
using Printf
using GeoParams

import JustRelax:
    stress, strain, elastic_iter_params!, PTArray, Velocity, SymmetricTensor, pureshear_bc!
import JustRelax:
    Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic, IGG
import JustRelax: compute_maxloc!, solve!

export solve!, pureshear_bc!

@parallel function update_τ_o!(
    τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o, τxx, τyy, τzz, τxy, τxz, τyz
)
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τzz_o) = @all(τzz)
    @all(τxy_o) = @all(τxy)
    @all(τxz_o) = @all(τxz)
    @all(τyz_o) = @all(τyz)
    return nothing
end

function update_τ_o!(stokes::StokesArrays{ViscoElastic,A,B,C,D,3}) where {A,B,C,D}
    # unpack
    τ, τ_o = stress(stokes)
    τxx, τyy, τzz, τxy, τxz, τyz = τ
    τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o = τ_o
    # copy
    @parallel update_τ_o!(
        τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o, τxx, τyy, τzz, τxy, τxz, τyz
    )
end

@parallel_indices (i, j, k) function compute_∇V!(∇V, Vx, Vy, Vz, _dx, _dy, _dz)
    @inbounds ∇V[i, j, k] =
        _dx * (Vx[i + 1, j + 1, k + 1] - Vx[i, j + 1, k + 1]) +
        _dy * (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j, k + 1]) +
        _dz * (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j + 1, k])
    return nothing
end

@parallel_indices (i, j, k) function compute_strain_rate!(
    ∇V, εxx, εyy, εzz, εyz, εxz, εxy, Vx, Vy, Vz, _dx, _dy, _dz
)
    @inbounds begin
        # normal components are all located @ cell centers
        if all((i, j, k) .≤ size(εxx))
            ∇Vᵢⱼₖ = ∇V[i, j, k] / 3.0
            # Compute ε_xx
            εxx[i, j, k] = _dx * (Vx[i + 1, j + 1, k + 1] - Vx[i, j + 1, k + 1]) - ∇Vᵢⱼₖ
            # Compute ε_yy
            εyy[i, j, k] = _dy * (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j, k + 1]) - ∇Vᵢⱼₖ
            # Compute ε_zz
            εzz[i, j, k] = _dz * (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j + 1, k]) - ∇Vᵢⱼₖ
        end
        # Compute ε_yz
        if all((i, j, k) .≤ size(εyz))
            εyz[i, j, k] =
                0.5 * (
                    _dz * (Vy[i + 1, j, k + 1] - Vy[i + 1, j, k]) +
                    _dy * (Vz[i + 1, j + 1, k] - Vz[i + 1, j, k])
                )
        end
        # Compute ε_xz
        if all((i, j, k) .≤ size(εxz))
            εxz[i, j, k] =
                0.5 * (
                    _dz * (Vx[i, j + 1, k + 1] - Vx[i, j + 1, k]) +
                    _dx * (Vz[i + 1, j + 1, k] - Vz[i, j + 1, k])
                )
        end
        # Compute ε_xy
        if all((i, j, k) .≤ size(εxy))
            εxy[i, j, k] =
                0.5 * (
                    _dy * (Vx[i, j + 1, k + 1] - Vx[i, j, k + 1]) +
                    _dx * (Vy[i + 1, j, k + 1] - Vy[i, j, k + 1])
                )
        end
    end
    return nothing
end

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

## With phase ratios and GeoParams
@parallel_indices (i, j, k) function compute_P!(P, P_old, RP, ∇V, η, rheology, phase_ratios, dt, r, θ_dτ)
    _Kdt        = fn_ratio(get_Kb, rheology, phase_ratios[i, j, k]) * dt |> inv
    RP[i, j, k] = -∇V[i, j, k] - (P[i, j, k] - P_old[i, j, k]) * _Kdt
    P[i, j, k]  =   P[i, j, k] + RP[i, j, k] / (inv(r * η[i, j, k] / θ_dτ) + _Kdt)
    return nothing
end

@parallel_indices (i, j, k) function compute_V!(
    Vx,
    Vy,
    Vz,
    Rx,
    Ry,
    Rz,
    P,
    fx,
    fy,
    fz,
    τxx,
    τyy,
    τzz,
    τyz,
    τxz,
    τxy,
    ητ,
    ηdτ,
    _dx,
    _dy,
    _dz,
)
    Base.@propagate_inbounds @inline harm_x(x) = 2.0 / (1.0 / x[i + 1, j, k] + 1.0 / x[i, j, k])
    Base.@propagate_inbounds @inline harm_y(x) = 2.0 / (1.0 / x[i, j + 1, k] + 1.0 / x[i, j, k])
    Base.@propagate_inbounds @inline harm_z(x) = 2.0 / (1.0 / x[i, j, k + 1] + 1.0 / x[i, j, k])
    Base.@propagate_inbounds @inline av_x(x)   = 0.5 * (x[i + 1, j, k] + x[i, j, k])
    Base.@propagate_inbounds @inline av_y(x)   = 0.5 * (x[i, j + 1, k] + x[i, j, k])
    Base.@propagate_inbounds @inline av_z(x)   = 0.5 * (x[i, j, k + 1] + x[i, j, k])
    Base.@propagate_inbounds @inline dx(x)     = x[i + 1, j, k] - x[i, j, k]
    Base.@propagate_inbounds @inline dy(x)     = x[i, j + 1, k] - x[i, j, k]
    Base.@propagate_inbounds @inline dz(x)     = x[i, j, k + 1] - x[i, j, k]

    @inbounds begin
        if all((i, j, k) .< size(Vx) .- 1)
            Rx_ijk =
                _dx * (τxx[i + 1, j    , k    ] - τxx[i    , j, k]) +
                _dy * (τxy[i + 1, j + 1, k    ] - τxy[i + 1, j, k]) +
                _dz * (τxz[i + 1, j    , k + 1] - τxz[i + 1, j, k]) -
                _dx * dx(P) + av_x(fx)
            Vx[i + 1, j + 1, k + 1] += Rx_ijk * ηdτ / av_x(ητ)
            Rx[i, j, k] = Rx_ijk
        end
        if all((i, j, k) .< size(Vy) .- 1)
            Ry_ijk =
                _dx * (τxy[i + 1, j + 1, k    ] - τxy[i, j + 1, k]) +
                _dy * (τyy[i    , j + 1, k    ] - τyy[i, j    , k]) +
                _dz * (τyz[i    , j + 1, k + 1] - τyz[i, j + 1, k]) -
                _dy * dy(P) + av_y(fy)
            Vy[i + 1, j + 1, k + 1] += Ry_ijk * ηdτ / av_y(ητ)
            Ry[i, j, k] = Ry_ijk
        end
        if all((i, j, k) .< size(Vz) .- 1)
            Rz_ijk =
                _dx * (τxz[i + 1, j    , k + 1] - τxz[i, j, k + 1]) +
                _dy * (τyz[i    , j + 1, k + 1] - τyz[i, j, k + 1]) +
                _dz * (τzz[i    , j    , k + 1] - τzz[i, j, k    ]) - 
                _dz * dz(P) + av_z(fz)
            Vz[i + 1, j + 1, k + 1] += Rz_ijk * ηdτ / av_z(ητ)
            Rz[i, j, k] = Rz_ijk
        end
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_τ!(
    τxx,
    τyy,
    τzz,
    τyz,
    τxz,
    τxy,
    τxx_o,
    τyy_o,
    τzz_o,
    τyz_o,
    τxz_o,
    τxy_o,
    εxx,
    εyy,
    εzz,
    εyz,
    εxz,
    εxy,
    η,
    G,
    dt,
    θ_dτ,
)
    Base.@propagate_inbounds @inline function harm_xy(x)
        4.0 / (
            1.0 / x[i - 1, j - 1, k] +
            1.0 / x[i - 1, j, k] +
            1.0 / x[i, j - 1, k] +
            1.0 / x[i, j, k]
        )
    end
    Base.@propagate_inbounds @inline function harm_xz(x)
        4.0 / (
            1.0 / x[i, j, k] +
            1.0 / x[i - 1, j, k] +
            1.0 / x[i, j, k - 1] +
            1.0 / x[i - 1, j, k - 1]
        )
    end
    Base.@propagate_inbounds @inline function harm_yz(x)
        4.0 / (
            1.0 / x[i, j, k] +
            1.0 / x[i, j - 1, k] +
            1.0 / x[i, j, k - 1] +
            1.0 / x[i, j - 1, k - 1]
        )
    end
    Base.@propagate_inbounds @inline function av_xy(x)
        0.25 * (x[i - 1, j - 1, k] + x[i - 1, j, k] + x[i, j - 1, k] + x[i, j, k])
    end
    Base.@propagate_inbounds @inline function av_xz(x)
        0.25 * (x[i, j, k] + x[i - 1, j, k] + x[i, j, k - 1] + x[i - 1, j, k - 1])
    end
    Base.@propagate_inbounds @inline function av_yz(x)
        0.25 * (x[i, j, k] + x[i, j - 1, k] + x[i, j, k - 1] + x[i, j - 1, k - 1])
    end
    Base.@propagate_inbounds @inline current(x) = x[i, j, k]

    @inbounds begin
        if all((i, j, k) .≤ size(τxx))
            # Compute τ_xx
            τxx[i, j, k] +=
                (
                    -(current(τxx) - current(τxx_o)) * current(η) / (current(G) * dt) -
                    current(τxx) + 2.0 * current(η) * current(εxx)
                ) / (θ_dτ + current(η) / (current(G) * dt) + 1.0)
            # Compute τ_yy
            τyy[i, j, k] +=
                (
                    -(current(τyy) - current(τyy_o)) * current(η) / (current(G) * dt) -
                    current(τyy) + 2.0 * current(η) * current(εyy)
                ) / (θ_dτ + current(η) / (current(G) * dt) + 1.0)
            # Compute τ_zz
            τzz[i, j, k] +=
                (
                    -(current(τzz) - current(τzz_o)) * current(η) / (current(G) * dt) -
                    current(τzz) + 2.0 * current(η) * current(εzz)
                ) / (θ_dτ + current(η) / (current(G) * dt) + 1.0)
        end
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            τxy[i, j, k] +=
                (
                    -(current(τxy) - current(τxy_o)) * harm_xy(η) / (harm_xy(G) * dt) -
                    current(τxy) + 2.0 * harm_xy(η) * current(εxy)
                ) / (θ_dτ + harm_xy(η) / (harm_xy(G) * dt) + 1.0)
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            τxz[i, j, k] +=
                (
                    -(current(τxz) - current(τxz_o)) * harm_xz(η) / (harm_xz(G) * dt) -
                    current(τxz) + 2.0 * harm_xz(η) * current(εxz)
                ) / (θ_dτ + harm_xz(η) / (harm_xz(G) * dt) + 1.0)
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            τyz[i, j, k] +=
                (
                    -(current(τyz) - current(τyz_o)) * harm_yz(η) / (harm_yz(G) * dt) -
                    current(τyz) + 2.0 * harm_yz(η) * current(εyz)
                ) / (θ_dτ + harm_yz(η) / (harm_yz(G) * dt) + 1.0)
        end
    end
    return nothing
end

@parallel_indices (i, j, k) function compute_τ_vertex!(
    τyz,
    τxz,
    τxy,
    τyz_o,
    τxz_o,
    τxy_o,
    εyz,
    εxz,
    εxy,
    η,
    G,
    dt,
    θ_dτ,
)
    
    Base.@propagate_inbounds @inline function av_xy(x)
        0.25 * (x[i - 1, j - 1, k] + x[i - 1, j, k] + x[i, j - 1, k] + x[i, j, k])
    end
    Base.@propagate_inbounds @inline function av_xz(x)
        0.25 * (x[i, j, k] + x[i - 1, j, k] + x[i, j, k - 1] + x[i - 1, j, k - 1])
    end
    Base.@propagate_inbounds @inline function av_yz(x)
        0.25 * (x[i, j, k] + x[i, j - 1, k] + x[i, j, k - 1] + x[i, j - 1, k - 1])
    end
    Base.@propagate_inbounds @inline current(x) = x[i, j, k]

    @inbounds begin
        # Compute τ_xy
        if (1 < i < size(τxy, 1)) && (1 < j < size(τxy, 2)) && k ≤ size(τxy, 3)
            τxy[i, j, k] +=
                (
                    -(current(τxy) - current(τxy_o)) * av_xy(η) / (G * dt) -
                    current(τxy) + 2.0 * av_xy(η) * current(εxy)
                ) / (θ_dτ + av_xy(η) / (G * dt) + 1.0)
        end
        # Compute τ_xz
        if (1 < i < size(τxz, 1)) && j ≤ size(τxz, 2) && (1 < k < size(τxz, 3))
            τxz[i, j, k] +=
                (
                    -(current(τxz) - current(τxz_o)) * av_xz(η) / (G * dt) -
                    current(τxz) + 2.0 * av_xz(η) * current(εxz)
                ) / (θ_dτ + av_xz(η) / (G * dt) + 1.0)
        end
        # Compute τ_yz
        if i ≤ size(τyz, 1) && (1 < j < size(τyz, 2)) && (1 < k < size(τyz, 3))
            τyz[i, j, k] +=
                (
                    -(current(τyz) - current(τyz_o)) * av_yz(η) / (G * dt) -
                    current(τyz) + 2.0 * av_yz(η) * current(εyz)
                ) / (θ_dτ + av_yz(η) / (G * dt) + 1.0)
        end
    end
    return nothing
end

# visco-elasto-plastic with GeoParams
@parallel_indices (i, j, k) function compute_τ_gp!(
    τxx,
    τyy, 
    τzz, 
    τyz, 
    τxz, 
    τxy,
    τII, 
    τxx_o, 
    τyy_o, 
    τzz_o, 
    τyz_o, 
    τxz_o, 
    τxy_o,
    εxx, 
    εyy, 
    εzz, 
    εyz, 
    εxz, 
    εxy,
    η, 
    η_vep, 
    args_η, 
    T, 
    rheology, 
    dt, 
    θ_dτ,
)
    # convinience closures
    @inline gather_yz(A) =  A[i, j, k], A[i    , j + 1, k], A[i, j    , k + 1], A[i    , j + 1, k + 1]
    @inline gather_xz(A) =  A[i, j, k], A[i + 1, j    , k], A[i, j    , k + 1], A[i + 1, j    , k + 1]
    @inline gather_xy(A) =  A[i, j, k], A[i + 1, j    , k], A[i, j + 1, k    ], A[i + 1, j + 1, k    ]
    @inline av(T) = 0.125 * (
        T[i, j, k  ] + T[i, j+1, k  ] + T[i+1, j, k  ] + T[i+1, j+1, k  ] +
        T[i, j, k+1] + T[i, j+1, k+1] + T[i+1, j, k+1] + T[i+1, j+1, k+1]
    )

    @inbounds begin
        # dτ_r = 1.0 / (θ_dτ + η[i, j, k] / (get_G(rheology[1]) * dt) + 1.0) # original
        dτ_r  = 1.0 / (θ_dτ / η[i, j, k] + 1.0 / η_vep[i, j, k]) # equivalent to dτ_r = @. 1.0/(θ_dτ + η/(G*dt) + 1.0)
        # Setup up input for GeoParams.jl
        # T_cell = 0.125 * (
        #     T[i, j, k  ] + T[i, j+1, k  ] + T[i+1, j, k  ] + T[i+1, j+1, k  ] +
        #     T[i, j, k+1] + T[i, j+1, k+1] + T[i+1, j, k+1] + T[i+1, j+1, k+1]
        # )
        # args = (; dt=dt, P = 1e6 * (1 - z[k]), T=T_cell, τII_old=0.0)
        args = (; dt=dt, P = (args_η.P[i, j, k]), depth = abs(args_η.depth[k]), T=av(T), τII_old=0.0)
        # args = (; dt=dt, P = (args_η.P[i, j, k]), depth = abs(args_η.depth[k]), T=1.e3, τII_old=0.0)

        εij_p = (
            εxx[i, j, k]+1e-25, 
            εyy[i, j, k]+1e-25, 
            εzz[i, j, k]+1e-25, 
            gather_yz(εyz).+1e-25, 
            gather_xz(εxz).+1e-25, 
            gather_xy(εxy).+1e-25
        )
        τij_p_o = (
            τxx_o[i, j, k],
            τyy_o[i, j, k],
            τzz_o[i, j, k],
            gather_yz(τyz_o), 
            gather_xz(τxz_o), 
            gather_xy(τxy_o)
        )
        phases = 1, 1, 1, (1,1,1,1), (1,1,1,1), (1,1,1,1) # for now hard-coded for a single phase
        # update stress and effective viscosity
        τij, τII[i, j, k], ηᵢ = compute_τij(rheology, εij_p, args, τij_p_o, phases)
        τ = ( # caching out improves a wee bit the performance
            τxx[i, j, k],
            τyy[i, j, k],
            τzz[i, j, k],
            # τyz[i, j, k],
            # τxz[i, j, k],
            # τxy[i, j, k], 
        )
        dτ_rηᵢ = dτ_r/ηᵢ
        τxx[i, j, k]  += dτ_rηᵢ * (-τ[1] + τij[1]) # NOTE: from GP Tij = 2*η_vep * εij
        τyy[i, j, k]  += dτ_rηᵢ * (-τ[2] + τij[2]) 
        τzz[i, j, k]  += dτ_rηᵢ * (-τ[3] + τij[3]) 
        # τyz[i, j, k]  += dτ_rηᵢ * (-τ[4] + τij[4]) 
        # τxz[i, j, k]  += dτ_rηᵢ * (-τ[5] + τij[5]) 
        # τxy[i, j, k]  += dτ_rηᵢ * (-τ[6] + τij[6]) 
        η_vep[i, j, k] = ηᵢ
    end
    return
end

## BOUNDARY CONDITIONS 

function JustRelax.pureshear_bc!(
    stokes::StokesArrays, di::NTuple{3,T}, li::NTuple{3,T}, εbg
) where {T}
    # unpack
    Vx, _, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz
    dx, _, dz = di
    lx, _, lz = li
    # Velocity pure shear boundary conditions
    stokes.V.Vx .= PTArray([
        -εbg * ((i - 1) * dx - 0.5 * lx) for i in 1:size(Vx, 1), j in 1:size(Vx, 2),
        k in 1:size(Vx, 3)
    ])
    return stokes.V.Vz .= PTArray([
        εbg * ((k - 1) * dz - 0.5 * lz) for i in 1:size(Vz, 1), j in 1:size(Vz, 2),
        k in 1:size(Vz, 3)
    ])
end

## 3D VISCO-ELASTIC STOKES SOLVER 

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,3},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{3,T},
    freeslip,
    ρg,
    η,
    K,
    G,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {A,B,C,D,T}
    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    nx, ny, nz = size(stokes.P)

    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end
    @parallel (1:size(ητ, 2), 1:size(ητ, 3)) free_slip_x!(ητ)
    @parallel (1:size(ητ, 1), 1:size(ητ, 3)) free_slip_y!(ητ)
    @parallel (1:size(ητ, 1), 1:size(ητ, 2)) free_slip_z!(ητ)

    # errors
    err = 2 * ϵ
    iter = 0
    cont = 0
    err_evo1 = Float64[]
    err_evo2 = Int64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_Rz = Float64[]
    norm_∇V = Float64[]

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (1:nx, 1:ny, 1:nz) compute_∇V!(
                stokes.∇V, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz, _di...
            )
            @parallel compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                η,
                K,
                dt,
                pt_stokes.r,
                pt_stokes.θ_dτ,
            )
            @parallel (1:(nx + 1), 1:(ny + 1), 1:(nz + 1)) compute_strain_rate!(
                stokes.∇V,
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.zz,
                stokes.ε.yz,
                stokes.ε.xz,
                stokes.ε.xy,
                stokes.V.Vx,
                stokes.V.Vy,
                stokes.V.Vz,
                _di...,
            )
            @parallel (1:(nx + 1), 1:(ny + 1), 1:(nz + 1)) compute_τ!(
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.zz,
                stokes.τ.yz,
                stokes.τ.xz,
                stokes.τ.xy,
                stokes.τ_o.xx,
                stokes.τ_o.yy,
                stokes.τ_o.zz,
                stokes.τ_o.yz,
                stokes.τ_o.xz,
                stokes.τ_o.xy,
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.zz,
                stokes.ε.yz,
                stokes.ε.xz,
                stokes.ε.xy,
                η,
                G,
                dt,
                pt_stokes.θ_dτ,
            )
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    stokes.V.Vx,
                    stokes.V.Vy,
                    stokes.V.Vz,
                    stokes.R.Rx,
                    stokes.R.Ry,
                    stokes.R.Rz,
                    stokes.P,
                    ρg...,
                    stokes.τ.xx,
                    stokes.τ.yy,
                    stokes.τ.zz,
                    stokes.τ.yz,
                    stokes.τ.xz,
                    stokes.τ.xy,
                    ητ,
                    pt_stokes.ηdτ,
                    _di...,
                )
                update_halo!(stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
            end

            apply_free_slip!(freeslip, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            push!(norm_Rx, maximum(abs.(stokes.R.Rx)))
            push!(norm_Ry, maximum(abs.(stokes.R.Ry)))
            push!(norm_Rz, maximum(abs.(stokes.R.Rz)))
            push!(norm_∇V, maximum(abs.(stokes.R.RP)))
            err = max(norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && ((verbose && err > ϵ) || iter == iterMax)
                @printf(
                    "iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_Rz[cont],
                    norm_∇V[cont]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    av_time = wtime0 / (iter - 1) # average time per iteration
    update_τ_o!(stokes) # copy τ into τ_o

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_Rz=norm_Rz,
        norm_∇V=norm_∇V,
        time=wtime0,
        av_time=av_time,
    )
end

## 3D VISCO-ELASTO-PLASTIC STOKES SOLVER WITH GeoParams.jl 

# @parallel function center2vertex!(vertex_yz, vertex_xz, vertex_xy, center_yz, center_xz, center_xy)
#     @inn_yz(vertex_yz) = @av_yza(center_yz)
#     @inn_xz(vertex_xz) = @av_xza(center_xz)
#     @inn_xy(vertex_xy) = @av_xya(center_xy)
#     return nothing
# end

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,3},
    thermal::ThermalArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{3,T},
    flow_bc::FlowBoundaryConditions,
    ρg,
    η,
    η_vep,
    args_η,
    rheology::MaterialParams,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {A,B,C,D,T}

    ## UNPACK

    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    ni = nx, ny, nz = size(stokes.P)
    z = LinRange(di[3]*0.5, 1.0-di[3]*0.5, nz)

    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end
    @parallel (1:ny, 1:nz) free_slip_x!(ητ)
    @parallel (1:nx, 1:nz) free_slip_y!(ητ)
    @parallel (1:nx, 1:ny) free_slip_z!(ητ)

    # errors
    err = 2 * ϵ
    iter = 0
    cont = 0
    err_evo1 = Float64[]
    err_evo2 = Int64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_Rz = Float64[]
    norm_∇V = Float64[]

    Kb = get_Kb(rheology)
    G  = get_G(rheology)
    @copy stokes.P0 stokes.P

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel (@idx ni) compute_∇V!(
                stokes.∇V, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz, _di...
            )
            @parallel compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                η,
                Kb,
                dt,
                pt_stokes.r,
                pt_stokes.θ_dτ,
            )
            @parallel (@idx ni) compute_strain_rate!(
                stokes.∇V,
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.zz,
                stokes.ε.yz,
                stokes.ε.xz,
                stokes.ε.xy,
                stokes.V.Vx,
                stokes.V.Vy,
                stokes.V.Vz,
                _di...,
            )
            @parallel (@idx ni) compute_τ_gp!(
                stokes.τ.xx,
                stokes.τ.yy,
                stokes.τ.zz,
                stokes.τ.yz_c, 
                stokes.τ.xz_c, 
                stokes.τ.xy_c,
                stokes.τ.II,
                stokes.τ_o.xx,
                stokes.τ_o.yy,
                stokes.τ_o.zz, 
                stokes.τ_o.yz, 
                stokes.τ_o.xz, 
                stokes.τ_o.xy,
                stokes.ε.xx,
                stokes.ε.yy,
                stokes.ε.zz, 
                stokes.ε.yz,
                stokes.ε.xz,
                stokes.ε.xy,
                η,
                η_vep,
                args_η,
                thermal.T,
                tupleize(rheology), # needs to be a tuple
                dt,
                pt_stokes.θ_dτ,
            )
            @parallel (@idx ni.+1) compute_τ_vertex!(
                stokes.τ.yz,
                stokes.τ.xz,
                stokes.τ.xy,
                stokes.τ_o.yz,
                stokes.τ_o.xz,
                stokes.τ_o.xy,
                stokes.ε.yz,
                stokes.ε.xz,
                stokes.ε.xy,
                η_vep,
                G,
                dt,
                pt_stokes.θ_dτ,
            )
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    stokes.V.Vx,
                    stokes.V.Vy,
                    stokes.V.Vz,
                    stokes.R.Rx,
                    stokes.R.Ry,
                    stokes.R.Rz,
                    stokes.P,
                    ρg[1],
                    ρg[2],
                    ρg[3],
                    stokes.τ.xx,
                    stokes.τ.yy,
                    stokes.τ.zz,
                    stokes.τ.yz,
                    stokes.τ.xz,
                    stokes.τ.xy,
                    ητ,
                    pt_stokes.ηdτ,
                    _di...,
                )
                update_halo!(stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
            end
            # apply_free_slip!(flow_bc, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
            flow_bcs!(stokes, flow_bc, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            push!(norm_Rx, maximum(abs.(stokes.R.Rx)))
            push!(norm_Ry, maximum(abs.(stokes.R.Ry)))
            push!(norm_Rz, maximum(abs.(stokes.R.Rz)))
            push!(norm_∇V, maximum(abs.(stokes.R.RP)))
            err = max(norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && (verbose  || iter == iterMax)
                @printf(
                    "iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_Rz[cont],
                    norm_∇V[cont]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    av_time = wtime0 / (iter - 1) # average time per iteration
    update_τ_o!(stokes) # copy τ into τ_o

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_Rz=norm_Rz,
        norm_∇V=norm_∇V,
        time=wtime0,
        av_time=av_time,
    )
end

# GeoParams and multiple phases
function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,3},
    thermal::ThermalArrays,
    pt_stokes::PTStokesCoeffs,
    di::NTuple{3,T},
    flow_bc::FlowBoundaryConditions,
    ρg,
    η,
    η_vep,
    phase_ratios::PhaseRatio,
    rheology::NTuple{N, AbstractMaterialParamsStruct},
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {A,B,C,D,T,N}

    ## UNPACK

    # solver related
    ϵ = pt_stokes.ϵ
    # geometry
    _di = @. 1 / di
    ni = nx, ny, nz = size(stokes.P)
    # z = LinRange(di[3]*0.5, 1.0-di[3]*0.5, nz)

    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end
    @parallel (1:ny, 1:nz) free_slip_x!(ητ)
    @parallel (1:nx, 1:nz) free_slip_y!(ητ)
    @parallel (1:nx, 1:ny) free_slip_z!(ητ)

    # errors
    err = 2 * ϵ
    iter = 0
    cont = 0
    err_evo1 = Float64[]
    err_evo2 = Int64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_Rz = Float64[]
    norm_∇V = Float64[]

    # Kb = get_Kb(rheology)
    # G  = get_G(rheology)
    @copy stokes.P0 stokes.P
    # λ = @zeros(ni...)

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin

            # Update viscosity
            args_ηv = (; T = thermal.T, P = stokes.P, dt=Inf)
            ν = iter > 1 ? 0.5 : 1.0
            @parallel (@idx ni) compute_viscosity!(η, ν, phase_ratios.center, @strain(stokes)..., args_ηv, rheology)
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_maxloc!(ητ, η)
                update_halo!(ητ)
            end
            @parallel (1:ny, 1:nz) free_slip_x!(ητ)
            @parallel (1:nx, 1:nz) free_slip_y!(ητ)
            @parallel (1:nx, 1:ny) free_slip_z!(ητ)

            # Update buoyancy
            @parallel (@idx ni) compute_ρg!(ρg[3], phase_ratios.center, rheology, (T=thermal.T, P=stokes.P))
         
            @parallel (@idx ni) compute_∇V!(
                stokes.∇V, @velocity(stokes)..., _di...
            )
            @parallel (@idx ni) compute_P!(
                stokes.P,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                η,
                rheology,
                phase_ratios.center,
                dt,
                pt_stokes.r,
                pt_stokes.θ_dτ,
            )
            @parallel (@idx ni) compute_strain_rate!(
                stokes.∇V,
                @strain(stokes)...,
                @velocity(stokes)...,
                _di...,
            )
            @parallel (@idx ni) compute_τ_new!(
                @stress_center(stokes)...,
                stokes.τ.II,
                @tensor(stokes.τ_o)...,
                @strain(stokes)...,
                stokes.P,
                η,
                η_vep,
                phase_ratios.center,
                rheology, # needs to be a tuple
                dt,
                pt_stokes.θ_dτ,
            )
            @parallel center2vertex!(stokes.τ.yz, stokes.τ.yz_c)
            @parallel center2vertex!(stokes.τ.xz, stokes.τ.xz_c)
            @parallel center2vertex!(stokes.τ.xy, stokes.τ.xy_c)

            # @parallel (@idx ni) compute_τ_gp!(
            #     @stress_center(stokes)...,
            #     stokes.τ.II,
            #     @tensor(stokes.τ_o)...,
            #     @strain(stokes)...,
            #     η,
            #     η_vep,
            #     args_η,
            #     thermal.T,
            #     tupleize(rheology), # needs to be a tuple
            #     dt,
            #     pt_stokes.θ_dτ,
            # )
            # @parallel (@idx ni.+1) compute_τ_vertex!(
            #     @shear(stokes.τ)...,
            #     @shear(stokes.τ_o)...,
            #     @shear(stokes.ε)...,
            #     η_vep,
            #     G,
            #     dt,
            #     pt_stokes.θ_dτ,
            # )
            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    @velocity(stokes)...,
                    stokes.R.Rx,
                    stokes.R.Ry,
                    stokes.R.Rz,
                    stokes.P,
                    ρg...,
                    @stress(stokes)...,
                    ητ,
                    pt_stokes.ηdτ,
                    _di...,
                )
                update_halo!(@velocity(stokes)...)
            end
            # apply_free_slip!(flow_bc, stokes.V.Vx, stokes.V.Vy, stokes.V.Vz)
            flow_bcs!(stokes, flow_bc, di)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            push!(norm_Rx, maximum(abs.(stokes.R.Rx)))
            push!(norm_Ry, maximum(abs.(stokes.R.Ry)))
            push!(norm_Rz, maximum(abs.(stokes.R.Rz)))
            push!(norm_∇V, maximum(abs.(stokes.R.RP)))
            err = max(norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont])
            push!(err_evo1, err)
            push!(err_evo2, iter)
            if igg.me == 0 && (verbose  || iter == iterMax)
                @printf(
                    "iter = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_Rz=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_Rz[cont],
                    norm_∇V[cont]
                )
            end
            isnan(err) && error("NaN(s)")
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations converged in $iter iterations")
        end
    end

    av_time = wtime0 / (iter - 1) # average time per iteration
    update_τ_o!(stokes) # copy τ into τ_o

    return (
        iter=iter,
        err_evo1=err_evo1,
        err_evo2=err_evo2,
        norm_Rx=norm_Rx,
        norm_Ry=norm_Ry,
        norm_Rz=norm_Rz,
        norm_∇V=norm_∇V,
        time=wtime0,
        av_time=av_time,
    )
end


@parallel_indices (i, j, k) function compute_τ_new!(
    τxx,
    τyy,
    τzz,
    τyz,
    τxz,
    τxy,
    TII,
    τxx_old,
    τyy_old,
    τzz_old,
    τyzv_old,
    τxzv_old,
    τxyv_old,
    εxx,
    εyy,
    εzz,
    εyzv,
    εxzv,
    εxyv,
    P,
    η,
    η_vep,
    phase_ratios,
    rheology,
    dt,
    θ_dτ,
)

    # convinience closures
    gather_yz(A) = _gather_yz(A, i, j, k)
    gather_xz(A) = _gather_xz(A, i, j, k)
    gather_xy(A) = _gather_xy(A, i, j, k)
    
    Val6 = Val(6)
    εII_0 = (εxx[i, j, k] == 0 && εyy[i, j, k] == 0 && εzz[i, j, k] == 0) ? 1e-15 : 0.0
    # @inbounds begin
        G           = fn_ratio(get_G, rheology, phase_ratios[i, j, k])
        _Gdt        = inv(G * dt)
        ηij         = η[i, j, k]
        η_e         = ηij * _Gdt
        dτ_r        = inv(θ_dτ + η_e + 1.0) # original
        # cache tensors
        εij_p   = (
            εxx[i, j, k] + εII_0, εyy[i, j, k] * εII_0 * 0.5, εzz[i, j, k] * εII_0 * 0.5, 
            0.125 * sum(gather_yz(εyzv)), 0.125 * sum(gather_xz(εxzv)), 0.125 * sum(gather_xy(εxyv))
        )
        τij_p_o = (
            τxx_old[i, j, k], τyy_old[i, j, k], τzz_old[i, j, k], 
            0.125 * sum(gather_yz(τyzv_old)), 0.125 * sum(gather_xz(τxzv_old)), 0.125 * sum(gather_xy(τxyv_old))
        )
        τij     = 
            τxx[i, j, k], τyy[i, j, k], τzz[i, j, k], τyz[i, j, k],  τxz[i, j, k],  τxy[i, j, k]

        # Stress increment
        dτ = ntuple(Val6) do i 
            dτ_r * (-(τij[i] - τij_p_o[i]) * η_e - τij[i] + 2.0 * ηij * (εij_p[i]))
        end
        τII_trial = second_invariant((dτ .+ τij)...)

        # plastic parameters
        is_pl, C, sinϕ, η_reg = plastic_params_phase(rheology, phase_ratios[i, j, k])
        # yield stress
        τy = C + P[i, j, k] * sinϕ 

        if is_pl && τII_trial > abs(τy) 
            # yield function
            F               = τII_trial - C - P[i, j, k] * sinϕ
            λ = λ0[i, j, k] = (F>0.0) * F * inv((inv(inv(ηij) + _Gdt) + η_reg) )

            # Partials of plastic potential
            λ_τII_trial = 0.5 * inv(τII_trial) * λ
            λdQdτ = ntuple(Val6) do i 
                (τij[i] + dτ[i]) * λ_τII_trial
            end

            # corrected stress
            dτ_pl = ntuple(Val6) do i 
                dτ_r * (-(τij[i] - τij_p_o[i]) * η_e - τij[i] + 2.0 * ηij * (εij_p[i] - λdQdτ[i]))
            end
            
            # update stress 
            τxx[i, j, k] += dτ_pl[1]
            τyy[i, j, k] += dτ_pl[2]
            τzz[i, j, k] += dτ_pl[3]
            τyz[i, j, k] += dτ_pl[4]
            τxz[i, j, k] += dτ_pl[5]
            τxy[i, j, k] += dτ_pl[6]

        else
            # update stress 
            τxx[i, j, k] += dτ[1]
            τyy[i, j, k] += dτ[2]
            τzz[i, j, k] += dτ[3]
            τyz[i, j, k] += dτ[4]
            τxz[i, j, k] += dτ[5]
            τxy[i, j, k] += dτ[6]
        end
        
        # visco-elastic strain rates
        ε_ve = ntuple(Val6) do i 
            εij_p[i] + 0.5 * τij_p_o[i] * _Gdt
        end

        εII_ve         = second_invariant(ε_ve...)
        τII            = second_invariant(τxx[i, j, k], τyy[i, j, k], τzz[i, j, k], τyz[i, j, k], τxz[i, j, k], τxy[i, j, k])
        η_vep[i, j, k] = τII * 0.5 * inv(εII_ve)
        TII[i,j,k]     = τII
    # end
    
    return nothing
end


## HELPER FUNCTIONS ---------------------------

@inline Base.@propagate_inbounds _gather_yz(A, i, j, k) = A[i, j, k], A[i    , j + 1, k], A[i, j    , k + 1], A[i    , j + 1, k + 1]
@inline Base.@propagate_inbounds _gather_xz(A, i, j, k) = A[i, j, k], A[i + 1, j    , k], A[i, j    , k + 1], A[i + 1, j    , k + 1]
@inline Base.@propagate_inbounds _gather_xy(A, i, j, k) = A[i, j, k], A[i + 1, j    , k], A[i, j + 1, k    ], A[i + 1, j + 1, k    ]
@inline Base.@propagate_inbounds _av(T, i, j, k) = 0.125 * (
    T[i, j, k  ] + T[i, j+1, k  ] + T[i+1, j, k  ] + T[i+1, j+1, k  ] +
    T[i, j, k+1] + T[i, j+1, k+1] + T[i+1, j, k+1] + T[i+1, j+1, k+1]
)

# Viscosity with GeoParams

@parallel_indices (i, j, k) function compute_viscosity!(η, ν, ratios_center, εxx, εyy, εzz, εyzv, εxzv, εxyv, args, rheology)

    # convinience closures
    av_T()       = _av(args.T, i, j, k)
    gather_yz(A) = _gather_yz(A, i, j, k)
    gather_xz(A) = _gather_xz(A, i, j, k)
    gather_xy(A) = _gather_xy(A, i, j, k)
    
    εII_0 = (εxx[i, j, k] == 0 && εyy[i, j, k] == 0 && εzz[i, j, k] == 0) ? 1e-15 : 0.0
    _zeros = (0.0, 0.0, 0.0, 0.0)
    ratio_ij = ratios_center[i, j, k]
    args_ij  = (; dt = args.dt, P = args.P[i, j, k],  T=av_T(), τII_old=0.0)
    εij_p = (
        εxx[i, j, k] + εII_0, 
        εyy[i, j, k] - εII_0 * 0.5, 
        εzz[i, j, k] - εII_0 * 0.5, 
        gather_yz(εyzv), 
        gather_xz(εxzv), 
        gather_xy(εxyv)
    )
    τij_p_o  = (
        0.0, 
        0.0, 
        0.0, 
        _zeros,
        _zeros,
        _zeros
    )
    # update stress and effective viscosity
    _, _, ηi   = compute_τij_ratio(rheology, ratio_ij, εij_p, args_ij, τij_p_o)
    ηi         = exp((1-ν)*log(η[i, j, k]) + ν*log(ηi))
    η[i, j, k] = clamp(2*ηi, 1e16, 1e24)
    
    return nothing
end

function compute_τij_ratio(rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio, εij_p, args_ij, τij_p_o) where N
    data = compute_phase_τij(rheology, ratio, εij_p, args_ij, τij_p_o)
    # average over phases
    τij =  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    τII = 0.0
    η_eff = 0.0
    for n in 1:N
        τij    = @. data[n][1] * ratio[n] + τij
        τII   +=    data[n][2] * ratio[n]
        η_eff +=    data[n][3] * ratio[n]
    end
    return τij, τII, η_eff
end

@generated function compute_phase_τij(rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio, εij_p, args_ij, τij_p_o) where N
    quote
        Base.@_inline_meta 
        empty_args = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0, 0.0
        Base.@nexprs $N i -> a_i = ratio[i] == 0 ? empty_args : compute_τij(rheology[i].CompositeRheology[1], εij_p, args_ij, τij_p_o) 
        Base.@ncall $N tuple a
    end
end

# Buoyancy forces with GeoParams

@parallel_indices (i, j, k) function compute_ρg!(ρg, phase_ratios, rheology, args)

    av_T() = _av(args.T, i, j, k) - 273.0

    ρg[i, j, k] = -compute_density_ratio(phase_ratios[i, j, k], rheology, (; T = av_T(), P=args.P[i, j, k])) *
        compute_gravity(rheology[1])
    return nothing
end

# Extract plastic parameters from `rheology`

function plastic_params_phase(rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio) where N
    data = _plastic_params_phase(rheology, ratio)
    # average over phases
    is_pl = false
    C = 0.0
    sinϕ = 0.0
    η_reg = 0.0
    for n in 1:N
        data[n][1] && (is_pl = true)
        C += data[n][2] * ratio[n]
        sinϕ += data[n][3] * ratio[n]
        η_reg += data[n][4] * ratio[n]
    end
    return is_pl, C, sinϕ, η_reg
end

@generated function plastic_params(v::NTuple{N, Any}) where N
    quote
        Base.@_inline_meta
        Base.@nexprs $N i -> isplastic(v[i]) && return true, v[i].C.val, v[i].sinϕ.val, v[i].η_vp.val
        (false, 0.0, 0.0, 0.0)
    end
end

@inline isplastic(x::AbstractPlasticity) = true
@inline isplastic(x) = false
@inline plastic_params(v) = plastic_params(v.CompositeRheology[1].elements)

@generated function _plastic_params_phase(rheology::NTuple{N,AbstractMaterialParamsStruct}, ratio) where N
    quote
        Base.@_inline_meta 
        empty_args =  (false, 0.0, 0.0, 0.0)
        Base.@nexprs $N i -> a_i = ratio[i] == 0 ? empty_args : plastic_params(rheology[i]) 
        Base.@ncall $N tuple a
    end
end

end # END OF MODULE

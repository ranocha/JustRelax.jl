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

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using JustRelax
using LinearAlgebra
using CUDA
using Printf

# using ..JustRelax: solve!
import JustRelax: stress, strain, elastic_iter_params!, PTArray, Velocity, SymmetricTensor
import JustRelax: Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic
import JustRelax: compute_maxloc!, solve!

import ..Stokes2D: compute_P!, compute_V!, compute_strain_rate!

export solve!

## 2D ELASTIC KERNELS

@parallel function compute_dV_elastic!(
    dVx::AbstractArray{T,2},
    dVy::AbstractArray{T,2},
    P::AbstractArray{T,2},
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    dτ_Rho::AbstractArray{T,2},
    ρg::AbstractArray{T,2},
    _dx::T,
    _dy::T,
) where {T}
    @all(dVx) = (@d_xi(τxx) * _dx + @d_ya(τxy) * _dy - @d_xi(P) * _dx) * @harm_xi(dτ_Rho)
    @all(dVy) =
        (@d_yi(τyy) * _dy + @d_xa(τxy) * _dx - @d_yi(P) * _dy - @harm_yi(ρg)) *
        @harm_yi(dτ_Rho)
    return nothing
end

@parallel function compute_Res!(
    Rx::AbstractArray{T,2},
    Ry::AbstractArray{T,2},
    P::AbstractArray{T,2},
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    ρg::AbstractArray{T,2},
    dx::T,
    dy::T,
) where {T}
    @all(Rx) = @d_xi(τxx) / dx + @d_ya(τxy) / dy - @d_xi(P) / dx
    @all(Ry) = @d_yi(τyy) / dy + @d_xa(τxy) / dx - @d_yi(P) / dy - @harm_yi(ρg)
    return nothing
end

@parallel function update_τ_o!(
    τxx_o::AbstractArray{T,2},
    τyy_o::AbstractArray{T,2},
    τxy_o::AbstractArray{T,2},
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
) where {T}
    @all(τxx_o) = @all(τxx)
    @all(τyy_o) = @all(τyy)
    @all(τxy_o) = @all(τxy)
    return nothing
end

function update_τ_o!(stokes::StokesArrays{ViscoElastic,A,B,C,D,2}) where {A,B,C,D}
    τ, τ_o = stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    @parallel update_τ_o!(τxx_o, τyy_o, τxy_o, τxx, τyy, τxy)
end

macro Gr()
    return esc(:(@all(Gdτ) / (G * dt)))
end
macro av_Gr()
    return esc(:(@av(Gdτ) / (G * dt)))
end
macro harm_Gr()
    return esc(:(@harm(Gdτ) / (G * dt)))
end
@parallel function compute_τ!(
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    τxx_o::AbstractArray{T,2},
    τyy_o::AbstractArray{T,2},
    τxy_o::AbstractArray{T,2},
    Gdτ::AbstractArray{T,2},
    εxx::AbstractArray{T,2},
    εyy::AbstractArray{T,2},
    εxy::AbstractArray{T,2},
    η::AbstractArray{T,2},
    G::T,
    dt::T,
) where {T}
    @all(τxx) =
        (@all(τxx) + @all(τxx_o) * @Gr() + T(2) * @all(Gdτ) * @all(εxx)) /
        (one(T) + @all(Gdτ) / @all(η) + @Gr())
    @all(τyy) =
        (@all(τyy) + @all(τyy_o) * @Gr() + T(2) * @all(Gdτ) * @all(εyy)) /
        (one(T) + @all(Gdτ) / @all(η) + @Gr())
    @all(τxy) =
        (@all(τxy) + @all(τxy_o) * @harm_Gr() + T(2) * @harm(Gdτ) * @all(εxy)) /
        (one(T) + @harm(Gdτ) / @harm(η) + @harm_Gr())
    return nothing
end


## 2D VISCO-ELASTIC STOKES SOLVER 

function JustRelax.solve!(
    stokes::StokesArrays{ViscoElastic,A,B,C,D,2},
    pt_stokes::PTStokesCoeffs,
    di::NTuple{2,T},
    li::NTuple{2,T},
    max_li,
    freeslip,
    ρg,
    η,
    G,
    dt;
    iterMax=10e3,
    nout=500,
    verbose=true,
) where {A,B,C,D,T}

    # unpack
    dx, dy = di
    _dx, _dy = inv.(di)
    lx, ly = li
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    εxx, εyy, εxy = strain(stokes)
    τ, τ_o = stress(stokes)
    τxx, τyy, τxy = τ
    τxx_o, τyy_o, τxy_o = τ_o
    P, ∇V = stokes.P, stokes.∇V
    Rx, Ry = stokes.R.Rx, stokes.R.Ry
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ,
    pt_stokes.dτ_Rho, pt_stokes.ϵ, pt_stokes.Re, pt_stokes.r,
    pt_stokes.Vpdτ

    # ~preconditioner
    ητ = deepcopy(η)
    @parallel compute_maxloc!(ητ, η)
    # PT numerical coefficients
    @parallel elastic_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li)

    _sqrt_leng_Rx = one(T) / sqrt(length(Rx))
    _sqrt_leng_Ry = one(T) / sqrt(length(Ry))
    _sqrt_leng_∇V = one(T) / sqrt(length(∇V))

    # errors
    err = 2 * ϵ
    iter = 0
    cont = 0
    err_evo1 = Float64[]
    err_evo2 = Float64[]
    norm_Rx = Float64[]
    norm_Ry = Float64[]
    norm_∇V = Float64[]

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel compute_strain_rate!(εxx, εyy, εxy, Vx, Vy, _dx, _dy)
            @parallel compute_P!(∇V, P, εxx, εyy, Gdτ, r)
            @parallel compute_τ!(
                τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Gdτ, εxx, εyy, εxy, η, G, dt
            )
            @parallel compute_dV_elastic!(dVx, dVy, P, τxx, τyy, τxy, dτ_Rho, ρg, _dx, _dy)
            @parallel compute_V!(Vx, Vy, dVx, dVy)

            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy)
        end

        iter += 1
        if iter % nout == 0 && iter > 1
            cont += 1
            wtime0 += @elapsed begin
                @parallel compute_Res!(Rx, Ry, P, τxx, τyy, τxy, ρg, dx, dy)
            end
            Vmin, Vmax = minimum(Vy), maximum(Vy)
            Pmin, Pmax = minimum(P), maximum(P)
            push!(norm_Rx, norm(Rx) / (Pmax - Pmin) * lx * _sqrt_leng_Rx)
            push!(norm_Ry, norm(Ry) / (Pmax - Pmin) * lx * _sqrt_leng_Ry)
            push!(norm_∇V, norm(∇V) / (Vmax - Vmin) * lx * _sqrt_leng_∇V)
            err = maximum([norm_Rx[cont], norm_Ry[cont], norm_∇V[cont]])
            push!(err_evo1, maximum([norm_Rx[cont], norm_Ry[cont], norm_∇V[cont]]))
            push!(err_evo2, iter)
            if verbose && (err < ϵ) || (iter == iterMax)
                @printf(
                    "Total steps = %d, err = %1.3e [norm_Rx=%1.3e, norm_Ry=%1.3e, norm_∇V=%1.3e] \n",
                    iter,
                    err,
                    norm_Rx[cont],
                    norm_Ry[cont],
                    norm_∇V[cont]
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

end # END OF MODULE

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
    stress, elastic_iter_params!, PTArray, Velocity, SymmetricTensor, pureshear_bc!
import JustRelax:
    Residual, StokesArrays, PTStokesCoeffs, AbstractStokesModel, ViscoElastic, IGG
import JustRelax: compute_maxloc!, solve!
# , second_invariant!, computeViscosity!, islinear, unpack_tensor3d

export solve!, pureshear_bc!

@parallel_indices (i, j, k) function update_τ_o!(
    τxx_o::AbstractArray{T,3},
    τyy_o::AbstractArray{T,3},
    τzz_o::AbstractArray{T,3},
    τxy_o::AbstractArray{T,3},
    τxz_o::AbstractArray{T,3},
    τyz_o::AbstractArray{T,3},
    τxx::AbstractArray{T,3},
    τyy::AbstractArray{T,3},
    τzz::AbstractArray{T,3},
    τxy::AbstractArray{T,3},
    τxz::AbstractArray{T,3},
    τyz::AbstractArray{T,3},
) where {T}
    if (i ≤ size(τxx, 1) && j ≤ size(τxx, 2) && k ≤ size(τxx, 3))
        τxx_o[i, j, k] = τxx[i, j, k]
    end
    if (i ≤ size(τyy, 1) && j ≤ size(τyy, 2) && k ≤ size(τyy, 3))
        τyy_o[i, j, k] = τyy[i, j, k]
    end
    if (i ≤ size(τzz, 1) && j ≤ size(τzz, 2) && k ≤ size(τzz, 3))
        τzz_o[i, j, k] = τzz[i, j, k]
    end
    if (i ≤ size(τxy, 1) && j ≤ size(τxy, 2) && k ≤ size(τxy, 3))
        τxy_o[i, j, k] = τxy[i, j, k]
    end
    if (i ≤ size(τxz, 1) && j ≤ size(τxz, 2) && k ≤ size(τxz, 3))
        τxz_o[i, j, k] = τxz[i, j, k]
    end
    if (i ≤ size(τyz, 1) && j ≤ size(τyz, 2) && k ≤ size(τyz, 3))
        τyz_o[i, j, k] = τyz[i, j, k]
    end
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

@parallel_indices (i, j, k) function compute_strain_rate!(
    εxx::AbstractArray{T,3},
    εyy::AbstractArray{T,3},
    εzz::AbstractArray{T,3},
    εyz::AbstractArray{T,3},
    εxz::AbstractArray{T,3},
    εxy::AbstractArray{T,3},
    Vx::AbstractArray{T,3},
    Vy::AbstractArray{T,3},
    Vz::AbstractArray{T,3},
    _dx::T,
    _dy::T,
    _dz::T,
) where {T}
    # Compute ε_xx
    if (i ≤ size(εxx, 1) && j ≤ size(εxx, 2) && k ≤ size(εxx, 3))
        εxx[i, j, k] = _dx * (Vx[i + 1, j + 1, k + 1] - Vx[i, j + 1, k + 1])
    end
    # Compute ε_yy
    if (i ≤ size(εyy, 1) && j ≤ size(εyy, 2) && k ≤ size(εyy, 3))
        εyy[i, j, k] = _dy * (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j, k + 1])
    end
    # Compute ε_zz
    if (i ≤ size(εzz, 1) && j ≤ size(εzz, 2) && k ≤ size(εzz, 3))
        εzz[i, j, k] = _dz * (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j + 1, k])
    end
    # Compute ε_xy
    if (i ≤ size(εxy, 1) && j ≤ size(εxy, 2) && k ≤ size(εxy, 3))
        εxy[i, j, k] =
            0.5 * (
                _dy * (Vx[i + 1, j + 1, k + 1] - Vx[i + 1, j, k + 1]) +
                _dx * (Vy[i + 1, j + 1, k + 1] - Vy[i, j + 1, k + 1])
            )
    end
    # Compute ε_xz
    if (i ≤ size(εxz, 1) && j ≤ size(εxz, 2) && k ≤ size(εxz, 3))
        εxz[i, j, k] =
            0.5 * (
                _dz * (Vx[i + 1, j + 1, k + 1] - Vx[i + 1, j + 1, k]) +
                _dx * (Vz[i + 1, j + 1, k + 1] - Vz[i, j + 1, k + 1])
            )
    end
    # Compute ε_yz
    if (i ≤ size(εyz, 1) && j ≤ size(εyz, 2) && k ≤ size(εyz, 3))
        εyz[i, j, k] =
            0.5 * (
                _dz * (Vy[i + 1, j + 1, k + 1] - Vy[i + 1, j + 1, k]) +
                _dy * (Vz[i + 1, j + 1, k + 1] - Vz[i + 1, j, k + 1])
            )
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_P_τ!(
    P::AbstractArray{T,3},
    τxx::AbstractArray{T,3},
    τyy::AbstractArray{T,3},
    τzz::AbstractArray{T,3},
    τxy::AbstractArray{T,3},
    τxz::AbstractArray{T,3},
    τyz::AbstractArray{T,3},
    τxx_o::AbstractArray{T,3},
    τyy_o::AbstractArray{T,3},
    τzz_o::AbstractArray{T,3},
    τxy_o::AbstractArray{T,3},
    τxz_o::AbstractArray{T,3},
    τyz_o::AbstractArray{T,3},
    εxx::AbstractArray{T,3},
    εyy::AbstractArray{T,3},
    εzz::AbstractArray{T,3},
    εyz::AbstractArray{T,3},
    εxz::AbstractArray{T,3},
    εxy::AbstractArray{T,3},
    Vx::AbstractArray{T,3},
    Vy::AbstractArray{T,3},
    Vz::AbstractArray{T,3},
    η::AbstractArray{T,3},
    Gdτ::AbstractArray{T,3},
    r::T,
    G::T,
    dt::M,
    _dx::T,
    _dy::T,
    _dz::T,
) where {T,M}
    # closures ---------------------------------------------
    @inline _inn_yz_Gdτ(i, j, k) = Gdτ[i, j + 1, k + 1]
    @inline _inn_xz_Gdτ(i, j, k) = Gdτ[i + 1, j, k + 1]
    @inline _inn_xy_Gdτ(i, j, k) = Gdτ[i + 1, j + 1, k]
    @inline _inn_yz_η(i, j, k) = η[i, j + 1, k + 1]
    @inline _inn_xz_η(i, j, k) = η[i + 1, j, k + 1]
    @inline _inn_xy_η(i, j, k) = η[i + 1, j + 1, k]
    @inline function _av_xyi_Gdτ(i, j, k)
        @inbounds (
            Gdτ[i, j, k + 1] +
            Gdτ[i + 1, j, k + 1] +
            Gdτ[i, j + 1, k + 1] +
            Gdτ[i + 1, j + 1, k + 1]
        ) * 0.25
    end
    @inline function _av_xzi_Gdτ(i, j, k)
        @inbounds (
            Gdτ[i, j + 1, k] +
            Gdτ[i + 1, j + 1, k] +
            Gdτ[i, j + 1, k + 1] +
            Gdτ[i + 1, j + 1, k + 1]
        ) * 0.25
    end
    @inline function _av_yzi_Gdτ(i, j, k)
        @inbounds (
            Gdτ[i + 1, j, k] +
            Gdτ[i + 1, j + 1, k] +
            Gdτ[i + 1, j, k + 1] +
            Gdτ[i + 1, j + 1, k + 1]
        ) * 0.25
    end
    @inline function _harm_xyi_η(i, j, k)
        @inbounds 1.0 / (
            1.0 / η[i, j, k + 1] +
            1.0 / η[i + 1, j, k + 1] +
            1.0 / η[i, j + 1, k + 1] +
            1.0 / η[i + 1, j + 1, k + 1]
        ) * 0.25
    end
    @inline function _harm_xzi_η(i, j, k)
        @inbounds 1.0 / (
            1.0 / η[i, j + 1, k] +
            1.0 / η[i + 1, j + 1, k] +
            1.0 / η[i, j + 1, k + 1] +
            1.0 / η[i + 1, j + 1, k + 1]
        ) * 0.25
    end
    @inline function _harm_yzi_η(i, j, k)
        @inbounds 1.0 / (
            1.0 / η[i + 1, j, k] +
            1.0 / η[i + 1, j + 1, k] +
            1.0 / η[i + 1, j, k + 1] +
            1.0 / η[i + 1, j + 1, k + 1]
        ) * 0.25
    end
    # ------------------------------------------------------

    # Compute pressure
    if (i ≤ size(P, 1) && j ≤ size(P, 2) && k ≤ size(P, 3))
        P[i, j, k] =
            P[i, j, k] -
            r *
            Gdτ[i, j, k] *
            (
                _dx * (Vx[i + 1, j, k] - Vx[i, j, k]) +
                _dy * (Vy[i, j + 1, k] - Vy[i, j, k]) +
                _dz * (Vz[i, j, k + 1] - Vz[i, j, k])
            )
    end
    # Compute τ_xx
    if (i ≤ size(τxx, 1) && j ≤ size(τxx, 2) && k ≤ size(τxx, 3))
        τxx[i, j, k] =
            (
                τxx[i, j, k] / _inn_yz_Gdτ(i, j, k) +
                τxx_o[i, j, k] / G / dt +
                T(2) * εxx[i, j, k]
            ) / (one(T) / _inn_yz_Gdτ(i, j, k) + one(T) / _inn_yz_η(i, j, k))
    end
    # Compute τ_yy
    if (i ≤ size(τyy, 1) && j ≤ size(τyy, 2) && k ≤ size(τyy, 3))
        τyy[i, j, k] =
            (
                τyy[i, j, k] / _inn_xz_Gdτ(i, j, k) +
                τyy_o[i, j, k] / G / dt +
                T(2) * εyy[i, j, k]
            ) / (one(T) / _inn_xz_Gdτ(i, j, k) + one(T) / _inn_xz_η(i, j, k))
    end
    # Compute τ_zz
    if (i ≤ size(τzz, 1) && j ≤ size(τzz, 2) && k ≤ size(τzz, 3))
        τzz[i, j, k] =
            (
                τzz[i, j, k] / _inn_xy_Gdτ(i, j, k) +
                τzz_o[i, j, k] / G / dt +
                T(2) * εzz[i, j, k]
            ) / (one(T) / _inn_xy_Gdτ(i, j, k) + one(T) / _inn_xy_η(i, j, k))
    end
    # Compute τ_xy
    if (i ≤ size(τxy, 1) && j ≤ size(τxy, 2) && k ≤ size(τxy, 3))
        τxy[i, j, k] =
            (
                τxy[i, j, k] / _av_xyi_Gdτ(i, j, k) +
                τxy_o[i, j, k] / G / dt +
                T(2) * εxy[i, j, k]
            ) / (one(T) / _av_xyi_Gdτ(i, j, k) + one(T) / _harm_xyi_η(i, j, k))
    end
    # Compute τ_xz
    if (i ≤ size(τxz, 1) && j ≤ size(τxz, 2) && k ≤ size(τxz, 3))
        τxz[i, j, k] =
            (
                τxz[i, j, k] / _av_xzi_Gdτ(i, j, k) +
                τxz_o[i, j, k] / G / dt +
                T(2) * εxz[i, j, k]
            ) /  (one(T) / _av_xzi_Gdτ(i, j, k) + one(T) / _harm_xzi_η(i, j, k))
    end
    # Compute τ_yz
    if (i ≤ size(τyz, 1) && j ≤ size(τyz, 2) && k ≤ size(τyz, 3))
        τyz[i, j, k] =
            (
                τyz[i, j, k] / _av_yzi_Gdτ(i, j, k) +
                τyz_o[i, j, k] / G / dt +
                T(2) * εyz[i, j, k]
            ) / (one(T) / _av_yzi_Gdτ(i, j, k) + one(T) / _harm_yzi_η(i, j, k))
    end
    return nothing
end

@inline function av_xi_ρg(f, i, j, k)
    return (f[i, j + 1, k + 1] + f[i + 1, j + 1, k + 1]) * 0.5
end
@inline function av_yi_ρg(f, i, j, k)
    return (f[i + 1, j, k + 1] + f[i + 1, j + 1, k + 1]) * 0.5
end
@inline function av_zi_ρg(f, i, j, k)
    return (f[i + 1, j + 1, k] + f[i + 1, j + 1, k + 1]) * 0.5
end

@parallel_indices (i, j, k) function compute_V!(
    Vx::AbstractArray{T,3},
    Vy::AbstractArray{T,3},
    Vz::AbstractArray{T,3},
    P::AbstractArray{T,3},
    fx::AbstractArray{T,3},
    fy::AbstractArray{T,3},
    fz::AbstractArray{T,3},
    τxx::AbstractArray{T,3},
    τyy::AbstractArray{T,3},
    τzz::AbstractArray{T,3},
    τxy::AbstractArray{T,3},
    τxz::AbstractArray{T,3},
    τyz::AbstractArray{T,3},
    dτ_Rho::AbstractArray{T,3},
    _dx::T,
    _dy::T,
    _dz::T,
    nx_1::N,
    nx_2::N,
    ny_1::N,
    ny_2::N,
    nz_1::N,
    nz_2::N,
) where {T,N}

    # closures ---------------------------------------------
    _av_xi_ρg = (i, j, k) -> av_xi_ρg(fx, i, j, k)
    _av_yi_ρg = (i, j, k) -> av_yi_ρg(fy, i, j, k)
    _av_zi_ρg = (i, j, k) -> av_zi_ρg(fz, i, j, k)

    @inline function _av_xi_dτ_Rho(i, j, k)
        (dτ_Rho[i, j + 1, k + 1] + dτ_Rho[i + 1, j + 1, k + 1]) * 0.5
    end
    @inline function _av_yi_dτ_Rho(i, j, k)
        (dτ_Rho[i + 1, j, k + 1] + dτ_Rho[i + 1, j + 1, k + 1]) * 0.5
    end
    @inline function _av_zi_dτ_Rho(i, j, k)
        (dτ_Rho[i + 1, j + 1, k] + dτ_Rho[i + 1, j + 1, k + 1]) * 0.5
    end
    # ------------------------------------------------------

    if (i ≤ nx_1) && (j ≤ ny_2) && (k ≤ nz_2)
        Vx[i + 1, j + 1, k + 1] = 
            Vx[i + 1, j + 1, k + 1] +
            (
                _dx * (τxx[i + 1, j, k] - τxx[i, j, k]) +
                _dy * (τxy[i, j + 1, k] - τxy[i, j, k]) +
                _dz * (τxz[i, j, k + 1] - τxz[i, j, k]) -
                _dx * (P[i + 1, j + 1, k + 1] - P[i, j + 1, k + 1]) +
                _av_xi_ρg(i, j, k)
            ) * _av_xi_dτ_Rho(i, j, k)
    end
    if (i ≤ nx_2) && (j ≤ ny_1) && (k ≤ nz_2)
        Vy[i + 1, j + 1, k + 1] =
            Vy[i + 1, j + 1, k + 1] +
            (
                _dy * (τyy[i, j + 1, k] - τyy[i, j, k]) +
                _dx * (τxy[i + 1, j, k] - τxy[i, j, k]) +
                _dz * (τyz[i, j, k + 1] - τyz[i, j, k]) -
                _dy * (P[i + 1, j + 1, k + 1] - P[i + 1, j, k + 1]) +
                _av_yi_ρg(i, j, k)
            ) * _av_yi_dτ_Rho(i, j, k)
    end
    if (i ≤ nx_2) && (j ≤ ny_2) && (k ≤ nz_1)
        Vz[i + 1, j + 1, k + 1] =
            Vz[i + 1, j + 1, k + 1] +
            (
                _dz * (τzz[i, j, k + 1] - τzz[i, j, k]) +
                _dx * (τxz[i + 1, j, k] - τxz[i, j, k]) +
                _dy * (τyz[i, j + 1, k] - τyz[i, j, k]) -
                _dz * (P[i + 1, j + 1, k + 1] - P[i + 1, j + 1, k]) +
                _av_zi_ρg(i, j, k)
            ) * _av_zi_dτ_Rho(i, j, k)
    end

    return nothing
end

@parallel_indices (i, j, k) function compute_Res!(
    ∇V::AbstractArray{T,3},
    Rx::AbstractArray{T,3},
    Ry::AbstractArray{T,3},
    Rz::AbstractArray{T,3},
    fx::AbstractArray{T,3},
    fy::AbstractArray{T,3},
    fz::AbstractArray{T,3},
    Vx::AbstractArray{T,3},
    Vy::AbstractArray{T,3},
    Vz::AbstractArray{T,3},
    P::AbstractArray{T,3},
    τxx::AbstractArray{T,3},
    τyy::AbstractArray{T,3},
    τzz::AbstractArray{T,3},
    τxy::AbstractArray{T,3},
    τxz::AbstractArray{T,3},
    τyz::AbstractArray{T,3},
    _dx::T,
    _dy::T,
    _dz::T,
) where {T}

    # closures ---------------------------------------------
    _av_xi_ρg = (i, j, k) -> av_xi_ρg(fx, i, j, k)
    _av_yi_ρg = (i, j, k) -> av_yi_ρg(fy, i, j, k)
    _av_zi_ρg = (i, j, k) -> av_zi_ρg(fz, i, j, k)
    # ------------------------------------------------------

    if (i ≤ size(∇V, 1)) && (j ≤ size(∇V, 2)) && (k ≤ size(∇V, 3))
        ∇V[i, j, k] =
            _dx * (Vx[i + 1, j, k] - Vx[i, j, k]) +
            _dy * (Vy[i, j + 1, k] - Vy[i, j, k]) +
            _dz * (Vz[i, j, k + 1] - Vz[i, j, k])
    end
    if (i ≤ size(Rx, 1)) && (j ≤ size(Rx, 2)) && (k ≤ size(Rx, 3))
        Rx[i, j, k] =
            _dx * (τxx[i + 1, j, k] - τxx[i, j, k]) +
            _dy * (τxy[i, j + 1, k] - τxy[i, j, k]) +
            _dz * (τxz[i, j, k + 1] - τxz[i, j, k]) -
            _dx * (P[i + 1, j + 1, k + 1] - P[i, j + 1, k + 1]) +
            _av_xi_ρg(i, j, k)
    end
    if (i ≤ size(Ry, 1)) && (j ≤ size(Ry, 2)) && (k ≤ size(Ry, 3))
        Ry[i, j, k] =
            _dy * (τyy[i, j + 1, k] - τyy[i, j, k]) +
            _dx * (τxy[i + 1, j, k] - τxy[i, j, k]) +
            _dz * (τyz[i, j, k + 1] - τyz[i, j, k]) -
            _dy * (P[i + 1, j + 1, k + 1] - P[i + 1, j, k + 1]) +
            _av_yi_ρg(i, j, k)
    end
    if (i ≤ size(Rz, 1)) && (j ≤ size(Rz, 2)) && (k ≤ size(Rz, 3))
        Rz[i, j, k] =
            _dz * (τzz[i, j, k + 1] - τzz[i, j, k]) +
            _dx * (τxz[i + 1, j, k] - τxz[i, j, k]) +
            _dy * (τyz[i, j + 1, k] - τyz[i, j, k]) -
            _dz * (P[i + 1, j + 1, k + 1] - P[i + 1, j + 1, k]) +
            _av_zi_ρg(i, j, k)
    end

    return nothing
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
    ni::NTuple{3,Integer},
    di::NTuple{3,T},
    li::NTuple{3,T},
    max_li,
    freeslip,
    ρg,
    η,
    G,
    dt,
    igg::IGG;
    iterMax=10e3,
    nout=500,
    b_width=(4, 4, 4),
    verbose=true,
) where {A,B,C,D,T}

    ## UNPACK
    # geometry
    dx, dy, dz = di
    _dx, _dy, _dz = @. 1 / di
    lx, ly, lz = li
    nx, ny, nz = ni
    nx_1, nx_2, ny_1, ny_2, nz_1, nz_2 = nx - 1, nx - 2, ny - 1, ny - 2, nz - 1, nz - 2
    # phsysics
    fx, fy, fz = ρg # gravitational forces
    Vx, Vy, Vz = stokes.V.Vx, stokes.V.Vy, stokes.V.Vz # velocity
    P, ∇V = stokes.P, stokes.∇V  # pressure and velociity divergence
    τ, τ_o = stress(stokes) # stress 
    τxx, τyy, τzz, τxy, τxz, τyz = τ
    τxx_o, τyy_o, τzz_o, τxy_o, τxz_o, τyz_o = τ_o
    εxx, εyy, εzz, εyz, εxz, εxy = unpack_tensor3d(ε)
    # solver related
    Rx, Ry, Rz = stokes.R.Rx, stokes.R.Ry, stokes.R.Rz
    Gdτ, dτ_Rho, ϵ, Re, r, Vpdτ = pt_stokes.Gdτ,
    pt_stokes.dτ_Rho, pt_stokes.ϵ, pt_stokes.Re, pt_stokes.r,
    pt_stokes.Vpdτ

    # ~preconditioner
    ητ = deepcopy(η)
    @hide_communication b_width begin # communication/computation overlap
        @parallel compute_maxloc!(ητ, η)
        update_halo!(ητ)
    end
    @parallel (1:size(ητ, 2), 1:size(ητ, 3)) free_slip_x!(ητ)
    @parallel (1:size(ητ, 1), 1:size(ητ, 3)) free_slip_y!(ητ)
    @parallel (1:size(ητ, 1), 1:size(ητ, 2)) free_slip_z!(ητ)

    # PT numerical coefficients
    @parallel elastic_iter_params!(dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li)

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

    # residual lengths
    _sqrt_len_Rx_g =
        1.0 / √(
            ((nx - 2 - 1) * igg.dims[1] + 2) *
            ((ny - 2 - 2) * igg.dims[2] + 2) *
            ((nz - 2 - 2) * igg.dims[3] + 2),
        )
    _sqrt_len_Ry_g =
        1.0 / √(
            ((nx - 2 - 2) * igg.dims[1] + 2) *
            ((ny - 2 - 1) * igg.dims[2] + 2) *
            ((nz - 2 - 2) * igg.dims[3] + 2),
        )
    _sqrt_len_Rz_g =
        1.0 / √(
            ((nx - 2 - 2) * igg.dims[1] + 2) *
            ((ny - 2 - 2) * igg.dims[2] + 2) *
            ((nz - 2 - 1) * igg.dims[3] + 2),
        )
    _sqrt_len_∇V_g =
        1.0 / √(
            ((nx - 2) * igg.dims[1] + 2) *
            ((ny - 2) * igg.dims[2] + 2) *
            ((nz - 2) * igg.dims[3] + 2),
        )

    # solver loop
    wtime0 = 0.0
    while iter < 2 || (err > ϵ && iter ≤ iterMax)
        wtime0 += @elapsed begin
            @parallel compute_strain_rate!(
                εxx,
                εyy,
                εzz,
                εyz,
                εxz,
                εxy,
                Vx,
                Vy,
                Vz,
                _dx,
                _dy,
                _dz,
            )
            @parallel compute_P_τ!(
                P,
                τxx,
                τyy,
                τzz,
                τxy,
                τxz,
                τyz,
                τxx_o,
                τyy_o,
                τzz_o,
                τxy_o,
                τxz_o,
                τyz_o,
                εxx,
                εyy,
                εzz,
                εyz,
                εxz,
                εxy,
                Vx,
                Vy,
                Vz,
                η,
                Gdτ,
                r,
                G,
                dt,
                _dx,
                _dy,
                _dz,
            )

            @hide_communication b_width begin # communication/computation overlap
                @parallel compute_V!(
                    Vx,
                    Vy,
                    Vz,
                    P,
                    fx,
                    fy,
                    fz,
                    τxx,
                    τyy,
                    τzz,
                    τxy,
                    τxz,
                    τyz,
                    dτ_Rho,
                    _dx,
                    _dy,
                    _dz,
                    nx_1,
                    nx_2,
                    ny_1,
                    ny_2,
                    nz_1,
                    nz_2,
                )
                update_halo!(Vx, Vy, Vz)
            end
            apply_free_slip!(freeslip, Vx, Vy, Vz)
        end

        iter += 1
        if iter % nout == 0 && iter > 5
            cont += 1

            wtime0 += @elapsed begin
                @parallel compute_Res!(
                    ∇V,
                    Rx,
                    Ry,
                    Rz,
                    fx,
                    fy,
                    fz,
                    Vx,
                    Vy,
                    Vz,
                    P,
                    τxx,
                    τyy,
                    τzz,
                    τxy,
                    τxz,
                    τyz,
                    _dx,
                    _dy,
                    _dz,
                )
            end

            Vmin, Vmax = minimum_mpi(Vx), maximum_mpi(Vx)
            Pmin, Pmax = minimum_mpi(P), maximum_mpi(P)
            push!(norm_Rx, norm_mpi(Rx) / (Pmax - Pmin) * lx * _sqrt_len_Rx_g)
            push!(norm_Ry, norm_mpi(Ry) / (Pmax - Pmin) * lx * _sqrt_len_Ry_g)
            push!(norm_Rz, norm_mpi(Rz) / (Pmax - Pmin) * lx * _sqrt_len_Rz_g)
            push!(norm_∇V, norm_mpi(∇V) / (Vmax - Vmin) * lx * _sqrt_len_∇V_g)
            err = maximum([norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont]])
           
            push!(
                err_evo1,
                maximum([norm_Rx[cont], norm_Ry[cont], norm_Rz[cont], norm_∇V[cont]]),
            )
            push!(err_evo2, iter)
            if igg.me == 0  && verbose && ((err < ϵ) || (iter == iterMax))
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

            if isnan(err)
                error("NaN")
            end
        end

        if igg.me == 0 && err ≤ ϵ
            println("Pseudo-transient iterations finished after $iter iterations")
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

end # END OF MODULE

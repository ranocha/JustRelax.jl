struct ThermalParameters{T}
    κ::T # thermal diffusivity

    function ThermalParameters(K::T, ρCp::T) where {T}
        κ = K ./ ρCp
        return new{T}(κ)
    end
end

@parallel function update_ΔT!(ΔT, T, Told)
    @all(ΔT) = @all(T) - @all(Told)
    return nothing
end

## GeoParams

function compute_diffusivity(rheology::MaterialParams, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * compute_density(rheology, args))
end
function compute_diffusivity(rheology::MaterialParams, ρ::Number, args)
    return compute_conductivity(rheology, args) *
           inv(compute_heatcapacity(rheology, args) * ρ)
end

# 1D THERMAL DIFFUSION MODULE

module ThermalDiffusion1D
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
using JustRelax
using CUDA
using GeoParams

import JustRelax: ThermalParameters, solve!, assign!, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs, solve!, compute_diffusivity, update_ΔT!

export solve!

## KERNELS

@parallel function compute_flux!(qTx, T, κ, _dx)
    @all(qTx) = -@av(κ) * @d(T) * _dx
    return nothing
end

@parallel_indices (i) function advect_T!(dT_dt, qTx, T, Vx, _dx)
    if i ≤ size(dT_dt, 1)
        dT_dt[i, j] =
            -(qTx[i + 1] - qTx[i]) * _dx -
            (Vx[i + 1] > 0) * Vx[i + 1] * (T[i + 1] - T[i]) * _dx -
            (Vx[i + 2] < 0) * Vx[i + 2] * (T[i + 2] - T[i + 1]) * _dx
    end
    return nothing
end

@parallel function advect_T!(dT_dt, qTx, _dx)
    @all(dT_dt) = -@d(qTx) * _dx
    return nothing
end

@parallel function update_T!(T, dT_dt, dt)
    @inn(T) = @inn(T) + @all(dT_dt) * dt
    return nothing
end

## SOLVER

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_parameters::ThermalParameters{M},
    thermal_bc::NamedTuple,
    di::NTuple{1,_T},
    dt,
) where {_T,M<:AbstractVector}

    # Compute some constant stuff
    _dx, = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(thermal.qTx, thermal.T, thermal_parameters.κ, _dx)
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, _dx)
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    return nothing
end

# upwind advection
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,1}},
    stokes,
    thermal_bc::NamedTuple,
    di::NTuple{1,_T},
    dt,
) where {_T,M<:AbstractArray{<:Any,1}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(thermal.qTx, thermal.T, thermal_parameters.κ, _dx)
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.T, stokes.V.Vx, _dx)
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    return nothing
end

end ## END OF 1D MODULE

# 2D THERMAL DIFFUSION MODULE

module ThermalDiffusion2D

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using JustRelax
using CUDA
using GeoParams

import JustRelax: ThermalParameters, solve!, assign!, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs, solve!, compute_diffusivity, update_ΔT!

export solve!

## KERNELS

@parallel function compute_flux!(qTx, qTy, T, κ, _dx, _dy)
    @all(qTx) = -@av_xi(κ) * @d_xi(T) * _dx
    @all(qTy) = -@av_yi(κ) * @d_yi(T) * _dy
    return nothing
end

@parallel_indices (i, j) function compute_flux!(
    qTx, qTy, T, rheology::MaterialParams, args, _dx, _dy
)
    @inline dTdxi(i, j) = (T[i + 1, j + 1] - T[i, j + 1]) * _dx
    @inline dTdyi(i, j) = (T[i + 1, j + 1] - T[i + 1, j]) * _dy

    if i ≤ size(qTx, 1) && j ≤ size(qTx, 2)
        qTx[i, j] = -compute_diffusivity(rheology, args) * dTdxi(i, j)
    end

    if i ≤ size(qTy, 1) && j ≤ size(qTy, 2)
        qTy[i, j] = -compute_diffusivity(rheology, args) * dTdyi(i, j)
    end

    return nothing
end

@parallel_indices (i, j) function advect_T!(dT_dt, qTx, qTy, T, Vx, Vy, _dx, _dy)
    if (i ≤ size(dT_dt, 1) && j ≤ size(dT_dt, 2))
        dT_dt[i, j] =
            -((qTx[i + 1, j] - qTx[i, j]) * _dx + (qTy[i, j + 1] - qTy[i, j]) * _dy) -
            (Vx[i + 1, j + 1] > 0) *
            Vx[i + 1, j + 1] *
            (T[i + 1, j + 1] - T[i, j + 1]) *
            _dx -
            (Vx[i + 2, j + 1] < 0) *
            Vx[i + 2, j + 1] *
            (T[i + 2, j + 1] - T[i + 1, j + 1]) *
            _dx -
            (Vy[i + 1, j + 1] > 0) *
            Vy[i + 1, j + 1] *
            (T[i + 1, j + 1] - T[i + 1, j]) *
            _dy -
            (Vy[i + 1, j + 2] < 0) *
            Vy[i + 1, j + 2] *
            (T[i + 1, j + 2] - T[i + 1, j + 1]) *
            _dy
    end
    return nothing
end

@parallel function advect_T!(dT_dt, qTx, qTy, _dx, _dy)
    @all(dT_dt) = -(@d_xa(qTx) * _dx + @d_ya(qTy) * _dy)
    return nothing
end

@parallel function update_T!(T, dT_dt, dt)
    @inn(T) = @inn(T) + @all(dT_dt) * dt
    return nothing
end

## SOLVER

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,2}},
    thermal_bc::NamedTuple,
    di::NTuple{2,_T},
    dt,
) where {_T,M<:AbstractArray{<:Any,2}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(
        thermal.qTx, thermal.qTy, thermal.T, thermal_parameters.κ, _dx, _dy
    )
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.qTy, _dx, _dy)
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    return nothing
end

# upwind advection
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,2}},
    stokes,
    thermal_bc::NamedTuple,
    di::NTuple{2,_T},
    dt,
) where {_T,M<:AbstractArray{<:Any,2}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(
        thermal.qTx, thermal.qTy, thermal.T, thermal_parameters.κ, _dx, _dy
    )
    @parallel advect_T!(
        thermal.dT_dt,
        thermal.qTx,
        thermal.qTy,
        thermal.T,
        stokes.V.Vx,
        stokes.V.Vy,
        _dx,
        _dy,
    )
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    return nothing
end

# GEOPARAMS VERSION

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::NamedTuple,
    rheology::MaterialParams,
    args::NamedTuple,
    di::NTuple{2,_T},
    dt;
    advection=true,
) where {_T,M<:AbstractArray{<:Any,2}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)
    nx, ny = size(thermal.T)

    # solve heat diffusion
    @parallel assign!(thermal.Told, thermal.T)
    @parallel (1:(nx - 1), 1:(ny - 1)) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.T, rheology, args, _dx, _dy
    )
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.qTy, _dx, _dy)
    @show extrema(thermal.dT_dt)
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    return nothing
end

# Upwind advection
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::NamedTuple,
    stokes,
    rheology::MaterialParams,
    args::NamedTuple,
    di::NTuple{2,_T},
    dt;
    advection=true,
) where {_T,M<:AbstractArray{<:Any,2}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)
    nx, ny = size(thermal.T)

    # solve heat diffusion
    @parallel assign!(thermal.Told, thermal.T)
    @parallel (1:(nx - 1), 1:(ny - 1)) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.T, rheology, args, _dx, _dy
    )
    @parallel advect_T!(
        thermal.dT_dt,
        thermal.qTx,
        thermal.qTy,
        thermal.T,
        stokes.V.Vx,
        stokes.V.Vy,
        _dx,
        _dy,
    )
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    return nothing
end

end

# 3D THERMAL DIFFUSION MODULE

module ThermalDiffusion3D

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using JustRelax
using MPI
using Printf
using CUDA
using GeoParams

import JustRelax:
    IGG, ThermalParameters, solve!, assign!, norm_mpi, thermal_boundary_conditions!
import JustRelax: ThermalArrays, PTThermalCoeffs, solve!, compute_diffusivity, update_ΔT!

export solve!

## KERNELS

@parallel function compute_flux!(qTx, qTy, qTz, T, κ, _dx, _dy, _dz)
    @all(qTx) = -@av_xi(κ) * @d_xi(T) * _dx
    @all(qTy) = -@av_yi(κ) * @d_yi(T) * _dy
    @all(qTz) = -@av_yi(κ) * @d_zi(T) * _dz
    return nothing
end

@parallel_indices (i, j, k) function compute_flux!(
    qTx, qTy, qTz, T, rheology::MaterialParams, args, _dx, _dy, _dz
)
    @inline dTdxi(i, j, k) = (T[i + 1, j + 1, k + 1] - T[i, j + 1, k + 1]) * _dx
    @inline dTdyi(i, j, k) = (T[i + 1, j + 1, k + 1] - T[i + 1, j, k + 1]) * _dy
    @inline dTdzi(i, j, k) = (T[i + 1, j + 1, k + 1] - T[i + 1, j + 1, k]) * _dz

    if i ≤ size(qTx, 1) && j ≤ size(qTx, 2) && k ≤ size(qTx, 3)
        qTx[i, j, k] = -compute_diffusivity(rheology, args) * dTdxi(i, j, k)
    end

    if i ≤ size(qTy, 1) && j ≤ size(qTy, 2) && k ≤ size(qTy, 3)
        qTy[i, j, k] = -compute_diffusivity(rheology, args) * dTdyi(i, j, k)
    end

    if i ≤ size(qTz, 1) && j ≤ size(qTz, 2) && k ≤ size(qTz, 3)
        qTz[i, j, k] = -compute_diffusivity(rheology, args) * dTdzi(i, j, k)
    end

    return nothing
end

@parallel_indices (i, j, k) function advect_T!(
    dT_dt, qTx, qTy, qTz, T, Vx, Vy, Vz, _dx, _dy, _dz
)
    if i ≤ size(dT_dt, 1) && j ≤ size(dT_dt, 2) && k ≤ size(dT_dt, 3)
        dT_dt[i, j, k] =
            -(
                (qTx[i + 1, j, k] - qTx[i, j, k]) * _dx +
                (qTy[i, j + 1, k] - qTy[i, j, k]) * _dy +
                (qTz[i, j, k + 1] - qTz[i, j, k]) * _dz
            ) -
            (Vx[i + 1, j + 1, k + 1] > 0) *
            Vx[i + 1, j + 1, k + 1] *
            (T[i + 1, j + 1, k + 1] - T[i, j + 1, k + 1]) *
            _dx -
            (Vx[i + 2, j + 1, k + 1] < 0) *
            Vx[i + 2, j + 1, k + 1] *
            (T[i + 2, j + 1, k + 1] - T[i + 1, j + 1, k + 1]) *
            _dx -
            (Vy[i + 1, j + 1, k + 1] > 0) *
            Vy[i + 1, j + 1, k + 1] *
            (T[i + 1, j + 1, k + 1] - T[i + 1, j, k + 1]) *
            _dy -
            (Vy[i + 1, j + 2, k + 1] < 0) *
            Vy[i + 1, j + 2, k + 1] *
            (T[i + 1, j + 2, k + 1] - T[i + 1, j + 1, k + 1]) *
            _dy -
            (Vz[i + 1, j + 1, k + 1] > 0) *
            Vz[i + 1, j + 1, k + 1] *
            (T[i + 1, j + 1, k + 1] - T[i + 1, j + 1, k]) *
            _dz -
            (Vz[i + 1, j + 1, k + 2] < 0) *
            Vz[i + 1, j + 1, k + 2] *
            (T[i + 1, j + 1, k + 2] - T[i + 1, j + 1, k + 1]) *
            _dz
    end
    return nothing
end

@parallel function advect_T!(dT_dt, qTx, qTy, qTz, _dx, _dy, _dz)
    @all(dT_dt) = -(@d_xa(qTx) * _dx + @d_ya(qTy) * _dy + @d_za(qTz) * _dz)
    return nothing
end

@parallel function update_T!(T, dT_dt, dt)
    @inn(T) = @inn(T) + @all(dT_dt) * dt
    return nothing
end

## SOLVER

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,3}},
    thermal_bc::NamedTuple,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(
        thermal.qTx,
        thermal.qTy,
        thermal.qTz,
        thermal.T,
        thermal_parameters.κ,
        _dx,
        _dy,
        _dz,
    )
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.qTy, thermal.qTz, _dx, _dy, _dz)
    @hide_communication b_width begin # communication/computation overlap
        @parallel update_T!(thermal.T, thermal.dT_dt, dt)
        update_halo!(thermal.T)
    end
    thermal_boundary_conditions!(thermal_bc, thermal.T)
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    return nothing
end

# upwind advection
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_parameters::ThermalParameters{<:AbstractArray{_T,3}},
    thermal_bc::NamedTuple,
    stokes,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _dx, _dy = inv.(di)

    @parallel assign!(thermal.Told, thermal.T)
    @parallel compute_flux!(
        thermal.qTx,
        thermal.qTy,
        thermal.qTz,
        thermal.T,
        thermal_parameters.κ,
        _dx,
        _dy,
        _dz,
    )
    @hide_communication b_width begin # communication/computation overlap
        @parallel advect_T!(
            thermal.dT_dt,
            thermal.qTx,
            thermal.qTy,
            thermal.qTz,
            thermal.T,
            stokes.V.Vx,
            stokes.V.Vy,
            stokes.V.Vz,
            _dx,
            _dy,
            _dz,
        )
        update_halo!(thermal.T)
    end
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    return nothing
end

# GEOPARAMS VERSION

function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::NamedTuple,
    rheology::MaterialParams,
    args::NamedTuple,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _dx, _dy, _dz = inv.(di)
    nx, ny, nz = size(thermal.T)

    # solve heat diffusion
    @parallel assign!(thermal.Told, thermal.T)
    @parallel (1:(nx - 1), 1:(ny - 1), 1:(nz - 1)) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.qTz, thermal.T, rheology, args, _dx, _dy, _dz
    )
    @parallel advect_T!(thermal.dT_dt, thermal.qTx, thermal.qTy, thermal.qTz, _dx, _dy, _dz)
    @hide_communication b_width begin # communication/computation overlap
        @parallel update_T!(thermal.T, thermal.dT_dt, dt)
        update_halo!(thermal.T)
    end
    thermal_boundary_conditions!(thermal_bc, thermal.T)
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    return nothing
end

# upwind advection 
function JustRelax.solve!(
    thermal::ThermalArrays{M},
    thermal_bc::NamedTuple,
    stokes,
    rheology::MaterialParams,
    args::NamedTuple,
    di::NTuple{3,_T},
    dt;
    b_width=(4, 4, 4),
) where {_T,M<:AbstractArray{<:Any,3}}

    # Compute some constant stuff
    _dx, _dy, _dz = inv.(di)
    nx, ny, nz = size(thermal.T)

    # solve heat diffusion
    @parallel assign!(thermal.Told, thermal.T)
    @parallel (1:(nx - 1), 1:(ny - 1), 1:(nz - 1)) compute_flux!(
        thermal.qTx, thermal.qTy, thermal.qTz, thermal.T, rheology, args, _dx, _dy, _dz
    )
    @hide_communication b_width begin # communication/computation overlap
        @parallel advect_T!(
            thermal.dT_dt,
            thermal.qTx,
            thermal.qTy,
            thermal.qTz,
            thermal.T,
            stokes.V.Vx,
            stokes.V.Vy,
            stokes.V.Vz,
            _dx,
            _dy,
            _dz,
        )
        update_halo!(thermal.T)
    end
    @parallel update_T!(thermal.T, thermal.dT_dt, dt)
    thermal_boundary_conditions!(thermal_bc, thermal.T)
    @parallel update_ΔT!(thermal.ΔT, thermal.T, thermal.Told)

    return nothing
end

end

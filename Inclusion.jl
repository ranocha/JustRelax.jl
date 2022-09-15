# include benchmark related functions
using JustRelax
using Printf, LinearAlgebra #, CairoMakie
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
    η[Rad2 .< rc] .= ηi
    phase[Rad2 .< rc] .= 2
    return η, phase
end

function fill_borders!(A)
    A[:, 1] .= A[:, 2]
    A[:, end] .= A[:, end - 1]
    A[1, :] .= A[2, :]
    return A[end, :] .= A[end - 1, :]
end

function viscoelastic_viscosity(MatParam, G, phase, dt, I::Vararg{Int64,N}) where {N}
    return 1.0 / (
        1.0 / G[I...] * dt +
        1.0 / computeViscosity_τII(MatParam, 0.0, phase[I...], (;); cutoff=(-Inf, Inf))
    )
end

@parallel_indices (i, j) function viscoelastic_viscosity!(ηve, MatParam, G, phase, dt)
    ηve[i, j] = inv(
        inv(G[i, j] * dt) +
        inv(computeViscosity_τII(MatParam, 0.0, phase[i, j], (;); cutoff=(-Inf, Inf))),
    )
    return nothing
end

@parallel_indices (i, j) function pt_viscosity!(ηpt, MatParam, G, Gdτ, phase, dt)
    ηpt[i, j] = inv(
        inv(G[i, j] * dt) +
        inv(computeViscosity_τII(MatParam, 0.0, phase[i, j], (;); cutoff=(-Inf, Inf))) + 
        inv(Gdτ[i,j])
    )
    return nothing
end


@parallel_indices (i, j) function update_plastic_strain!(λ, P, ηve, τII, phase, MatParam)
    function _update_Ftrial!(λ, i, j)
        @inbounds Ftrial = compute_yieldfunction(MatParam, phase[i, j], (P=P[i, j], τII=τII[i, j]))

        @inbounds if Ftrial > 0.0
            λ[i, j] = Ftrial * inv(ηve[i, j])
        else
            λ[i, j] = 0.0
        end
    end

    _update_Ftrial!(λ, i, j)

    return nothing
end


@parallel_indices (i, j) function yield_surface!(F, P, τII, phase, MatParam)
    F[i,j] = compute_yieldfunction(MatParam, phase[i, j], (P=P[i, j], τII=τII[i, j]))
    return nothing
end

# @parallel_indices (i, j) function update_Ftrial!(Ftrial, λ, P, ηvp, τII, phase, MatParam)
#     Ftrial[i, j] =
#         compute_yieldfunction(MatParam, phase[i, j], (P=P[i, j], τII=τII[i, j])) -
#         λ[i, j] * ηvp[i, j]

#     return nothing
# end

# @parallel_indices (i, j) function update_plastic_strain!(λ, Ftrial, ηve, ηvp)
#     if Ftrial[i, j] < 0.0
#         λ[i, j] = Ftrial[i, j] * 1.0 / (ηve[i, j] + ηvp[i, j])
#     end
#     return nothing
# end

@parallel_indices (i, j) function second_invariant!(CII, Cxx, Cyy, Cxy)
    normal_component(i, j) = @inbounds (Cxx[i, j], Cyy[i, j])
    function shear_component(i, j)
        @inbounds (Cxy[i - 1, j - 1], τxy[i, j], Cxy[i - 1, j], Cxy[i, j - 1])
    end

    CII[i, j] = second_invariant_staggered(normal_component(i, j), shear_component(i, j))

    return nothing
end

@parallel function second_invariant_vertex!(CII, Cxx, Cyy, Cxy)
    @all(CII) = second_invariant((@all(Cxx), @all(Cyy), @all(Cxy)))

    return nothing
end


@parallel (1:nx, 1:ny) second_invariant_vertex!(τII, τxx, τyy, τxyv)

# @parallel_indices (i, j) function stress_corrections!(
#     τxx, τyy, τxy, τII, εxx, εyy, εxy, λ, ηve
# )
#     av(i, j) = (τII[i - 1, j - 1] + τII[i, j] + τII[i - 1, j] + τII[i, j - 1]) * 0.25

#     τxx[i, j] += 2.0 * ηve[i, j] * (- λ[i, j] * (0.5 * τxx[i, j] / τII[i, j]))
#     τyy[i, j] += 2.0 * ηve[i, j] * (- λ[i, j] * (0.5 * τyy[i, j] / τII[i, j]))
#     τxy[i - 1, j - 1] +=
#         2.0 * ηve[i, j] * (- λ[i, j] * (0.5 * τxy[i, j] / av(i, j)))

#     # @all(τxx) =
#     #     (@all(τxx) + @all(τxx_o) * @Gr2() + T(2) * @all(Gdτ) * @all(εxx)) /
#     #     (one(T) + @all(Gdτ) / @all(η) + @Gr2())
#     # @all(τyy) =
#     #     (@all(τyy) + @all(τyy_o) * @Gr2() + T(2) * @all(Gdτ) * @all(εyy)) /
#     #     (one(T) + @all(Gdτ) / @all(η) + @Gr2())
#     # @all(τxy) =
#     #     (@all(τxy) + @all(τxy_o) * @harm_Gr2() + T(2) * @harm(Gdτ) * @all(εxy)) /
#     #     (one(T) + @harm(Gdτ) / @harm(η) + @harm_Gr2())

#     return nothing
# end

# @parallel_indices (i, j) function stress_corrections!(
#     τxx, τyy, τxy, τII, εxx, εyy, εxy, λ, ηve, Gdτ
# )
#     # av(A, i, j) = (A[i - 1, j - 1] + A[i, j] + A[i - 1, j] + A[i, j - 1]) * 0.25
#     av(A, i, j) = sum(@inbounds inv(A[ii, jj]) for ii in (i - 1):i, jj in (j - 1):j) * 0.25
#     function harm(A, i, j)
#         # 4.0 / (
#         #     1.0 / A[i - 1, j - 1] + 1.0 / A[i, j] + 1.0 / A[i - 1, j] + 1.0 / A[i, j - 1]
#         # )
#         4.0 / (sum(@inbounds inv(A[ii, jj]) for ii in (i - 1):i, jj in (j - 1):j))
#     end
#     visc_eff(i, j) = 2.0 / (1.0 / Gdτ[i, j] + 1.0 / ηve[i, j])
#     function update_normal_stress(τii, εii, i, j)
#         visc_eff(i, j) * (εii[i, j] - λ[i, j] * (0.5 * τii[i, j] / τII[i, j]))
#     end

#     τxx[i, j] = update_normal_stress(τxx, εxx, i, j)
#     τyy[i, j] = update_normal_stress(τyy, εyy, i, j)
#     if i > 1 && j > 1
#         τxy[i - 1, j - 1] +=
#             2.0 / (1.0 / harm(Gdτ, i, j) + 1.0 / harm(ηve, i, j)) *
#             (εxy[i, j] - harm(λ, i, j) * (0.5 * τxy[i, j] / harm(τII, i, j)))
#     end

#     return nothing
# end

@parallel function stress_corrections!(
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    τxx_trial::AbstractArray{T,2},
    τyy_trial::AbstractArray{T,2},
    τxy_trial::AbstractArray{T,2},
    τxx_o::AbstractArray{T,2},
    τyy_o::AbstractArray{T,2},
    τxy_o::AbstractArray{T,2},
    Gdτ::AbstractArray{T,2},
    εxx::AbstractArray{T,2},
    εyy::AbstractArray{T,2},
    εxy::AbstractArray{T,2},
    η::AbstractArray{T,2},
    λ::AbstractArray{T,2},
    G::AbstractArray{T,2},
    dt::T,
) where {T}
    @all(τxx) =
        (@all(τxx) + @all(τxx_o) * @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) *( @all(εxx) - @all(λ) * (0.5 * @all(τxx_trial) / @all(τII)))) /
        (one(T) + @all(Gdτ) / @all(η) +  @all(Gdτ)/(@all(G)*dt))
    @all(τyy) =
        (@all(τyy) + @all(τyy_o) *  @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) *( @all(εyy) - @all(λ) * (0.5 * @all(τyy_trial) / @all(τII)))) /
        (one(T) + @all(Gdτ) / @all(η) +  @all(Gdτ)/(@all(G)*dt))
    @all(τxy) =
        (@all(τxy) + @all(τxy_o) * @harm(Gdτ)/(@harm(G)*dt) + T(2) * @harm(Gdτ) * (@all(εxy) - @harm(λ) * (@all(τxy_trial) / @harm(τII)))) /
        (one(T) + @harm(Gdτ) / @harm(η) + @harm(Gdτ)/(@harm(G)*dt))
    return nothing
end

@parallel function stress_corrections2!(
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    τxx_trial::AbstractArray{T,2},
    τyy_trial::AbstractArray{T,2},
    τxy_trial::AbstractArray{T,2},
    τxx_o::AbstractArray{T,2},
    τyy_o::AbstractArray{T,2},
    τxy_o::AbstractArray{T,2},
    εxx::AbstractArray{T,2},
    εyy::AbstractArray{T,2},
    εxy::AbstractArray{T,2},
    ηve::AbstractArray{T,2},
    λ::AbstractArray{T,2},
    G::AbstractArray{T,2},
    dt::T,
) where {T}
    @all(τxx) = 2.0*@all(ηve) * (@all(εxx) - @all(λ) * (0.5 * @all(τxx_trial) / @all(τII))) +
        inv(@all(G)*dt) * @all(τxx_o)
    @all(τyy) = 2.0*@all(ηve) * (@all(εyy) - @all(λ) * (0.5 * @all(τyy_trial) / @all(τII))) +
        inv(@all(G)*dt) * @all(τyy_o)
    @all(τxy) = 2.0*@av(ηve) * (@all(εxy) - @all(λ) * (@all(τxy_trial) / @av(τII))) +
        inv(@av(G)*dt) * @all(τxy_o)
    return nothing
end

@parallel function stress_corrections_vertex!(
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    τxx_trial::AbstractArray{T,2},
    τyy_trial::AbstractArray{T,2},
    τxy_trial::AbstractArray{T,2},
    τxx_o::AbstractArray{T,2},
    τyy_o::AbstractArray{T,2},
    τxy_o::AbstractArray{T,2},
    εxx::AbstractArray{T,2},
    εyy::AbstractArray{T,2},
    εxy::AbstractArray{T,2},
    ηve::AbstractArray{T,2},
    λ::AbstractArray{T,2},
    G::AbstractArray{T,2},
    dt::T,
) where {T}
    @all(τxx) = 2.0*@all(ηve) * (@all(εxx) - @all(λ) * (0.5 * @all(τxx_trial) / @all(τII))) +
        inv(@all(G)*dt) * @all(τxx_o)
    @all(τyy) = 2.0*@all(ηve) * (@all(εyy) - @all(λ) * (0.5 * @all(τyy_trial) / @all(τII))) +
        inv(@all(G)*dt) * @all(τyy_o)
    @all(τxy) = 2.0*@all(ηve) * (@all(εxy) - 0.5 * @all(λ) * (@all(τxy_trial) / @all(τII))) +
        inv(@all(G)*dt) * @all(τxy_o)
    return nothing
end

@parallel function stress_corrections_vertex2!(
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    τxx_trial::AbstractArray{T,2},
    τyy_trial::AbstractArray{T,2},
    τxy_trial::AbstractArray{T,2},
    τxx_o::AbstractArray{T,2},
    τyy_o::AbstractArray{T,2},
    τxy_o::AbstractArray{T,2},
    Gdτ::AbstractArray{T,2},
    εxx::AbstractArray{T,2},
    εyy::AbstractArray{T,2},
    εxy::AbstractArray{T,2},
    η::AbstractArray{T,2},
    λ::AbstractArray{T,2},
    G::AbstractArray{T,2},
    dt::T,
) where {T}
    @all(τxx) =
        (@all(τxx) + @all(τxx_o) * @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) *( @all(εxx) - @all(λ) * (0.5 * @all(τxx_trial) / @all(τII)))) /
        (one(T) + @all(Gdτ) / @all(η) +  @all(Gdτ)/(@all(G)*dt))
    @all(τyy) =
        (@all(τyy) + @all(τyy_o) *  @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) *( @all(εyy) - @all(λ) * (0.5 * @all(τyy_trial) / @all(τII)))) /
        (one(T) + @all(Gdτ) / @all(η) +  @all(Gdτ)/(@all(G)*dt))
    @all(τxy) =
        (@all(τxy) + @all(τxy_o) * @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) * (@all(εxy) - 0.5 * @all(λ) * (@all(τxy_trial) / @all(τII)))) /
        (one(T) + @all(Gdτ) / @all(η) + @all(Gdτ)/(@all(G)*dt))
    return nothing
end

@parallel function trial_stress!(
    τxx_trial::AbstractArray{T,2},
    τyy_trial::AbstractArray{T,2},
    τxy_trial::AbstractArray{T,2},
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
    G::AbstractArray{T,2},
    dt::T,
) where {T}
    @all(τxx_trial) =
        (@all(τxx) + @all(τxx_o) * @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) * @all(εxx)) /
        (one(T) + @all(Gdτ) / @all(η) +  @all(Gdτ)/(@all(G)*dt))
    @all(τyy_trial) =
        (@all(τyy) + @all(τyy_o) *  @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) * @all(εyy)) /
        (one(T) + @all(Gdτ) / @all(η) +  @all(Gdτ)/(@all(G)*dt))
    @all(τxy_trial) =
        (@all(τxy) + @all(τxy_o) * @harm(Gdτ)/(@harm(G)*dt) + T(2) * @harm(Gdτ) * @all(εxy)) /
        (one(T) + @harm(Gdτ) / @harm(η) + @harm(Gdτ)/(@harm(G)*dt))
    return nothing
end

# @parallel function trial_stress_vertex!(
#     τxx_trial::AbstractArray{T,2},
#     τyy_trial::AbstractArray{T,2},
#     τxy_trial::AbstractArray{T,2},
#     τxx::AbstractArray{T,2},
#     τyy::AbstractArray{T,2},
#     τxy::AbstractArray{T,2},
#     τxx_o::AbstractArray{T,2},
#     τyy_o::AbstractArray{T,2},
#     τxy_o::AbstractArray{T,2},
#     Gdτ::AbstractArray{T,2},
#     εxx::AbstractArray{T,2},
#     εyy::AbstractArray{T,2},
#     εxy::AbstractArray{T,2},
#     η::AbstractArray{T,2},
#     G::AbstractArray{T,2},
#     dt::T,
# ) where {T}
#     @all(τxx_trial) =
#         (@all(τxx) + @all(τxx_o) * @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) * @all(εxx)) /
#         (one(T) + @all(Gdτ) / @all(η) +  @all(Gdτ)/(@all(G)*dt))
#     @all(τyy_trial) =
#         (@all(τyy) + @all(τyy_o) *  @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) * @all(εyy)) /
#         (one(T) + @all(Gdτ) / @all(η) +  @all(Gdτ)/(@all(G)*dt))
#     @all(τxy_trial) =
#         (@all(τxy) + @all(τxy_o) * @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) * @all(εxy)) /
#         (one(T) + @all(Gdτ) / @all(η) + @all(Gdτ)/(@all(G)*dt))
#     return nothing
# end
   

# @parallel function trial_stress_vertex!(
#     τxx_trial::AbstractArray{T,2},
#     τyy_trial::AbstractArray{T,2},
#     τxy_trial::AbstractArray{T,2},
#     τxx::AbstractArray{T,2},
#     τyy::AbstractArray{T,2},
#     τxy::AbstractArray{T,2},
#     τxx_o::AbstractArray{T,2},
#     τyy_o::AbstractArray{T,2},
#     τxy_o::AbstractArray{T,2},
#     Gdτ::AbstractArray{T,2},
#     εxx::AbstractArray{T,2},
#     εyy::AbstractArray{T,2},
#     εxy::AbstractArray{T,2},
#     η::AbstractArray{T,2},
#     G::AbstractArray{T,2},
#     dt::T,
# ) where {T}
#     @all(τxx_trial) =
#         (@all(τxx) + @all(τxx_o) * @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) * @all(εxx)) /
#         (one(T) + @all(Gdτ) / @all(η) +  @all(Gdτ)/(@all(G)*dt))
#     @all(τyy_trial) =
#         (@all(τyy) + @all(τyy_o) *  @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) * @all(εyy)) /
#         (one(T) + @all(Gdτ) / @all(η) +  @all(Gdτ)/(@all(G)*dt))
#     @all(τxy_trial) =
#         (@all(τxy) + @all(τxy_o) * @all(Gdτ)/(@all(G)*dt) + T(2) * @all(Gdτ) * @all(εxy)) /
#         (one(T) + @all(Gdτ) / @all(η) + @all(Gdτ)/(@all(G)*dt))
#     return nothing
# end

@parallel function stress_corrections_vertex3!(
    τxx::AbstractArray{T,2},
    τyy::AbstractArray{T,2},
    τxy::AbstractArray{T,2},
    τxx_trial::AbstractArray{T,2},
    τyy_trial::AbstractArray{T,2},
    τxy_trial::AbstractArray{T,2},
    εxx::AbstractArray{T,2},
    εyy::AbstractArray{T,2},
    εxy::AbstractArray{T,2},
    ηpt::AbstractArray{T,2},
    λ::AbstractArray{T,2},
) where {T}
    @all(τxx) = 2.0 * @all(ηpt) * (@all(εxx) - @all(λ) * (0.5 * @all(τxx_trial) / @all(τII))) 
    @all(τyy) = 2.0 * @all(ηpt) * (@all(εyy) - @all(λ) * (0.5 * @all(τyy_trial) / @all(τII)))
    @all(τxy) = 2.0 * @all(ηpt) * (@all(εxy) - 0.5 * @all(λ) * (@all(τxy_trial) / @all(τII))) 
    return nothing
end

@parallel function trial_stress_vertex2!(
    τxx_trial::AbstractArray{T,2},
    τyy_trial::AbstractArray{T,2},
    τxy_trial::AbstractArray{T,2},
    εxx::AbstractArray{T,2},
    εyy::AbstractArray{T,2},
    εxy::AbstractArray{T,2},
    ηpt::AbstractArray{T,2},
) where {T}
    @all(τxx_trial) = 2.0 * @all(ηpt) * @all(εxx) 
    @all(τyy_trial) = 2.0 * @all(ηpt) * @all(εyy)
    @all(τxy_trial) = 2.0 * @all(ηpt) * @all(εxy) 
    return nothing
end

@parallel function center2vertex_inn!(Av, Ac)
    @inn(Av) = @av(Ac)
    return nothing 
end

function center2vertex!(Av, Ac)
    freeslip = (freeslip_x=true, freeslip_y=true)
    @parallel center2vertex_inn!(Av, Ac)
    apply_free_slip!(freeslip, Av, Av)
end

@parallel function _vertex2center!(Ac, Av)
    @all(Ac) = @av(Av)
    return nothing 
end

@inline vertex2center!(Ac, Av) = @parallel _vertex2center!(Ac, Av)

# @parallel_indices (i, j) function stress_corrections!(
#     τxx, τyy, τxy, τII, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, λ, ηve, G, Gdτ, _dt
# )
#     av(A, i, j) = (A[i - 1, j - 1] + A[i, j] + A[i - 1, j] + A[i, j - 1]) * 0.25
#     harm(A, i, j) = 4.0 / (1.0/A[i - 1, j - 1] + 1.0/A[i, j] + 1.0/A[i - 1, j] + 1.0/A[i, j - 1]) 
#     visc_eff(i, j) = 2.0 / (1.0 / Gdτ[i,j] + 1.0 / ηve[i, j])
#     update_normal_stress(τii, τii_o, εii, i,j) = visc_eff(i, j) * ((εii[i,j] - λ[i, j] * (0.5 * τii[i, j] / τII[i, j])) + _dt * 0.5 / G[i,j] * τii_o[i, j] + 0.5 * τii[i, j] / Gdτ[i,j])

#     τxx[i, j] = update_normal_stress(τxx, τxx_o, εxx, i, j)
#     τyy[i, j] = update_normal_stress(τyy, τyy_o, εyy, i, j)
#     τxy[i - 1, j - 1] +=
#         2.0 / (1.0 / harm(Gdτ, i, j) + 1.0 / harm(ηve, i,  j)) * (εxy[i,j] - λ[i, j] * (0.5 * τxy[i, j] / av(τII, i, j)) + _dt * 0.5 / harm(G,i,j) * τxy_o[i, j] + 0.5 * τxy[i, j] / harm(Gdτ,i,j))

#     # τxx[i, j] = visc_eff(i, j) * ((εxx[i,j] - λ[i, j] * (0.5 * τxx[i, j] / τII[i, j])) + _dt * 0.5 / G[i,j] * τxx_o[i, j] + 0.5 * τxx[i, j] / Gdτ[i,j])
#     # τyy[i, j] = visc_eff(i, j) * ((εyy[i,j] - λ[i, j] * (0.5 * τyy[i, j] / τII[i, j])) + _dt * 0.5 / G[i,j] * τyy_o[i, j] + 0.5 * τyy[i, j] / Gdτ[i,j])

#     return nothing
# end

# @parallel function viscoplastic_viscosity!(ηvp, τII, εII)
#     @all(ηvp) = 0.5 * @all(τII) / @all(εII)
#     return nothing
# end

@parallel function effective_strain_rate!(
    εxx_eff,
    εyy_eff,
    εxy_eff,
    τxx,
    τyy,
    τxy,
    τxx_o,
    τyy_o,
    τxy_o,
    εxx,
    εyy,
    εxy,
    G,
    Gdτ,
    _dt,
)
    # @all(εxx_eff) = @all(εxx) + 0.5 * (_dt * @all(τxx_o) / @all(G) + @all(τxx) / @all(Gdτ))
    # @all(εyy_eff) = @all(εyy) + 0.5 * (_dt * @all(τyy_o) / @all(G) + @all(τyy) / @all(Gdτ))
    # @all(εxy_eff) =
    #     @all(εxy) + 0.5 * (_dt * @all(τxy_o) / @harm(G) + @all(τxy) / @harm(Gdτ))
    @all(εxx_eff) = @all(εxx) + 0.5 * (_dt * @all(τxx_o) / @all(G))
    @all(εyy_eff) = @all(εyy) + 0.5 * (_dt * @all(τyy_o) / @all(G))
    @all(εxy_eff) = @all(εxy) + 0.5 * (_dt * @all(τxy_o) / @harm(G))
    return nothing
end

@parallel function effective_strain_rate_vertex!(
    εxx_eff,
    εyy_eff,
    εxy_eff,
    τxx_o,
    τyy_o,
    τxy_o,
    τxx,
    τyy,
    τxy,
    εxx,
    εyy,
    εxy,
    G,
    Gdτ,
    _dt,
)
    @all(εxx_eff) = @all(εxx) + 0.5 * (_dt * @all(τxx_o) / @all(G)) + 0.5 * (_dt * @all(τxx) / @all(Gdτ))
    @all(εyy_eff) = @all(εyy) + 0.5 * (_dt * @all(τyy_o) / @all(G)) + 0.5 * (_dt * @all(τyy) / @all(Gdτ))
    @all(εxy_eff) = @all(εxy) + 0.5 * (_dt * @all(τxy_o) / @all(G)) + 0.5 * (_dt * @all(τxy) / @all(Gdτ))
    return nothing
end

@parallel_indices (i,j) function viscoplastic_viscosity!(ηvp, η, λ, τII, εII)
    if λ[i,j] == 0.0
        ηvp[i,j] = η[i,j]
    else
        ηvp[i,j] = 0.5 * τII[i,j] / εII[i,j]
    end
    return nothing
end

Δη = 1e-3
nx = 31
ny = 31
lx = 1e0
ly = 1e0
rc = 1e-1
εbg = 1e0

function solVi(; Δη=1e-3, nx=256 - 1, ny=256 - 1, lx=1e1, ly=1e1, rc=1e0, εbg=1e0)
    CharDim = GEO_units(; length=10km, viscosity=1e20Pa * s)        # Characteristic dimensions
    MatParam = (
        SetMaterialParams(;
            Name="Matrix",
            Phase=1,
            Density=PT_Density(; ρ0=3000kg / m^3, β=0.0 / Pa),
            CreepLaws=LinearViscous(; η=1e20Pa * s),
            Plasticity=DruckerPrager(; C=1.6NoUnits, ϕ = 0NoUnits),
            # Plasticity=DruckerPrager(; C=(1.6/cosd(30))NoUnits),
            CharDim=CharDim,
        ),
        SetMaterialParams(;
            Name="Inclusion",
            Phase=2,
            Density=PT_Density(; ρ0=3000kg / m^3, β=0.0 / Pa),
            CreepLaws=LinearViscous(; η=1e20Pa * s),
            Plasticity=DruckerPrager(; C=1.6NoUnits, ϕ = 0NoUnits),
            # Plasticity=DruckerPrager(; C=(1.6/cosd(30))NoUnits),
            CharDim=CharDim,
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
    xci = Tuple([(di[i] / 2):di[i]:(li[i] - di[i] / 2) for i in 1:nDim]) # nodes at the center of the cells
    xvi = Tuple([0:di[i]:li[i] for i in 1:nDim]) # nodes at the vertices of the cells

    ## (Physical) Time domain and discretization
    ttot = 1 # total simulation time
    Δt = 1   # physical time step

    stokes = StokesArrays(ni, ViscoElastic)
    # general numerical coeffs for PT stokes
    pt_stokes = PTStokesCoeffs(ni, di)

    ## Setup-specific parameters and fields
    ξ = 4.0         # Maxwell relaxation time
    G = @ones(ni...)         # elastic shear modulus

    η0 = 1e0  # matrix viscosity
    ηi = 1e0 # inclusion viscosity
    η, phase = solvi_viscosity(ni, di, li, rc, η0, ηi) # viscosity field
    ηpt = @zeros(ni...)
    G[phase .== 2] .*= 0.5
    G[phase .== 2] .*= 0.125
    dt = η0 / (maximum(G) * ξ)

    ηvp = @fill(0.0, ni...) # viscoplastic viscosity
    ηve = @zeros(ni...) # viscoelastic viscosity
    @parallel (1:ni[1], 1:ni[2]) viscoelastic_viscosity!(ηve, MatParam, G, phase, dt)

    ## Plasticity 
    Ftrial = @zeros(ni...)
    λ = @zeros(ni...)
    τII = @ones(ni...)
    εII = @ones(ni...)

    ## Boundary conditions
    pureshear_bc!(stokes, di, li, εbg)
    stokes.V.Vx .= [εbg * x for x in xvi[1], y in xci[2]]
    stokes.V.Vy .= [-εbg * y for x in xci[1], y in xvi[2]]
    freeslip = (freeslip_x=true, freeslip_y=true)

    ###
    # unpack
    _dt = inv(dt)
    dx, dy = di
    _dx, _dy = inv.(di)
    lx, ly = li
    Vx, Vy = stokes.V.Vx, stokes.V.Vy
    dVx, dVy = stokes.dV.Vx, stokes.dV.Vy
    εxx, εyy, εxy = JustRelax.strain(stokes)
    εxx_eff, εyy_eff, εxy_eff = deepcopy(JustRelax.strain(stokes))
    τ, τ_o = JustRelax.stress(stokes)
    τxx, τyy, τxy = τ
    τxx_trial, τyy_trial, τxy_trial = deepcopy(τ)
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
    F = @zeros(ni...)
    local iters

    iterMax = 100e3
    ϵ = 1e-6
    nout = 100
    # for t_it in 1:10
    # iters = solve!(stokes, pt_stokes, di, li, max_li, freeslip, ρ, η; iterMax=10e3)

    # f = Figure(resolution=(1200,1200))
    # ax1 = Axis(f[1,1])
    # ax2 = Axis(f[2,1])


    evo_Txx = []
    for _ in 1:10
        # ~preconditioner
        ητ = deepcopy(ηve)
        @parallel JustRelax.compute_maxloc!(ητ, ηve)
        # PT numerical coefficients
        @parallel JustRelax.Elasticity2D.elastic_iter_params!(
            dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li
        )

        # errors
        err = 2 * ϵ
        iter = 0
        cont = 0
        idx = (2:(ni[1] - 1), 2:(ni[2] - 1))

        # solver loop
        iter = 1
        while err > ϵ && iter ≤ iterMax
            @parallel (1:nx, 1:ny) pt_viscosity!(ηpt, MatParam, G, Gdτ, phase, dt)

            @parallel JustRelax.Stokes2D.compute_strain_rate!(εxx, εyy, εxy, Vx, Vy, _dx, _dy)
            center2vertex!(εxyv, εxy)
            center2vertex!(τxy_ov, τxy_o)
            @parallel JustRelax.Stokes2D.compute_P!(∇V, P, εxx, εyy, Gdτ, r)
            # @show extrema(P), extrema(∇V)

            # @parallel JustRelax.Elasticity2D.compute_τ!(
            #     τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Gdτ, εxx, εyy, εxy, η, G, dt
            # )
            # @parallel trial_stress!(
            #     τxx_trial, τyy_trial, τxy_trial, τxx, τyy, τxy, τxx_o, τyy_o, τxy_o, Gdτ, εxx, εyy, εxy, η, G, dt,
            # )
            @parallel effective_strain_rate_vertex!(
                εxx_eff,
                εyy_eff,
                εxy_effv,
                τxx_o,
                τyy_o,
                τxy_ov,
                τxx,
                τyy,
                τxyv,
                εxx,
                εyy,
                εxyv,
                G,
                Gdτ,
                _dt,
            )
            # @parallel trial_stress_vertex!(
            #     τxx_trial, τyy_trial, τxy_trialv, τxx, τyy, τxyv, τxx_o, τyy_o, τxy_ov, Gdτ, εxx_eff, εyy_eff, εxy_effv, η, G, dt,
            # )
            @parallel trial_stress_vertex2!(
                τxx_trial, τyy_trial, τxy_trialv, εxx_eff, εyy_eff, εxy_effv, ηpt
            )

            # # Plasticity kernels
            # # if iter > 1
            # @parallel idx second_invariant!(τII, τxx_trial, τyy_trial, τxy_trial)
            @parallel second_invariant_vertex!(τII, τxx_trial, τyy_trial, τxy_trialv)

            # @parallel idx second_invariant!(εII, εxx, εyy, εxy)
            # @parallel effective_strain_rate!(
            #     εxx_eff,
            #     εyy_eff,
            #     εxy_eff,
            #     τxx,
            #     τyy,
            #     τxy,
            #     τxx_o,
            #     τyy_o,
            #     τxy_o,
            #     εxx,
            #     εyy,
            #     εxy,
            #     G,
            #     Gdτ,
            #     _dt,
            # )
            # # # for f in (τxx, τyy, τxy)
            # # #     apply_free_slip!(freeslip, f, f)
            # # # end
            # @parallel idx second_invariant!(εII, εxx_eff, εyy_eff, εxy_eff)
            # for C in (τII, εII)
            #     apply_free_slip!(freeslip, C, C)
            # end

            @parallel (1:nx, 1:ny) update_plastic_strain!(λ, P, ηve, τII, phase, MatParam)
            # @parallel (1:nx, 1:ny) yield_surface!(F, P, τII, phase, MatParam)

            # @parallel idx stress_corrections!(
            #     τxx, τyy, τxy, τII, εxx_eff, εyy_eff, εxy_eff, λ, ηve, Gdτ
            # )
            # @parallel stress_corrections!(
            #     τxx, τyy, τxy, τxx_trial, τyy_trial, τxy_trial, τxx_o, τyy_o, τxy_o, Gdτ, εxx, εyy, εxy, η, λ, G, dt,
            # )
            # @parallel stress_corrections2!(
            #     τxx, τyy, τxy, τxx_trial, τyy_trial, τxy_trial, τxx_o, τyy_o, τxy_o, εxx, εyy, εxy, ηve, λ, G, dt,
            # )
            # @parallel stress_corrections_vertex!(
            #     τxx, τyy, τxyv, τxx_trial, τyy_trial, τxy_trialv, τxx_o, τyy_o, τxy_ov, εxx_eff, εyy_eff, εxy_effv, ηve, λ, G, dt,
            # )
            # @parallel stress_corrections_vertex2!(
            #     τxx, τyy, τxyv, τxx_trial, τyy_trial, τxy_trialv, τxx_o, τyy_o, τxy_ov, Gdτ, εxx_eff, εyy_eff, εxy_effv, η, λ, G, dt,
            # )
            @parallel stress_corrections_vertex3!(
                τxx, τyy, τxyv, τxx_trial, τyy_trial, τxy_trialv, εxx_eff, εyy_eff, εxy_effv, ηpt, λ,
            )
            # @parallel effective_strain_rate!(
            #     εxx_eff,
            #     εyy_eff,
            #     εxy_eff,
            #     τxx,
            #     τyy,
            #     τxy,
            #     τxx_o,
            #     τyy_o,
            #     τxy_o,
            #     εxx,
            #     εyy,
            #     εxy,
            #     G,
            #     Gdτ,
            #     _dt,
            # )
            # @parallel idx second_invariant!(εII, εxx_eff, εyy_eff, εxy_eff)
            # @parallel idx second_invariant!(τII, τxx, τyy, τxy)
            @parallel second_invariant_vertex!(τII, τxx, τyy, τxyv)
            # for C in (τII, εII)
            #     apply_free_slip!(freeslip, C, C)
            # end

            @parallel (1:nx, 1:ny) viscoplastic_viscosity!(ηvp, ηve, λ, τII, εII)
            vertex2center!(τxy, τxyv)

            @parallel JustRelax.Elasticity2D.compute_dV_elastic!(
                dVx, dVy, P, Rx, Ry, τxx, τyy, τxy, dτ_Rho, ρ, _dx, _dy
            )
            @parallel JustRelax.Stokes2D.compute_V!(Vx, Vy, dVx, dVy)

            # @parallel JustRelax.compute_maxloc!(ητ, ηvp)
            # # PT numerical coefficients
            # @parallel JustRelax.Elasticity2D.elastic_iter_params!(
            #     dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li
            # )
            # free slip boundary conditions
            apply_free_slip!(freeslip, Vx, Vy)

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
            @parallel JustRelax.compute_maxloc!(ητ, ηve)
            apply_free_slip!(freeslip, ητ, ητ)
            # PT numerical coefficients
            @parallel JustRelax.Elasticity2D.elastic_iter_params!(
                dτ_Rho, Gdτ, ητ, Vpdτ, G, dt, Re, r, max_li
            )
            iter += 1
            @show maximum(τII)

            # f, ax, h = heatmap(xvi[1], xvi[2], τxx)

            # JustRelax.Elasticity2D.update_τ_o!(stokes)
            # h=heatmap!(ax1, xvi[1],xvi[2], εII, colormap=:inferno)
            # scatter!(ax2, (t, maximum(τxx)), color=:black)
            # f
            # p3 = heatmap(xc, yc, τxx' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="τii")
        end
        JustRelax.Elasticity2D.update_τ_o!(stokes)
        push!(evo_Txx, maximum(τxx))
        @show maximum(τII)
        f,ax,h=heatmap(xvi[1],xvi[2], λ, colormap=:inferno)
        Colorbar(f[1,2], h)
        f
        # @show maximum(λ)
        # # f,ax,h=heatmap(xvi[1],xvi[2], P, colormap=:inferno)
        # # # f,ax,h=heatmap(xvi[1],xvi[2], λ, colormap=:inferno)
        # h=heatmap!(ax1, xvi[1],xvi[2], τII, colormap=:inferno)
        # # Colorbar(f[1,2], h)
        # scatter!(ax2, (t, maximum(τxx)), color=:black)
        # @show t += dt
        # f
    end
    # f
    
    # Lx,Ly=lx,ly
    # xv,yv =xvi
    # xc,yc =xci
    # p1 = heatmap(xv, yc, Vx' , aspect_ratio=1, xlims=(0, Lx), ylims=(dy/2, Ly-dy/2), c=:inferno, title="Vx")
    # # p2 = heatmap(xc, yv, Vy' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="Vy")
    # # p2 = heatmap(xc, yc, λ' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="η_vep")
    # # p2 = heatmap(xc, yc, Exx' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="η_vep")
    # # p2 = heatmap(xc, yc, η_vep' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="η_vep")
    # p2 = heatmap(xc, yc, P' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="τii")
    # p3 = heatmap(xc, yc, τII' , aspect_ratio=1, xlims=(dx/2, Lx-dx/2), ylims=(0, Ly), c=:inferno, title="τii")
    # p4 = plot(1:length(evo_Txx), evo_Txx , legend=false, xlabel="time", ylabel="max(τxx)", linewidth=0, markershape=:circle, framestyle=:box, markersize=3)
    # #     plot!(evo_t, 2.0.*εbg.*μ0.*(1.0.-exp.(.-evo_t.*G0./μ0)), linewidth=2.0) # analytical solution for VE loading
    # #     plot!(evo_t, 2.0.*εbg.*μ0.*ones(size(evo_t)), linewidth=2.0)  
    # display(plot(p1,p2,p3,p4))
    # end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), stokes, iters
end
# f,ax,h=heatmap(xvi[1],xvi[2], ηve)
# f,ax,h=heatmap(xvi[1],xvi[2], Vx)
# f,ax,h=heatmap(xvi[1],xvi[2], P, colormap=:inferno)
# f,ax,h=heatmap(xvi[1],xvi[2], λ, colormap=:inferno)
# f,ax,h=heatmap(xvi[1],xvi[2], εII)
# f,ax,h=heatmap(xvi[1],xvi[2], εxx)
# f,ax,h=heatmap(xvi[1],xvi[2], εyy)
# f,ax,h=heatmap(xvi[1],xvi[2], τII, colormap=:inferno)
# f,ax,h=heatmap(xvi[1],xvi[2], τxx_trial, colormap=:inferno)
# f,ax,h=heatmap(xvi[1],xvi[2], τxx.-τxx_trial, colormap=:inferno)
# # f, ax, h = heatmap(xvi[1], xvi[2], τxx_trial)
f, ax, h = heatmap(xvi[1], xvi[2], τxx)
# f, ax, h = heatmap(xvi[1], xvi[2], τyy)
# f, ax, h = heatmap(xvi[1], xvi[2], τxy)
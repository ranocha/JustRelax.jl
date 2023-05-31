# solver related
ϵ = pt_stokes.ϵ
# geometry
_di = @. 1 / di
ni = nx, ny, nz = size(stokes.P)

# ~preconditioner
ητ = deepcopy(η)
# @hide_communication b_width begin # communication/computation overlap
#     @parallel compute_maxloc!(ητ, η)
#     update_halo!(ητ)
# end
# @parallel (1:ny, 1:nz) free_slip_x!(ητ)
# @parallel (1:nx, 1:nz) free_slip_y!(ητ)
# @parallel (1:nx, 1:ny) free_slip_z!(ητ)

@copy stokes.P0 stokes.P

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
        # Update viscosity
        args_ηv = (; T = thermal.T, P = stokes.P, dt=Inf)
        ν = iter > 1 ? 0.5 : 1.0
        @parallel (@idx ni) JustRelax.Elasticity3D.compute_viscosity!(η, ν, phase_ratios, @strain(stokes)..., args_ηv, rheology)
        @hide_communication b_width begin # communication/computation overlap
            @parallel JustRelax.compute_maxloc!(ητ, η)
            update_halo!(ητ)
        end
        @parallel (1:ny, 1:nz) free_slip_x!(ητ)
        @parallel (1:nx, 1:nz) free_slip_y!(ητ)
        @parallel (1:nx, 1:ny) free_slip_z!(ητ)
        # Update buoyancy
        @parallel (@idx ni) JustRelax.Elasticity3D.compute_ρg!(ρg[3], phase_ratios, rheology, (T=thermal.T, P=stokes.P))
     
        @parallel (@idx ni) JustRelax.Elasticity3D.compute_∇V!(
            stokes.∇V, @velocity(stokes)..., _di...
        )
        @parallel (@idx ni) JustRelax.Elasticity3D.compute_P!(
            stokes.P,
            stokes.P0,
            stokes.R.RP,
            stokes.∇V,
            η,
            rheology,
            phase_ratios,
            dt,
            pt_stokes.r,
            pt_stokes.θ_dτ,
        )
        @parallel (@idx ni) JustRelax.Elasticity3D.compute_strain_rate!(
            stokes.∇V,
            @strain(stokes)...,
            @velocity(stokes)...,
            _di...,
        )
        @parallel (@idx ni) JustRelax.Elasticity3D.compute_τ_new!(
            @stress_center(stokes)...,
            stokes.τ.II,
            @tensor(stokes.τ_o)...,
            @strain(stokes)...,
            stokes.P,
            η,
            η_vep,
            phase_ratios,
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
            @parallel JustRelax.Elasticity3D.compute_V!(
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




    
τxx, τyy, τzz, τyz, τxz, τxy = @stress_center(stokes)
TII = stokes.τ.II
τxx_old, τyy_old, τzz_old, τyzv_old, τxzv_old, τxyv_old = @tensor(stokes.τ_o)
εxx, εyy, εzz, εyzv, εxzv, εxyv = @strain(stokes)
P = stokes.P
phase_ratios = phase_ratios.center
θ_dτ = pt_stokes.θ_dτ

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
            εxx[i, j, k] + εII_0, εyy[i, j, k] - εII_0*0.5, εzz[i, j, k] - εII_0*0.5, 
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
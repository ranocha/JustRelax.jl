"""
    compute_ρg!(ρg, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`.
"""
@parallel_indices (i, j) function compute_ρg!(ρg, rheology, args)

    i1, j1 = i + 1, j + 1
    i2 = i + 2
    @inline av(T) = 0.25 * (T[i1, j] + T[i2, j] + T[i1, j1] + T[i2, j1]) - 273.0
    
    # index arguments for the current cell cell center
    args_ij = (; T = av(args.T), P=args.P[i, j])

    # compute ρg = density * gravity
    @inbounds ρg[i, j] = -compute_ρg(rheology, args_ij)

    return nothing
end

@parallel_indices (i, j, k) function compute_ρg!(ρg, rheology, args)

    av_T() = _av(args.T, i, j, k) - 273.0

    # index arguments for the current cell cell center
    args_ijk =(; T = av_T(), P=args.P[i, j, k])

    ρg[i, j, k] = compute_ρg(rheology, args_ijk)
    
    return nothing
end

"""
    compute_ρg!(ρg, phase_ratios, rheology, args)

Calculate the buoyance forces `ρg` for the given GeoParams.jl `rheology` object and correspondent arguments `args`. 
The `phase_ratios` are used to compute the density of the composite rheology.
"""
@parallel_indices (i, j) function compute_ρg!(ρg, phase_ratios, rheology, args)

    i1, j1 = i + 1, j + 1
    i2 = i + 2
    @inline av(T) = 0.25 * (T[i1,j] + T[i2,j] + T[i1,j1] + T[i2,j1]) - 273.0

    # index arguments for the current cell cell center
    args_ij = (; T = av(args.T), P=args.P[i, j])

    # compute ρg = density * gravity
    @inbounds ρg[i, j] = -compute_ρg(rheology, args_ij, phase_ratios[i, j])

    return nothing
end

@parallel_indices (i, j, k) function compute_ρg!(ρg, phase_ratios, rheology, args)

    av_T() = _av(args.T, i, j, k) - 273.0

    # index arguments for the current cell cell center
    args_ijk =(; T = av_T(), P=args.P[i, j, k])

    ρg[i, j, k] = compute_ρg(rheology, args_ijk, phase_ratios[i, j, k])
    return nothing
end

@inline compute_ρg(rheology, args) = compute_density(rheology, args) * compute_gravity(rheology[1])
@inline compute_ρg(rheology, args, phase_ratios) = compute_density(phase_ratios, rheology, args) * compute_gravity(rheology[1])

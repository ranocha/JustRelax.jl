using MPI

@parallel_indices (i, j, k) function init_T!(T, z)
    if z[k] == maximum(z)
        T[i, j, k] = 300.0
    elseif z[k] == minimum(z)
        T[i, j, k] = 3500.0
    else
        T[i, j, k] = 1900.0
    end
    return nothing
end

function diffusion_3D(;
    nx=32,
    ny=32,
    nz=32,
    lx=100e3,
    ly=100e3,
    lz=100e3,
    ρ0=3.3e3,
    Cp0=1.2e3,
    K0=3.0,
    init_MPI=MPI.Initialized() ? false : true,
    finalize_MPI=false,
)
    Myr = 1e6 * 3600 * 24 * 365.25
    ttot = 10 * Myr # total simulation time

    # Physical domain
    ni = (nx, ny, nz)
    li = (lx, ly, lz)  # domain length in x- and y-
    di = @. li / (ni-1) # grid step in x- and -y
    xci, xvi = lazy_grid(di, li; origin=(0, 0, -lz)) # nodes at the center and vertices of the cells

    igg = IGG(init_global_grid(nx, ny, nz; init_MPI=init_MPI)...) # init MPI

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal = ThermalArrays(ni)

    # physical parameters
    κ = K0/(ρ0*Cp0)
    ρ = @fill(ρ0, ni...)
    Cp = @fill(Cp0, ni...)
    K = @fill(K0, ni...)
    ρCp = @. Cp * ρ
    thermal_parameters = ThermalParameters(K, ρCp)

    # Boundary conditions
    thermal_bc = (frontal=true, lateral=true)

    @parallel (1:nx, 1:ny, 1:nz) init_T!(thermal.T, xvi[3])
    @parallel assign!(thermal.Told, thermal.T)

    dt = 0.5 / 4.1 * min(di...)^2 / κ

    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))

    # Physical time loop
    scatter(thermal.T[:], Z)
    while it < nt
        solve!(
            thermal,
            thermal_parameters,
            thermal_bc,
            di,
            dt
        )
        t += dt
        it += 1
        scatter!(thermal.T[:], Z)
    end

    finalize_global_grid(; finalize_MPI=finalize_MPI)

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), thermal.T
end

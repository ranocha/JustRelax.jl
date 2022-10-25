@parallel_indices (i, j) function init_T!(T, z)
    if z[j] == maximum(z)
        T[i, j] = 300.0
    elseif z[j] == minimum(z)
        T[i, j] = 3500.0
    else
        T[i, j] = 1900.0
    end
    return nothing
end

function diffusion_2D(; nx=32, ny=32, lx=100e3, ly=100e3, ρ0=3.3e3, Cp0=1.2e3, K0=3.0)
    Myr = 1e6 * 3600 * 24 * 365.25
    ttot = 10 * Myr # total simulation time

    # Physical domain
    ni = (nx, ny)
    li = (lx, ly)  # domain length in x- and y-
    di = @. li / (ni - 1) # grid step in x- and -y
    xci, xvi = lazy_grid(di, li; origin=(0, -ly)) # nodes at the center and vertices of the cells

    ## Allocate arrays needed for every Thermal Diffusion
    # general thermal arrays
    thermal = ThermalArrays(ni)

    # physical parameters
    κ = K0 / (ρ0 * Cp0)
    ρ = @fill(ρ0, ni...)
    Cp = @fill(Cp0, ni...)
    K = @fill(K0, ni...)
    ρCp = @. Cp * ρ
    thermal_parameters = ThermalParameters(K, ρCp)
    thermal_bc = (flux_x=true, flux_y=false)

    @parallel (1:nx, 1:ny) init_T!(thermal.T, xvi[2])
    @parallel assign!(thermal.Told, thermal.T)

    dt = 0.5 / 4.1 * min(di...)^2 / κ

    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))
    # Physical time loop
    while it < nt
        solve!(thermal, thermal_parameters, thermal_bc, di, dt)
        it += 1
        t += dt
    end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), thermal.T
end

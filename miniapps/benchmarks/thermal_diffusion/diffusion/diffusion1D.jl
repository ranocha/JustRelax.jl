@parallel_indices (i) function init_T!(T)
    if i == length(T)
        T[i] = 300.0
    elseif i == 1
        T[i] = 3500.0
    else
        T[i] = 1900.0
    end
    return nothing
end

function diffusion_1D(; nx=128, lx=100e3, ρ0=3.3e3, Cp0=1.2e3, K0=3.0)
    Myr = 1e6 * 3600 * 24 * 365.25
    ttot = 10 * Myr # total simulation time

    # Physical domain
    ni = (nx,)
    li = (lx,)  # domain length in x- and y-
    di = @. li / (ni - 1) # grid step in x- and -y
    xci, xvi = lazy_grid(di, li; origin=(-lx,)) # nodes at the center and vertices of the cells

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
    thermal_bc = (flux_y=false,)

    thermal.T .= 1900.0
    thermal.T[end] = 0.0
    thermal.T[1] = 3500.0
    @parallel assign!(thermal.Told, thermal.T)

    dt = 0.5 / 4.1 * min(di...)^2 / κ

    t = 0.0
    it = 0
    nt = Int(ceil(ttot / dt))
    # Physical time loop
    scatter(thermal.T, xvi[1])
    while it < nt
        solve!(thermal, thermal_parameters, thermal_bc, di, dt)
        it += 1
        t += dt
        scatter!(thermal.T, xvi[1])
    end

    return (ni=ni, xci=xci, xvi=xvi, li=li, di=di), thermal.T
end

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 3)

macro harm_xi_ρg(ix, iy, iz)
    return esc(
        :(
            1.0 / (1.0 / fx[$ix, $iy + 1, $iz + 1] + 1.0 / fx[$ix + 1, $iy + 1, $iz + 1]) *
            2.0
        ),
    )
end
macro harm_yi_ρg(ix, iy, iz)
    return esc(
        :(
            1.0 / (1.0 / fy[$ix + 1, $iy, $iz + 1] + 1.0 / fy[$ix + 1, $iy + 1, $iz + 1]) *
            2.0
        ),
    )
end
macro harm_zi_ρg(ix, iy, iz)
    return esc(
        :(
            1.0 / (1.0 / fz[$ix + 1, $iy + 1, $iz] + 1.0 / fz[$ix + 1, $iy + 1, $iz + 1]) *
            2.0
        ),
    )
end

function foo!(Rx, Ry, Rz, fx, fy, fz)

    for ix in axes(Rx,1), iy in axes(Rx,2), iz in axes(Rx,3)
        if (1 < ix < size(Rx,1)) && (1 < iy < size(Rx,1)) && (1 < iz < size(Rx,1)) 
            Rx[ix, iy, iz] = @harm_xi_ρg(ix, iy, iz)
            Ry[ix, iy, iz] = @harm_xi_ρg(ix, iy, iz)
            Rz[ix, iy, iz] = @harm_xi_ρg(ix, iy, iz)
        end
    end

    return nothing
end

function bar!(Rx, Ry, Rz, fx, fy, fz)

    # closures ---------------------------------------------
    @inline function harm_xi_ρg(ix, iy, iz)
        return 1.0 / (1.0 / fx[ix, iy + 1, iz + 1] + 1.0 / fx[ix + 1, iy + 1, iz + 1]) * 2.0
    end
    @inline function harm_yi_ρg(ix, iy, iz)
        return 1.0 / (1.0 / fy[ix + 1, iy, iz + 1] + 1.0 / fy[ix + 1, iy + 1, iz + 1]) * 2.0
    end
    @inline function harm_zi_ρg(ix, iy, iz)
        return 1.0 / (1.0 / fz[ix + 1, iy + 1, iz] + 1.0 / fz[ix + 1, iy + 1, iz + 1]) * 2.0
    end
    # ------------------------------------------------------

    for ix in axes(Rx,1), iy in axes(Rx,2), iz in axes(Rx,3)
        if (1 < ix < size(Rx,1)) && (1 < iy < size(Rx,1)) && (1 < iz < size(Rx,1)) 
            Rx[ix, iy, iz] = harm_xi_ρg(ix, iy, iz)
            Ry[ix, iy, iz] = harm_xi_ρg(ix, iy, iz)
            Rz[ix, iy, iz] = harm_xi_ρg(ix, iy, iz)
        end
    end

    return nothing
end


@parallel_indices (ix,iy,iz) function foo_ps!(Rx, Ry, Rz, fx, fy, fz)

    if (1 < ix < size(Rx,1)) && (1 < iy < size(Rx,1)) && (1 < iz < size(Rx,1)) 
        Rx[ix, iy, iz] = @harm_xi_ρg(ix, iy, iz)
        Ry[ix, iy, iz] = @harm_xi_ρg(ix, iy, iz)
        Rz[ix, iy, iz] = @harm_xi_ρg(ix, iy, iz)
    end

    return nothing
end

@parallel_indices (ix,iy,iz) function bar_ps!(Rx, Ry, Rz, fx, fy, fz)

    # closures ---------------------------------------------
    @inline function harm_xi_ρg(ix, iy, iz)
        @inbounds 1.0 / (1.0 / fx[ix, iy + 1, iz + 1] + 1.0 / fx[ix + 1, iy + 1, iz + 1]) * 2.0
    end
    @inline function harm_yi_ρg(ix, iy, iz)
        @inbounds 1.0 / (1.0 / fy[ix + 1, iy, iz + 1] + 1.0 / fy[ix + 1, iy + 1, iz + 1]) * 2.0
    end
    @inline function harm_zi_ρg(ix, iy, iz)
        @inbounds 1.0 / (1.0 / fz[ix + 1, iy + 1, iz] + 1.0 / fz[ix + 1, iy + 1, iz + 1]) * 2.0
    end
    # ------------------------------------------------------

    if (1 < ix < size(Rx,1)) && (1 < iy < size(Rx,1)) && (1 < iz < size(Rx,1)) 
        Rx[ix, iy, iz] = harm_xi_ρg(ix, iy, iz)
        Ry[ix, iy, iz] = harm_xi_ρg(ix, iy, iz)
        Rz[ix, iy, iz] = harm_xi_ρg(ix, iy, iz)
    end

    return nothing
end

n=128
Rx, Ry, Rz, fx, fy, fz = ntuple(i->rand(n,n), 6)

# @code_typed foo!(Rx, Ry, Rz, fx, fy, fz)
# @code_typed bar!(Rx, Ry, Rz, fx, fy, fz)

@btime foo!($Rx, $Ry, $Rz, $fx, $fy, $fz) # 81.233 ms (0 allocations: 0 bytes)

@btime bar!($Rx, $Ry, $Rz, $fx, $fy, $fz) # 81.139 ms (0 allocations: 0 bytes)

# @parallel (1:n,1:n,1:n) foo_ps!(Rx, Ry, Rz, fx, fy, fz)
# @parallel (1:n,1:n,1:n) bar_ps!(Rx, Ry, Rz, fx, fy, fz)

@btime @parallel (1:$n,1:$n,1:$n) foo_ps!($Rx, $Ry, $Rz, $fx, $fy, $fz) # 5.743  ms(21 allocations: 2.03 KiB)
@btime @parallel (1:$n,1:$n,1:$n) bar_ps!($Rx, $Ry, $Rz, $fx, $fy, $fz) # 5.782 ms (40 allocations: 2.05 KiB)


@parallel function foo(Rx, dt)
    @all(Rx) = @all(Rx)/dt
    return 
end

@parallel function foo1(Rx, dt)
    _dt() = 1/dt 

    @all(Rx) = @all(Rx)*_dt()
    return 
end


@parallel_indices (i,j) function foo2(Rx, dt)
    _dt() = 1/dt 

    Rx[i,j] = Rx[i,j]*_dt()
    return 
end

dt = rand()

@parallel foo1(Rx, dt)
@parallel (1:n,1:n) foo2(Rx, dt)

@btime bar($a)

a=rand(20)
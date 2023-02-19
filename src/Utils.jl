# unpacks fields of the struct x into a tuple
@generated function unpack(x::T) where T
    return quote
        Base.@_inline_meta
        tuple(_unpack(x, fieldnames($T))...)
    end
end
_unpack(a, fields) = (getfield(a, fi) for fi in fields)

macro unpack(x)
    return quote
        unpack($(esc(x)))
    end
end

@inline compute_dt(S::StokesArrays, di) = compute_dt(S.V, di, Inf)
@inline compute_dt(S::StokesArrays, di, dt_diff) = compute_dt(S.V, di, dt_diff)

@inline function compute_dt(V::Velocity, di, dt_diff)
    return compute_dt(unpack(V), di, dt_diff)
end

@inline function compute_dt(V, di, dt_diff)
    n = inv(length(V) + 0.1)
    dt_adv =
        mapreduce(x->x[1]/max(size(x[2])...), max, zip(di, V)) * n
    return min(dt_diff, dt_adv)
end

@inline tupleize(v) = (v,)
@inline tupleize(v::Tuple) = v

"""
    sumt(X::Vararg{NamedTuple, N}) where N

Sum the components of the `NamedTuple`s in `X`
"""
function sumt(X::Vararg{NamedTuple, N}) where N
    keys1 = keys(first(X))
    values1 = values(first(X))
    for i in 2:length(X)
        @assert keys1 == keys(X[i])
        values1 = values1 .+ values(X[i]) 
    end
    return (; zip(keys1, values1)...)
end

import Base: +, -, *, /
for op in (:+, :-, :*, :/)
    @eval begin
        ($op)(x::Number, X::NamedTuple) = (; zip(keys(X), ($op).(x, values(X)))...)
        ($op)(X::NamedTuple, x::Number) = ($op)(x, X) 
    end
end

# MACROS

"""
    copy(B, A)

Convinience macro to copy data from the array `A` into array `B`
"""
macro copy(B, A)
    return quote
        copyto!($(esc(B)), $(esc(A)))
    end
end

"""
    @add(I, args...)

Add `I` to the scalars in `args`    
"""
macro add(I, args...)
    quote
        Base.@_inline_meta
        v = (; $(esc.(args)...))
        return values(v) .+ $(esc(I))
    end
end

macro tuple(A)
    return quote
        _tuple($(esc(A)))
    end
end

_tuple(V::Velocity{<:AbstractArray{T,2}}) where {T} = V.Vx, V.Vy
_tuple(V::Velocity{<:AbstractArray{T,3}}) where {T} = V.Vx, V.Vy, V.Vz
_tuple(R::Residual{<:AbstractArray{T,2}}) where {T} = R.Rx, R.Ry
_tuple(R::Residual{<:AbstractArray{T,3}}) where {T} = R.Rx, R.Ry, R.Rz
_tuple(A::SymmetricTensor{<:AbstractArray{T,2}}) where {T} = A.xx, A.yy, A.xy
_tuple(A::SymmetricTensor{<:AbstractArray{T,3}}) where {T} = A.xx, A.yy, A.zz, A.yz, A.xz, A.xy

"""
    @idx(args...)

Make a linear range from `1` to `args[i]`, with `i ∈ [1, ..., n]`
"""
macro idx(args...)
    return quote
        _idx(tuple($(esc.(args)...))...)
    end
end

@inline Base.@pure _idx(args::Vararg{Int,N}) where {N} = ntuple(i -> 1:args[i], Val(N))
@inline Base.@pure _idx(args::NTuple{N,Int}) where {N} = ntuple(i -> 1:args[i], Val(N))

## Memory allocators

macro allocate(ni...)
    return esc(:(PTArray(undef, $(ni...))))
end

# Others

@parallel function assign!(B::AbstractArray{T,N}, A::AbstractArray{T,N}) where {T,N}
    @all(B) = @all(A)
    return nothing
end

# MPI reductions 

function mean_mpi(A)
    mean_l = mean(A)
    return MPI.Allreduce(mean_l, MPI.SUM, MPI.COMM_WORLD) / MPI.Comm_size(MPI.COMM_WORLD)
end

function norm_mpi(A)
    sum2_l = sum(A .^ 2)
    return sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD))
end

function minimum_mpi(A)
    min_l = minimum(A)
    return MPI.Allreduce(min_l, MPI.MIN, MPI.COMM_WORLD)
end

function maximum_mpi(A)
    max_l = maximum(A)
    return MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD)
end

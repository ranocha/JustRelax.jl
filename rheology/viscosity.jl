using Adapt

abstract type AbstractViscosity end

struct Viscosity{T,R1,R2} <: AbstractViscosity
    val::T
    linearviscous::R1
    elasticity::R2
end

Adapt.@adapt_structure Viscosity

struct LinearViscous{T}
    η0::T
end

@inline (η::LinearViscous)(args...) = η.η0

struct StagCreep{T}
    η0::T
    E::T
    V::T
end

@inline (η::StagCreep)(T, z) = η.η0 * exp(-η.E * T + η.V*(1-z))

# Elastic component
struct Elasticity{T}
    G::T
end

Base.getindex(η::AbstractViscosity, I::Vararg{Union{Integer, Colon},N}) where {N} = η.val[I...]

for op in (:+, :-, :/, :*)
    @eval begin
        Base.$(op)(η::AbstractViscosity, A::AbstractArray) = Broadcast(Base.$(op), η.val, A)
        Base.$(op)(A::AbstractArray, η::AbstractViscosity) = Broadcast(Base.$(op), A, η.val)
    end
end

for op in (:extrema, :minimum, :maximum, :unique, :length, :size, :eachindex)
    @eval begin
        Base.$(op)(η::AbstractViscosity) = Base.$(op)(η.val)
    end
end

@inline (η::Elasticity)(dt) = η.G * dt

"""
    compute_viscosity(η::T, args::NTuple{nargs,Tuple}) where {T <: AbstractViscosity, nargs}

Compute the effective viscosity of a `AbstractViscosity`
"""
@inline @generated function compute_viscosity(η::T, args) where {T<:AbstractViscosity}
    functors = fieldnames(T)[2:end]
    nf = length(functors)
    if nf > 1
        ex = 0.0
        for i in 1:nf
            functor = functors[i]
            ex = :(1 / (η.$(functor)(args[$i]...)) + $ex)
        end

        return :(1 / $(ex))
    else
        return :(η.$(fieldnames(T)[2])(args...))
    end
end

# case for several material phases that are identified with an integer
@inline @generated function compute_viscosity(
    η::T, phase::Integer, args
) where {T<:AbstractViscosity}
    functors = fieldnames(T)[2:end]
    nf = length(functors)
    if nf > 1
        ex = 0.0
        for i in 1:nf
            functor = functors[i]
            ex = :(1 / (getindex(η.$(functor), phase)(args[$i]...)) + $ex)
        end

        return :(1 / $(ex))
    else
        return :(getindex(η.$(fieldnames(T)[2]), phase)(args...))
    end
end

# apply (unsafe) getindex to any kind of NTuple
@inline function tupleindex(args::NTuple{N,Any}, I...) where {N}
    return tuple((unsafeindex(argi, I...) for argi in args)...)
end

@inline unsafeindex(args::NTuple{N,AbstractArray}, I...) where {N} = getindex.(args, I...)
@inline unsafeindex(args::Number, kwargs...) = args
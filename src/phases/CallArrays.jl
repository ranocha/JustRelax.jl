using CellArrays, StaticArrays 

@inline cellnum(A::CellArray) = prod(cellsize(A))
@inline cellaxes(A) = map(Base.oneto, cellnum(A))

"""
    element(A, element_indices..., cell_indices...)

Return a the element with `element_indices` of the Cell with `cell_indices` of the CellArray `A`.

## Arguments
- `element_indices::Int|NTuple{N,Int}`: the `element_indices` that designate the field in accordance with `A`'s cell type.
- `cell_indices::Int|NTuple{N,Int}`: the `cell_indices` that designate the cell in accordance with `A`'s cell type.
"""
Base.@propagate_inbounds @inline element(A::CellArray{SVector, N, D, T_elem}, i::Int, icell::Vararg{Int, Nc}) where {T_elem, N, Nc, D}  = viewelement(A, i, icell...)
Base.@propagate_inbounds @inline element(A::CellArray, i::T, j::T, icell::Vararg{Int, Nc})                    where {Nc, T<:Int}        = viewelement(A, i, j, icell...)

Base.@propagate_inbounds @inline function viewelement(A::CellArray{SMatrix{Ni, Nj, T, N_array}, N, D, T_elem}, i, j, icell::Vararg{Int, Nc}) where {Nc, Ni, Nj, N_array, T, N, T_elem, D} 
    idx_element = cart2ind((Ni, Nj), i, j)
    idx_cell = cart2ind(A.dims, icell...)
    _viewelement(A.data, idx_element, idx_cell)
end

Base.@propagate_inbounds @inline function viewelement(A::CellArray{SVector{Ni, T}, N, D, T_elem}, i, icell::Vararg{Int, Nc})  where {Nc, Ni, N, T, T_elem, D}
    idx_cell = cart2ind(A.dims, icell...)
    _viewelement(A.data, i, idx_cell)
end

Base.@propagate_inbounds @inline _viewelement(A::Array, idx, icell) = A[1, idx, icell]
Base.@propagate_inbounds @inline _viewelement(A, idx, icell)  = A[icell, idx, 1]

"""
    setelement!(A, x, element_indices..., cell_indices...)

Store the given value `x` at the given element with `element_indices` of the cell with the indices `cell_indices`

## Arguments
- `x::Number`: value to be stored in the index `element_indices` of the cell with `cell_indices`.
- `element_indices::Int|NTuple{N,Int}`: the `element_indices` that designate the field in accordance with `A`'s cell type.
- `cell_indices::Int|NTuple{N,Int}`: the `cell_indices` that designate the cell in accordance with `A`'s cell type.
"""
Base.@propagate_inbounds @inline function setelement!(A::CellArray{SMatrix{Ni, Nj, T, N_array}, N, D, T_elem}, x::T, i, j, icell::Vararg{Int, Nc})  where {Nc, Ni, Nj, N_array, T, N, T_elem, D} 
    idx_element = cart2ind((Ni, Nj), i, j)
    idx_cell = cart2ind(A.dims, icell...)
    _setelement!(A.data, x, idx_element, idx_cell)
end

Base.@propagate_inbounds @inline function setelement!(A::CellArray{SVector{Ni, T}, N, D, T_elem}, x::T, i, icell::Vararg{Int, Nc}) where {Nc, Ni, T, N, T_elem, D} 
    idx_cell = cart2ind(A.dims, icell...)
    _setelement!(A.data, x, i, idx_cell)
end

Base.@propagate_inbounds @inline _setelement!(A::Array, x, idx::Int, icell::Int)  = (A[1, idx, icell]= x)
Base.@propagate_inbounds @inline _setelement!(A, x, idx::Int, icell::Int) = (A[icell, idx, 1] = x)

## Helper functions

# """
#     cart2ind(A)
#
# Return the linear index of a `n`-dimensional array corresponding to the cartesian indices `I`
#
# """
@inline cart2ind(n::NTuple{N1, Int}, I::Vararg{Int, N2}) where {N1, N2} = LinearIndices(n)[CartesianIndex(I...)]
@inline cart2ind(ni::T, nj::T, i::T, j::T)               where T<:Int   = cart2ind( (ni, nj), i, j)
@inline cart2ind(ni::T, nj::T, nk::T, i::T, j::T, k::T)  where T<:Int   = cart2ind( (ni, nj, nk), i, j, k)
## Helper functions

# """
#     cart2ind(A)
#
# Return the linear index of a `n`-dimensional array corresponding to the cartesian indices `I`
#
# """
@inline cart2ind(n::NTuple{N1, Int}, I::Vararg{Int, N2}) where {N1, N2} = LinearIndices(n)[CartesianIndex(I...)]
@inline cart2ind(ni::T, nj::T, i::T, j::T)               where T<:Int   = cart2ind( (ni, nj), i, j)
@inline cart2ind(ni::T, nj::T, nk::T, i::T, j::T, k::T)  where T<:Int   = cart2ind( (ni, nj, nk), i, j, k)

## Fallbacks
import Base: getindex, setindex!

@inline element(A::Union{Array,CuArray}, I::Vararg{Int, N}) where {N} = getindex(A, I...)
@inline setelement!(A::Union{Array,CuArray}, x::Number, I::Vararg{Int, N}) where {N} = setindex!(A, x, I...)

## Convinience macros

macro cell(ex)
    ex = if ex.head === (:(=))
        _set(ex)
    else
        _get(ex)
    end
    :($(esc(ex)))
end

@inline _get(ex) = Expr(:call, element, ex.args...)
@inline _set(ex) = Expr(:call, setelement!, ex.args[1].args[1], ex.args[2], ex.args[1].args[2:end]...)
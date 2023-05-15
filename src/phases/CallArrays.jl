using CellArrays 

import Base.setindex!

Base.@propagate_inbounds @inline function setindex!(A::CellArray, x, cell::Int, I::Vararg{Int, N}) where N

    Base.@propagate_inbounds @inline f(A::Array, x, cell, idx) = A[1, cell, idx] = x
    Base.@propagate_inbounds @inline f(A, x, cell, idx) = A[idx, cell, 1] = x
    
    n = A.dims
    idx = LinearIndices(n)[CartesianIndex(I...)]

    return f(A.data, x, cell, idx)
end

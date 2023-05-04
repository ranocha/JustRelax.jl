struct Phases{T}
    vertex::T
    center::T
end

struct PhaseRatio{T}
    vertex::T
    center::T
    
    function PhaseRatio(ni, num_phases)
        center = @fill(0.0, ni..., celldims=(num_phases,)) 
        vertex = @fill(0.0, ni.+1..., celldims=(num_phases,)) 
        T = typeof(center)
        return new{T}(vertex, center)
    end
end


ni = 128,128
x = @fill(0.0, ni..., celldims=(3, ))

@inline nphases(::CellArray{StaticArraysCore.SArray{Tuple{N}, T, N1, N}, N2, N3, T_Array}) where {N, T, N1, N2, N3, T_Array} = N

@generated function phase_ratios(x)
    quote
        Base.@_inline_meta
        phases = 1, 2, 1, 2, 2 
        num_phases = Val(nphases(x))
        Base.@nexprs $num_phases i -> reps_i = (
            c = 0;
            for j in eachindex(phases)
                c += (phases[j] == i)
            end;
            c / length(phases)
        )
        Base.@ncall $num_phases tuple reps
    end
end

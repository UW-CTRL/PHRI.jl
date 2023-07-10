

matrix_to_vector_of_vectors(mat::VecOrMat{T}) where {T} = Vector{eltype(mat)}[eachrow(mat)...]
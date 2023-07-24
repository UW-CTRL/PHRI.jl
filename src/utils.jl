matrix_to_vector_of_vectors(mat::VecOrMat{T}) where {T} = Vector{eltype(mat)}[eachrow(mat)...]

vector_of_vectors_to_matrix(vec_of_vec::Vector{Vector{T}}) where {T} = mapreduce(permutedims, vcat, vec_of_vec)
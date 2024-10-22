matrix_to_vector_of_vectors(mat::VecOrMat{T}) where {T} = Vector{eltype(mat)}[eachrow(mat)...]

vector_of_vectors_to_matrix(vec_of_vec::Vector{Vector{T}}) where {T} = mapreduce(permutedims, vcat, vec_of_vec)

# Simulation utils
struct PlannerParams
    hps::PlannerHyperparameters
    opt_params::PlannerOptimizerParams
    counterpart_hps::PlannerHyperparameters
    counterpart_opt_params::PlannerOptimizerParams
end

struct HITLParams
    ego_planner_params::PlannerParams
    other_goal_state::Vector{Float64}
end

struct SFMParams
end

struct IPSimParams
    ego_planner_params::PlannerParams
    other_planner_params::PlannerParams
end

struct SFMSimParams
    ego_planner_params::PlannerParams
    other_planner_params::SFMParams
end

struct SimData
    sim_params::Union{HITLParams, SFMParams, IPSimParams}
    solve_times::Tuple{Vector{Float64}, Union{Vector{Float64}, Nothing}}
    ego_states::Matrix{Float64}
    ego_controls::Matrix{Float64}
    other_states::Matrix{Float64}
    other_controls::Matrix{Float64}
end

struct HITLSimData
    sim_params
    ego_states::Matrix{Float64}
    other_states::Matrix{Float64}
end


function mohrs_circle_states(dyn::DynamicallyExtendedUnicycle, initial_start_state::Vector{Float64}, initial_goal_state::Vector{Float64}, theta_resolution::T) where T<:Number
    list_entries = floor(Int, 2 * pi / theta_resolution)
    states_list = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, list_entries)

    isp = initial_start_state[1:2]
    igp = initial_goal_state[1:2]
    
    center = [(isp[1] + igp[1]) / 2, (isp[2] + igp[2]) / 2]
    R = sqrt((isp[1] - center[1])^2 + (isp[2] - center[2])^2)

    initial_θ = atan((isp[2] - center[2]) / (isp[1] - center[1]))

    for i in 0:floor(Int, list_entries / 2)
        states_list[i + 1] = ([center[1] - R * cos(theta_resolution * i - initial_θ), center[2] - R * sin(theta_resolution * i - initial_θ), theta_resolution * i, initial_start_state[end]], [center[1] + R * cos(theta_resolution * i - initial_θ), center[2] + R * sin(theta_resolution * i - initial_θ), theta_resolution * i, initial_start_state[end]])

        states_list[i + floor(Int, list_entries / 2)] = ([center[1] + R * cos(theta_resolution * i - initial_θ), center[2] + R * sin(theta_resolution * i - initial_θ), theta_resolution * i - pi, initial_start_state[end]], [center[1] - R * cos(theta_resolution * i - initial_θ), center[2] - R * sin(theta_resolution * i - initial_θ), theta_resolution * i - pi, initial_start_state[end]])
    end
    states_list
end

function compute_path_length(path::Matrix{Float64})
    if length(path[1, :]) >= 2
        path = path[:, 1:2]
    else
        throw(ArgumentError("Invalid Path --- must have at least two dimensions"))
    end

    N = length(path[:, 1])
    print(N)
    l = sum(norm(path[i, :] - path[i-1, :]) for i in 2:N)
end

function compute_path_length(path::Vector{Vector{Float64}})
    if length(path[1]) >= 2
        path = vector_of_vectors_to_matrix(path)[:, 1:2]
    else
        throw(ArgumentError("Invalid Path --- must have at least two dimensions"))
    end

    N = length(path[:, 1])
    l = sum(norm(path[i, :] - path[i-1, :]) for i in 2:N)
end

function accel_to_dynamically_extended_unicycle(accel::Vector{Float64}, θ::Float64, v::Vector{Float64})
    if norm(v) == 0. || norm(accel) == 0.
        a = norm(accel)
        ω = 0.0
        control = [ω, a]
    elseif norm(v) <= 0.1
        a = norm(accel)
        ω = dot(accel, v) / (norm(accel) * norm(v))
        control = [ω, a]
    else
        v = norm(v)
        a, ω = [cos(θ) sin(θ); -sin(θ) / v cos(θ) / v] * accel
        control = [ω, a]
    end

    control
end 
      
function wrap2pi(θ)
    θ_ = mod(θ, 2π)
    if θ_ > π
        return -2*π + θ_
    elseif θ_ < -π
        return θ_ + 2 * π
    else
        return θ_
    end
end   
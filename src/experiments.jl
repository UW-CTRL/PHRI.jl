include("dynamics.jl")
include("human.jl")
include("mpc.jl")
include("planner.jl")
include("planner_utils.jl")
include("sim.jl")
include("utils.jl")

using ProgressBars

function simulation_sweep(ego_ip::InteractionPlanner, other_ip::InteractionPlanner, ego_boundary_conditions::Vector{Tuple{Vector{Float64}, Vector{Float64}}}, other_boundary_conditions::Vector{Tuple{Vector{Float64}, Vector{Float64}}})
    runs = maximum([length(ego_boundary_conditions), length(other_boundary_conditions)])

    if length(ego_boundary_conditions) == 1 
        for i in 1:(runs - 1)
            push!(ego_boundary_conditions, ego_boundary_conditions[1])
        end
    elseif length(other_boundary_conditions) == 1
        for i in 1:(runs - 1)
            push!(other_boundary_conditions, other_boundary_conditions[1])
        end
    end

    if length(ego_boundary_conditions) != length(other_boundary_conditions)
        throw(ArgumentError("length of 'ego_boundary_conditions' and 'other_boundary_conditions' must match"))
    end

    runs_dict = Dict()

    for j in ProgressBar(1:runs)
        copied_ego_ip = deepcopy(ego_ip)
        copied_other_ip = deepcopy(other_ip)

        copied_ego_ip.ego_planner.incon.opt_params.initial_state = ego_boundary_conditions[j][1]
        copied_ego_ip.ego_planner.incon.opt_params.goal_state = ego_boundary_conditions[j][2]
        copied_other_ip.other_planner.incon.opt_params.initial_state = other_boundary_conditions[j][1]
        copied_other_ip.other_planner.incon.opt_params.goal_state = other_boundary_conditions[j][2]

        ego_params = PlannerParams(copied_ego_ip.ego_planner.incon.hps, copied_ego_ip.ego_planner.incon.opt_params)
        other_params = PlannerParams(copied_other_ip.ego_planner.incon.hps, copied_other_ip.ego_planner.incon.opt_params)

        sim_params = IPSimParams(ego_params, other_params)

        ego_states, ego_controls, other_states, other_controls = simulate(copied_ego_ip, copied_other_ip, 50)

        sim_data = SimData(sim_params, ego_states, ego_controls, other_states, other_controls)

        runs_dict["Run $(j)"] = sim_data

        # deleting variables
        copied_ego_ip = Nothing
        copied_other_ip = Nothing
        ego_params = Nothing
        other_params = Nothing
        sim_params = Nothing
        ego_states = Nothing
        ego_controls = Nothing
        other_states = Nothing
        other_controls = Nothing
        sim_data = Nothing
    end

    runs_dict
end
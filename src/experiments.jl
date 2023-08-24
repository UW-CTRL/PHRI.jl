include("dynamics.jl")
include("human.jl")
include("mpc.jl")
include("planner.jl")
include("planner_utils.jl")
include("sim.jl")
include("utils.jl")

using ProgressBars

struct SimMetrics
    avg_energy::Float64
    avg_accel_per_segment::Float64
    PI::Float64
    time_not_moving::Float64
    path_efficiency::Float64
    proactiveness::Float64
end

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

        ego_params = PlannerParams(copied_ego_ip.ego_planner.incon.hps, copied_ego_ip.ego_planner.incon.opt_params, copied_ego_ip.other_planner.incon.hps, copied_ego_ip.other_planner.incon.opt_params)
        other_params = PlannerParams(copied_other_ip.ego_planner.incon.hps, copied_other_ip.ego_planner.incon.opt_params, copied_other_ip.other_planner.incon.hps, copied_other_ip.other_planner.incon.opt_params)

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

function compute_average_control_effort(sim_data::SimData)
    sim_horizon = length(sim_data.ego_states[:, 1])
    ego_us = sim_data.ego_controls
    other_us = sim_data.other_controls

    Dict("Ego Avg Energy" => sum(norm(ego_us[t, :]) for t in 1:sim_horizon-1) / (sim_horizon-1), "Other Avg Energy" => sum(norm(other_us[t, :]) for t in 1:sim_horizon-1) / sim_horizon)
end

function compute_path_irregularity_index(sim_data::SimData)
    sim_horizon = length(sim_data.ego_states[:, 1])
    
    ego_dyn = sim_data.sim_params.ego_planner_params.hps.dynamics
    ego_xs = sim_data.ego_states
    ego_us = sim_data.ego_controls
    ego_goal = sim_data.sim_params.ego_planner_params.opt_params.goal_state
    
    other_dyn = sim_data.sim_params.other_planner_params.hps.dynamics
    other_xs = sim_data.other_states
    other_us = sim_data.other_controls
    other_goal = sim_data.sim_params.other_planner_params.opt_params.goal_state
    
    ego_PI = 0.
    other_PI = 0.

    ego_ideal_problem = IdealProblem(ego_dyn, ego_hps, ego_opt_params)
    other_ideal_problem = IdealProblem(other_dyn, other_hps, other_opt_params)

    ego_velocities = get_velocity(ego_dyn, matrix_to_vector_of_vectors(ego_xs)[1:end-1], matrix_to_vector_of_vectors(ego_us))
    other_velocities = get_velocity(other_dyn, matrix_to_vector_of_vectors(other_xs)[1:end-1], matrix_to_vector_of_vectors(other_us))

    for i in 1:sim_horizon-1
        # ego PI
        copied_ego_ideal_problem = deepcopy(ego_ideal_problem)
        copied_ego_ideal_problem.opt_params = PlannerOptimizerParams(ego_dyn, ego_hps, ego_xs[i, :], ego_goal, "ECOS")
        solve(copied_ego_ideal_problem, iterations=3)

        ego_state = vector_of_vectors_to_matrix(copied_ego_ideal_problem.opt_params.previous_states)[1, :]
        ego_control = vector_of_vectors_to_matrix(copied_ego_ideal_problem.opt_params.previous_controls)[1, :]

        ego_ideal_velocity = get_velocity(ego_dyn, ego_state, ego_control)

        if norm(ego_velocities[i]) * norm(ego_ideal_velocity) != 0
            ego_PI += acos(dot(ego_velocities[i], ego_ideal_velocity) / (norm(ego_velocities[i]) * norm(ego_ideal_velocity)))
        end

        copied_ego_ideal_problem = Nothing

        # other PI
        copied_other_ideal_problem = deepcopy(other_ideal_problem)
        copied_other_ideal_problem.opt_params = PlannerOptimizerParams(other_dyn, other_hps, other_xs[i, :], other_goal, "ECOS")
        solve(copied_other_ideal_problem, iterations=3)

        other_state = vector_of_vectors_to_matrix(copied_other_ideal_problem.opt_params.previous_states)[1, :]
        other_control = vector_of_vectors_to_matrix(copied_other_ideal_problem.opt_params.previous_controls)[1, :]

        other_ideal_velocity = get_velocity(other_dyn, other_state, other_control)

        if norm(other_velocities[i]) * norm(other_ideal_velocity) != 0
            other_PI += acos(dot(other_velocities[i], other_ideal_velocity) / (norm(other_velocities[i]) * norm(other_ideal_velocity)))
        end

        copied_other_ideal_problem = Nothing
    end

    Dict("ego PI" => ego_PI, "other PI" => other_PI)
end

function compute_average_acceleration_per_segment(sim_data::SimData)
    dt = sim_data.sim_params.ego_planner_params.hps.dynamics.dt
    sim_horizon = length(sim_data.ego_states[:, 1])
    
    ego_dyn = sim_data.sim_params.ego_planner_params.hps.dynamics
    ego_xs = sim_data.ego_states
    ego_us = sim_data.ego_controls
    
    other_dyn = sim_data.sim_params.other_planner_params.hps.dynamics
    other_xs = sim_data.other_states
    other_us = sim_data.other_controls

    ego_velocities = get_velocity(ego_dyn, matrix_to_vector_of_vectors(ego_xs)[1:end-1], matrix_to_vector_of_vectors(ego_us))
    other_velocities = get_velocity(other_dyn, matrix_to_vector_of_vectors(other_xs)[1:end-1], matrix_to_vector_of_vectors(other_us))

    ego_a = abs(sum((norm(ego_velocities[i]) - norm(ego_velocities[i-1])) / dt for i in 2:sim_horizon-1) / (sim_horizon - 1))
    other_a = abs(sum((norm(other_velocities[i]) - norm(other_velocities[i-1])) / dt for i in 2:sim_horizon-1) / (sim_horizon - 1))

    Dict("ego average acceleration" => ego_a, "other average acceleration" => other_a)
end

function compute_path_efficiency(sim_data::SimData)
    sim_horizon = length(sim_data.ego_states[:, 1])

    ego_dyn = sim_data.sim_params.ego_planner_params.hps.dynamics
    ego_xs = sim_data.ego_states
    ego_goal = sim_data.sim_params.ego_planner_params.opt_params.goal_state

    other_dyn = sim_data.sim_params.other_planner_params.hps.dynamics
    other_xs = sim_data.other_states
    other_goal = sim_data.sim_params.other_planner_params.opt_params.goal_state

    ego_hps.time_horizon = sim_horizon
    other_hps.time_horizon = sim_horizon

    ego_opt_params = PlannerOptimizerParams(ego_dyn, ego_hps, ego_xs[1, :], ego_goal, "ECOS")
    other_opt_params = PlannerOptimizerParams(other_dyn, other_hps, other_xs[1, :], other_goal, "ECOS")

    ego_ideal_problem = IdealProblem(ego_dyn, ego_hps, ego_opt_params)
    other_ideal_problem = IdealProblem(other_dyn, other_hps, other_opt_params)

    solve(ego_ideal_problem, iterations=3)
    solve(other_ideal_problem, iterations=3)

    ego_ideal_length = compute_path_length(ego_ideal_problem.opt_params.previous_states)
    other_ideal_length = compute_path_length(other_ideal_problem.opt_params.previous_states)

    ego_path_efficiency = compute_path_length(ego_xs) / ego_ideal_length
    other_path_efficiency = compute_path_length(other_xs) / other_ideal_length

    Dict("Ego Path Efficiency" => ego_path_efficiency, "Other Path Efficiency" => other_path_efficiency)
end

function evaluate_sim(sim_data::SimData)
end
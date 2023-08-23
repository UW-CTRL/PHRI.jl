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

function evaluate_sim(sim_data::SimData)
    dt = sim_data.sim_params.ego_planner_params.hps.dynamics.dt
    sim_horizon = length(sim_data.ego_states[:, 1])
    
    ego_dyn = sim_data.sim_params.ego_planner_params.hps.dynamics
    ego_xs = sim_data.ego_states
    ego_us = sim_data.ego_controls
    ego_goal = sim_data.sim_params.ego_planner_params.opt_params.goal_state
    
    other_dyn = sim_data.sim_params.other_planner_params.hps.dynamics
    other_xs = sim_data.other_states
    other_us = sim_data.other_controls
    other_goal = sim_data.sim_params.other_planner_params.opt_params.goal_state

    ego_velocities = get_velocity(ego_dyn, matrix_to_vector_of_vectors(ego_xs)[1:end-1], matrix_to_vector_of_vectors(ego_us))
    other_velocities = get_velocity(other_dyn, matrix_to_vector_of_vectors(other_xs)[1:end-1], matrix_to_vector_of_vectors(other_us))

    # avg energy
    ego_avg_energy = sum(norm(ego_us[t, :]) * dt for t in 1:sim_horizon-1) / sim_horizon
    other_avg_energy = sum(norm(other_us[t, :]) * dt for t in 1:sim_horizon-1) / sim_horizon

    # Path Irregularity Index (PI)
    ego_PI = sum(acos(dot(ego_velocities[i][:], ego_goal[1:2] - ego_xs[i, 1:2]) / (norm(ego_velocities[i][:]) * norm(ego_goal[1:2] - ego_xs[i, 1:2]))) for i in 1:sim_horizon-1)
    other_PI = sum(acos(dot(other_velocities[i][:], other_goal[1:2] - other_xs[i, 1:2]) / (norm(other_velocities[i][:]) * norm(other_goal[1:2] - other_xs[i, 1:2]))) for i in 1:sim_horizon-1)

    # avg acceleration per segment
    ego_avg_accel_per_segment = abs(sum((norm(ego_velocities[i]) - norm(ego_velocities[i-1])) / dt for i in 2:sim_horizon-1) / (sim_horizon - 1))
    other_avg_accel_per_segment = abs(sum((norm(other_velocities[i]) - norm(other_velocities[i-1])) / dt for i in 2:sim_horizon-1) / (sim_horizon - 1))

    # time not moving
    ego_tnm = 0.0
    for i in 2:sim_horizon
        if norm(ego_xs[i, 1:2] - ego_xs[i-1, 1:2]) / dt < 0.1
            ego_tnm += dt
        end
    end
    other_tnm = 0.0
    for i in 2:sim_horizon
        if norm(other_xs[i, 1:2] - other_xs[i-1, 1:2]) / dt < 0.1
            other_tnm += dt
        end
    end
end
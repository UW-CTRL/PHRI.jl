include("dynamics.jl")
include("human.jl")
include("mpc.jl")
include("planner.jl")
include("planner_utils.jl")
include("sim.jl")
include("utils.jl")

using ProgressBars

struct SimMetrics
    control_effort::Dict{String, Float64}
    PI::Dict{String, Float64}
    avg_accel::Dict{String, Float64}
    PE::Dict{String, Float64}
    min_dist::Dict{String, Float64}
    ttc::Dict{String, Vector{Float64}}
    θ::Dict{String, Vector{Float64}}
    dθ_dt::Dict{String, Vector{Float64}}
    time::Dict{String, Any}
end

function simulation_sweep(ego_ip::InteractionPlanner, other_ip::InteractionPlanner, sim_horizon, ego_boundary_conditions::Vector{Tuple{Vector{Float64}, Vector{Float64}}}, other_boundary_conditions::Vector{Tuple{Vector{Float64}, Vector{Float64}}})
    runs = maximum([length(ego_boundary_conditions), length(other_boundary_conditions)])

    ego_ego_hps = ego_ip.ego_planner.incon.hps
    ego_other_hps = ego_ip.other_planner.incon.hps
    other_ego_hps = other_ip.ego_planner.incon.hps
    other_other_hps = other_ip.other_planner.incon.hps

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

    runs_dict = Dict{String, SimData}()

    for j in ProgressBar(1:runs)
        sim_ego_ip = InteractionPlanner(ego_ego_hps, ego_other_hps, ego_boundary_conditions[j][1], other_boundary_conditions[j][1], ego_boundary_conditions[j][2], other_boundary_conditions[j][2], "ECOS")
        sim_other_ip = InteractionPlanner(other_ego_hps, other_other_hps, other_boundary_conditions[j][1], ego_boundary_conditions[j][1], other_boundary_conditions[j][2], ego_boundary_conditions[j][2], "ECOS")

        ego_params = PlannerParams(sim_ego_ip.ego_planner.incon.hps, sim_ego_ip.ego_planner.incon.opt_params, sim_ego_ip.other_planner.incon.hps, sim_ego_ip.other_planner.incon.opt_params)
        other_params = PlannerParams(sim_other_ip.ego_planner.incon.hps, sim_other_ip.ego_planner.incon.opt_params, sim_other_ip.other_planner.incon.hps, sim_other_ip.other_planner.incon.opt_params)

        sim_params = IPSimParams(ego_params, other_params)

        ego_states, ego_controls, other_states, other_controls, solve_time = simulate(sim_ego_ip, sim_other_ip, sim_horizon)

        sim_data = SimData(sim_params, solve_time, ego_states, ego_controls, other_states, other_controls)

        runs_dict["Run $(j)"] = sim_data

        # deleting variables
        sim_ego_ip = nothing
        sim_other_ip = nothing
        ego_params = nothing
        other_params = nothing
        sim_params = nothing
        ego_states = nothing
        ego_controls = nothing
        other_states = nothing
        other_controls = nothing
        sim_data = nothing
    end

    runs_dict
end

function compute_average_control_effort(sim_data::SimData)
    sim_horizon = length(sim_data.ego_states[:, 1])
    ego_us = sim_data.ego_controls
    other_us = sim_data.other_controls

    Dict("Ego Avg Control Effort" => sum(norm(ego_us[t, :]) for t in 1:sim_horizon-1) / (sim_horizon-1), "Other Avg Control Effort" => sum(norm(other_us[t, :]) for t in 1:sim_horizon-1) / sim_horizon)
end

function compute_path_irregularity_index(sim_data::SimData)
    sim_horizon = length(sim_data.ego_states[:, 1])

    ego_hps = sim_data.sim_params.ego_planner_params.hps
    ego_opt_params = sim_data.sim_params.ego_planner_params.opt_params
    ego_dyn = sim_data.sim_params.ego_planner_params.hps.dynamics
    ego_xs = sim_data.ego_states
    ego_us = sim_data.ego_controls
    ego_goal = sim_data.sim_params.ego_planner_params.opt_params.goal_state
    
    other_hps = sim_data.sim_params.other_planner_params.hps
    other_opt_params = sim_data.sim_params.other_planner_params.opt_params
    other_dyn = sim_data.sim_params.other_planner_params.hps.dynamics
    other_xs = sim_data.other_states
    other_us = sim_data.other_controls
    other_goal = sim_data.sim_params.other_planner_params.opt_params.goal_state
    
    ego_PI = 0.
    other_PI = 0.

    ego_velocities = get_velocity(ego_dyn, matrix_to_vector_of_vectors(ego_xs)[1:end-1], matrix_to_vector_of_vectors(ego_us))
    other_velocities = get_velocity(other_dyn, matrix_to_vector_of_vectors(other_xs)[1:end-1], matrix_to_vector_of_vectors(other_us))

    for i in 1:sim_horizon-1
        # ego PI
        ego_opt_params = PlannerOptimizerParams(ego_dyn, ego_hps, ego_xs[i, :], ego_goal, "ECOS")
        ego_ideal_problem = IdealProblem(ego_dyn, ego_hps, ego_opt_params)
        solve(ego_ideal_problem, iterations=3)

        ego_state = vector_of_vectors_to_matrix(ego_ideal_problem.opt_params.previous_states)[1, :]
        ego_control = vector_of_vectors_to_matrix(ego_ideal_problem.opt_params.previous_controls)[1, :]

        ego_ideal_velocity = get_velocity(ego_dyn, ego_state, ego_control)

        if norm(ego_velocities[i]) * norm(ego_ideal_velocity) != 0
            ego_PI += acos(dot(ego_velocities[i], ego_ideal_velocity) / (norm(ego_velocities[i]) * norm(ego_ideal_velocity)))
        end

        ego_ideal_problem = nothing

        # other PI
        other_opt_params = PlannerOptimizerParams(other_dyn, other_hps, other_xs[i, :], other_goal, "ECOS")
        other_ideal_problem = IdealProblem(other_dyn, other_hps, other_opt_params)
        solve(other_ideal_problem, iterations=3)

        other_state = vector_of_vectors_to_matrix(other_ideal_problem.opt_params.previous_states)[1, :]
        other_control = vector_of_vectors_to_matrix(other_ideal_problem.opt_params.previous_controls)[1, :]

        other_ideal_velocity = get_velocity(other_dyn, other_state, other_control)

        if norm(other_velocities[i]) * norm(other_ideal_velocity) != 0
            other_PI += acos(dot(other_velocities[i], other_ideal_velocity) / (norm(other_velocities[i]) * norm(other_ideal_velocity)))
        end

        other_ideal_problem = nothing
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
    ego_hps = sim_data.sim_params.ego_planner_params.hps

    other_dyn = sim_data.sim_params.other_planner_params.hps.dynamics
    other_xs = sim_data.other_states
    other_goal = sim_data.sim_params.other_planner_params.opt_params.goal_state
    other_hps = sim_data.sim_params.other_planner_params.hps

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

function compute_minimum_distance(sim_data::SimData)
    sim_horizon = length(sim_data.ego_states[:, 1])
    ego_xs = sim_data.ego_states
    other_xs = sim_data.other_states

    state_difference = ego_xs[:, 1:2] - other_xs[:, 1:2]
    min_distance = minimum([norm(state_difference[i, :]) for i in 1:sim_horizon])

    Dict("Min Distance" => min_distance)
end

function compute_time_to_collision(sim_data::SimData)
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

    time_to_collision = Vector{Float64}(undef, sim_horizon-1)

    ego_velos = get_velocity(ego_dyn, matrix_to_vector_of_vectors(ego_xs)[1:end-1], matrix_to_vector_of_vectors(ego_us))
    other_velos = get_velocity(other_dyn, matrix_to_vector_of_vectors(other_xs)[1:end-1], matrix_to_vector_of_vectors(other_us))

    for i in 1:sim_horizon-1
        collision_point = nothing
        ego_projected_pos = [ego_xs[i, 1:2] + ego_velos[i][:] * t for t in 0:sim_horizon-1]
        other_projected_pos = [other_xs[i, 1:2] + other_velos[i][:] * t for t in 0:sim_horizon-1]
        distances = norm.(ego_projected_pos - other_projected_pos)
        for j in 1:sim_horizon
            if distances[j] < 0.75
                collision_point = j
                break
            end
        end

        if collision_point !== nothing
            relative_speed = norm(ego_velos[i][:] - other_velos[i][:])
            distance_to_collision = distances[1] - distances[collision_point]
            time_to_collision[i] = distance_to_collision / relative_speed
        else
            time_to_collision[i] = NaN
        end
    end
    
    Dict("Time to collision" => time_to_collision)
end

function compute_θ(sim_data)
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


    ego_velos = get_velocity(ego_dyn, matrix_to_vector_of_vectors(ego_xs)[1:end-1], matrix_to_vector_of_vectors(ego_us))
    other_velos = get_velocity(other_dyn, matrix_to_vector_of_vectors(other_xs)[1:end-1], matrix_to_vector_of_vectors(other_us))

    ego_theta = [acos(dot(ego_velos[i][:], [1., 0]) / (norm(ego_velos[i][:]))) for i in 1:sim_horizon-1]
    other_theta = [acos(dot(other_velos[i][:], [1., 0]) / (norm(other_velos[i][:]))) for i in 1:sim_horizon-1]

    Dict("Ego θ" => ego_theta, "Other θ" => other_theta)
end

function compute_dθ_dt(sim_data::SimData)
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

    ego_θ = compute_θ(sim_data)["Ego θ"]
    other_θ = compute_θ(sim_data)["Other θ"]

    ego_dθ_dt = [abs((ego_θ[t] - ego_θ[t-1]))/ dt for t in 2:sim_horizon-1]
    other_dθ_dt = [abs((other_θ[t] - other_θ[t-1])) / dt for t in 2:sim_horizon-1]

    Dict("Ego dθ/dt" => ego_dθ_dt, "Other dθ/dt" => other_dθ_dt)
end

function compute_time(sim_data)
    dt = sim_data.sim_params.ego_planner_params.hps.dynamics.dt
    sim_horizon = length(sim_data.ego_states[:, 1])

    ego_solve_times = sim_data.solve_times[1]
    ego_avg_solve_time = sum(ego_solve_times) / (sim_horizon - 1)
    ego_max_solve_time = maximum(ego_solve_times)
    ego_planning_deadline_overruns = length(findall(x -> x >= 0.1, ego_solve_times))

    timing_dict = Dict("Ego Solve Times" => ego_solve_times, "Ego Average Solve Time" => ego_avg_solve_time, "Ego Max Solve Time" => ego_max_solve_time, "Ego Planning Deadline Overruns" => ego_planning_deadline_overruns)

    if sim_data.solve_times[2] !== nothing
        other_solve_times = sim_data.solve_times[2]
        other_avg_solve_time = sum(other_solve_times) / (sim_horizon - 1)
        other_max_solve_time = maximum(other_solve_times)
        other_planning_deadline_overruns = length(findall(x -> x >= 0.1, other_solve_times))
        timing_dict["Other Solve Times"] = other_solve_times
        timing_dict["Other Average Solve Time"] = other_avg_solve_time
        timing_dict["Other Max Solve Time"] = other_max_solve_time
        timing_dict["Other Planning Deadline Overruns"] = other_planning_deadline_overruns
    end

    timing_dict
end

function evaluate_sim(sim_data::SimData)
    # function that returns all metrics from a simulation run
    metrics = SimMetrics(compute_average_control_effort(sim_data),
                        compute_path_irregularity_index(sim_data),
                        compute_average_acceleration_per_segment(sim_data),
                        compute_path_efficiency(sim_data),
                        compute_minimum_distance(sim_data),
                        compute_time_to_collision(sim_data),
                        compute_θ(sim_data),
                        compute_dθ_dt(sim_data),
                        compute_time(sim_data)
    )
end
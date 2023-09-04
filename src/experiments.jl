include("dynamics.jl")
include("human.jl")
include("mpc.jl")
include("planner.jl")
include("planner_utils.jl")
include("sim.jl")
include("utils.jl")

using ProgressBars
using Gtk
using Cairo


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
    plots::Dict{String, Plots.Plot{Plots.GRBackend}}
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

function simulation_sweep(ego::DynamicallyExtendedUnicycle, other_ip::InteractionPlanner, sim_horizon, ego_boundary_conditions::Vector{Tuple{Vector{Float64}, Vector{Float64}}}, other_boundary_conditions::Vector{Tuple{Vector{Float64}, Vector{Float64}}}; p=2., q=2., τ=2., ψ=pi/6, c=0.3)
    runs = maximum([length(ego_boundary_conditions), length(other_boundary_conditions)])

    ego_ego_hps = other_ip.other_planner.incon.hps
    ego_other_hps = other_ip.ego_planner.incon.hps
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
        sim_other_ip = InteractionPlanner(other_ego_hps, other_other_hps, other_boundary_conditions[j][1], ego_boundary_conditions[j][1], other_boundary_conditions[j][2], ego_boundary_conditions[j][2], "ECOS")

        ego_params = PlannerParams(sim_other_ip.other_planner.incon.hps, sim_other_ip.other_planner.incon.opt_params, sim_other_ip.ego_planner.incon.hps, sim_other_ip.ego_planner.incon.opt_params)
        other_params = PlannerParams(sim_other_ip.ego_planner.incon.hps, sim_other_ip.ego_planner.incon.opt_params, sim_other_ip.other_planner.incon.hps, sim_other_ip.other_planner.incon.opt_params)

        sim_params = IPSimParams(ego_params, other_params)

        ego_states, ego_controls, other_states, other_controls = simulate_human_social_forces(ego, sim_other_ip, ego_boundary_conditions[j][1], ego_boundary_conditions[j][2], sim_horizon, p=2., q=2., τ=2., ψ=pi/6, c=0.3)

        sim_data = SimData(sim_params, ([0.], nothing), ego_states, ego_controls, other_states, other_controls)

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

function run_experiment(ego_ip::InteractionPlanner, other_ip::InteractionPlanner, sim_horizon, ego_boundary_conditions::Vector{Tuple{Vector{Float64}, Vector{Float64}}}, other_boundary_conditions::Vector{Tuple{Vector{Float64}, Vector{Float64}}}, save_path=""::String)
    start_time = time()
    println("-" ^ 80)
    println("Running Simulations")
    println("-" ^ 80)
    sweep_data = simulation_sweep(ego_ip, other_ip, sim_horizon, ego_boundary_conditions, other_boundary_conditions)
    println("-" ^ 80)
    println("Evaluating Simulations")
    println("-" ^ 80)
    metrics = evaluate_sim(sweep_data)
    end_time = time()
    if save_path != ""
        serialize(save_path, metrics)
    end

    print("Experiment finished in $(end_time - start_time)")

    metrics
end

function run_experiment(ego::DynamicallyExtendedUnicycle, other_ip::InteractionPlanner, sim_horizon, ego_boundary_conditions::Vector{Tuple{Vector{Float64}, Vector{Float64}}}, other_boundary_conditions::Vector{Tuple{Vector{Float64}, Vector{Float64}}}; p=2., q=2., τ=2., ψ=pi/6, c=0.3, save_path=""::String)
    start_time = time()
    println("-" ^ 80)
    println("Running Simulations")
    println("-" ^ 80)
    sweep_data = simulation_sweep(ego, other_ip, sim_horizon, ego_boundary_conditions, other_boundary_conditions, p=2., q=2., τ=2., ψ=pi/6, c=0.3)
    println("-" ^ 80)
    println("Evaluating Simulations")
    println("-" ^ 80)
    metrics = evaluate_sim(sweep_data)
    end_time = time()
    if save_path != ""
        serialize(save_path, metrics)
    end

    print("Experiment finished in $(end_time - start_time)")

    metrics
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
            ego_PI += acos(round(dot(ego_velocities[i], ego_ideal_velocity) / (norm(ego_velocities[i]) * norm(ego_ideal_velocity)), digits=4))
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

function compute_time(sim_data::SimData)
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

function plot_ttc(ttc::Dict{String, Vector{Float64}})
    ttc_list = ttc["Time to collision"]
    N = length(ttc_list)

    ttc_plot = scatter(1:N, ttc_list, color=:red, markersize=3, label="Time to Collision", ylabel="Projected time to collision (s)", xlabel="Simulation timepoint", title="Time to collision plot", margin=5mm, ylims=[-1, 31])

    ttc_plot
end

function plot_θ(θs::Dict{String, Vector{Float64}})
    ego_θs = θs["Ego θ"]
    other_θs = θs["Other θ"]
    N = length(ego_θs)

    θ_plot = plot(1:N, ego_θs, color=:blue, linewidth=2, label="Ego θ", xlabel="Simulation timepoint", ylabel="θ (rad)", title="θ Plot", margin=5mm, ylims=[-0.1, π+0.1])
    plot!(θ_plot, 1:N, other_θs, color=:red, linewidth=2, label="Other θ")

    θ_plot
end

function plot_dθ_dt(dθ_dts::Dict{String, Vector{Float64}})
    ego_dθ_dts = dθ_dts["Ego dθ/dt"]
    other_dθ_dts = dθ_dts["Other dθ/dt"]
    N = length(ego_dθ_dts)

    dθ_dt_plot = plot(1:N, ego_dθ_dts, color=:blue, linewidth=2, label="Ego dθ/dt", xlabel="Simulation timepoint", ylabel="dθ/dt (rad)", title="dθ/dt Plot", margin=5mm, ylims=[-0.1, 1.5])
    plot!(dθ_dt_plot, 1:N, other_dθ_dts, color=:red, linewidth=2, label="Other dθ/dt")

    dθ_dt_plot
end

function combine_sim_data_plots(sim_data::SimData)
    width = 2000
    height = 800
    overview_plot = plot_solve_solution(sim_data)

    ttc_plot = plot_ttc(compute_time_to_collision(sim_data))
    θ_plot = plot_θ(compute_θ(sim_data))
    dθ_dt_plot = plot_dθ_dt(compute_dθ_dt(sim_data))

    l = @layout [a; b c d]
    plot(overview_plot, ttc_plot, θ_plot, dθ_dt_plot, layout=l, size=(width, height), margin=10mm)
end

function evaluate_sim(sim_data::SimData)
    # function that returns all metrics from a simulation run
    plot_dict = Dict{String, Plots.Plot{Plots.GRBackend}}()
    plot_dict["Overview Plot"] = plot_solve_solution(sim_data)
    plot_dict["ttc"] = plot_ttc(compute_time_to_collision(sim_data))
    plot_dict["θ"] = plot_θ(compute_θ(sim_data))
    plot_dict["dθ/dt"] = plot_dθ_dt(compute_dθ_dt(sim_data))
    plot_dict["Combined Plot"] = combine_sim_data_plots(sim_data)

    metrics = SimMetrics(compute_average_control_effort(sim_data),
                        compute_path_irregularity_index(sim_data),
                        compute_average_acceleration_per_segment(sim_data),
                        compute_path_efficiency(sim_data),
                        compute_minimum_distance(sim_data),
                        compute_time_to_collision(sim_data),
                        compute_θ(sim_data),
                        compute_dθ_dt(sim_data),
                        compute_time(sim_data),
                        plot_dict
    )
end

function evaluate_sim(sim_data_sweep::Dict{String, SimData})
    N_runs = length(sim_data_sweep)
    metrics_dict = Dict{String, SimMetrics}()

    for i in ProgressBar(1:N_runs)
        plot_dict = Dict{String, Plots.Plot{Plots.GRBackend}}()
        plot_dict["Overview Plot"] = plot_solve_solution(sim_data_sweep["Run $(i)"])
        plot_dict["ttc"] = plot_ttc(compute_time_to_collision(sim_data_sweep["Run $(i)"]))
        plot_dict["θ"] = plot_θ(compute_θ(sim_data_sweep["Run $(i)"]))
        plot_dict["dθ/dt"] = plot_dθ_dt(compute_dθ_dt(sim_data_sweep["Run $(i)"]))
        plot_dict["Combined Plot"] = combine_sim_data_plots(sim_data_sweep["Run $(i)"])

        metrics = SimMetrics(compute_average_control_effort(sim_data_sweep["Run $(i)"]),
                            compute_path_irregularity_index(sim_data_sweep["Run $(i)"]),
                            compute_average_acceleration_per_segment(sim_data_sweep["Run $(i)"]),
                            compute_path_efficiency(sim_data_sweep["Run $(i)"]),
                            compute_minimum_distance(sim_data_sweep["Run $(i)"]),
                            compute_time_to_collision(sim_data_sweep["Run $(i)"]),
                            compute_θ(sim_data_sweep["Run $(i)"]),
                            compute_dθ_dt(sim_data_sweep["Run $(i)"]),
                            compute_time(sim_data_sweep["Run $(i)"]),
                            plot_dict
        )
        metrics_dict["Run $(i)"] = metrics
        metrics = nothing
    end

    metrics_dict
end

function format_data(data::Vector{Tuple{String, String}})
    max_label_width = maximum(length.([d[1] for d in data]))
    formatted_lines = []
    for d in data
        if d[1] == ""
            push!(formatted_lines, "<b>-</b>"^(max_label_width * 2))
        else
            push!(formatted_lines, string("<b>", d[1], "</b>", d[2]))
        end
    end
    formatted_string = join(formatted_lines, "\n")
    return formatted_string
end

function display_data(sim_sweep::Dict{String, SimMetrics}; h = 2000, w = 1400)
    io_combined = PipeBuffer()

    N_runs = length(sim_sweep)
    combined_plot(n) = show(io_combined, MIME("image/png"),  sim_sweep["Run $(n)"].plots["Combined Plot"])

    win = GtkWindow("Sim Sweep Data Display", h, w)

    vbox = GtkBox(:v, spacing=10)  # Create a vertical box layout
    push!(win, vbox)

    slide = GtkScale(false, 1:N_runs)
    Gtk.G_.value(slide, N_runs)
    push!(vbox, slide)

    canvas_1 = GtkCanvas()
    push!(vbox, canvas_1)

    n = Int(Gtk.GAccessor.value(slide))    

    data = [
    ("<u>Sim $(n) Metrics</u>", ""),
    (" ", " "),
    ("Ego Average Acceleration = ", "$(round(sim_sweep["Run $(n)"].avg_accel["ego average acceleration"], sigdigits=4))"),
    ("Other Average Acceleration = ", "$(round(sim_sweep["Run $(n)"].avg_accel["other average acceleration"], sigdigits=4))"),
    ("", ""),
    ("Ego PI = ", "$(round(sim_sweep["Run $(n)"].PI["ego PI"], sigdigits=4))"),
    ("Other PI = ", "$(round(sim_sweep["Run $(n)"].PI["other PI"], sigdigits=4))"),
    ("", ""),
    ("Ego PE = ", "$(round(sim_sweep["Run $(n)"].PE["Ego Path Efficiency"], sigdigits=4))"),
    ("Other PE = ", "$(round(sim_sweep["Run $(n)"].PE["Other Path Efficiency"], sigdigits=4))"),
    ("", ""),
    ("Minimum Distance = ", "$(round(sim_sweep["Run $(n)"].min_dist["Min Distance"], sigdigits=4))"),
    ("", ""),
    ("Max Solve Time: ", "$(round(sim_sweep["Run $(n)"].time["Ego Max Solve Time"], sigdigits=4)) s"),
    ("Average Solve Time: ", "$(round(sim_sweep["Run $(n)"].time["Ego Average Solve Time"], sigdigits=4)) s"),
    ("Planning Deadline Overruns: ", "$(trunc(Int, sim_sweep["Run $(n)"].time["Ego Planning Deadline Overruns"]))")
    ]   

    global text_label = GtkLabel("", margin_top=0)
    GAccessor.markup(text_label, format_data(data))
    push!(vbox, text_label)



    set_gtk_property!(vbox, :expand, canvas_1, true)
    set_gtk_property!(text_label, :vexpand, false)
    set_gtk_property!(text_label, :margin_top, 0)
    set_gtk_property!(text_label, :margin_bottom, 100)
    # set_gtk_property!(text_label, :margin_left, 15)

    function update_text(slide)
        n = Int(Gtk.GAccessor.value(slide))

        data = [
            ("<u>Sim $(n) Metrics</u>", ""),
            (" ", " "),
            ("Ego Average Acceleration = ", "$(round(sim_sweep["Run $(n)"].avg_accel["ego average acceleration"], sigdigits=4))"),
            ("Other Average Acceleration = ", "$(round(sim_sweep["Run $(n)"].avg_accel["other average acceleration"], sigdigits=4))"),
            ("", ""),
            ("Ego PI = ", "$(round(sim_sweep["Run $(n)"].PI["ego PI"], sigdigits=4))"),
            ("Other PI = ", "$(round(sim_sweep["Run $(n)"].PI["other PI"], sigdigits=4))"),
            ("", ""),
            ("Ego PE = ", "$(round(sim_sweep["Run $(n)"].PE["Ego Path Efficiency"], sigdigits=4))"),
            ("Other PE = ", "$(round(sim_sweep["Run $(n)"].PE["Other Path Efficiency"], sigdigits=4))"),
            ("", ""),
            ("Minimum Distance = ", "$(round(sim_sweep["Run $(n)"].min_dist["Min Distance"], sigdigits=4))"),
            ("", ""),
            ("Max Solve Time: ", "$(round(sim_sweep["Run $(n)"].time["Ego Max Solve Time"], sigdigits=4)) s"),
            ("Average Solve Time: ", "$(round(sim_sweep["Run $(n)"].time["Ego Average Solve Time"], sigdigits=4)) s"),
            ("Planning Deadline Overruns: ", "$(trunc(Int, sim_sweep["Run $(n)"].time["Ego Planning Deadline Overruns"]))")
            ]   

        GAccessor.markup(text_label, format_data(data))
    end

    @guarded draw(canvas_1) do widget
        ctx = getgc(canvas_1)
        n = Int(Gtk.GAccessor.value(slide))
        combined_plot(n)
        plot_img = read_from_png(io_combined)
        set_source_surface(ctx, plot_img, 0, 0)
        paint(ctx)
    end

    id_2 = signal_connect(update_text, slide, "value-changed")
    id_1 = signal_connect((w) -> draw(canvas_1), slide, "value-changed")
    showall(win)
    show(canvas_1)
    show(text_label)
end
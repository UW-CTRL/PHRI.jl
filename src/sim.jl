include("planner.jl")

function simulate(ego_ip::InteractionPlanner, other_ip::InteractionPlanner, sim_horizon::Int64; ibr_iterations=3::Int64, leader="ego"::String)
    # Given the IP problem setup of the ego agent and other agent
    # initialize matrices for saving the paths

    ego_dyn = ego_ip.ego_planner.incon.hps.dynamics
    other_dyn = other_ip.ego_planner.incon.hps.dynamics

    ego_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    ego_controls = Vector{Vector{Float64}}(undef, sim_horizon)
    other_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    other_controls = Vector{Vector{Float64}}(undef, sim_horizon)

    ego_traj[1] = ego_ip.ego_planner.incon.opt_params.initial_state
    other_traj[1] = other_ip.ego_planner.incon.opt_params.initial_state

    ego_solve_times = Vector{Float64}(undef, sim_horizon)
    other_solve_times = Vector{Float64}(undef, sim_horizon)

    # Uses MPC function to simulate to a given time horizon
    for i in 1:(sim_horizon)

        ego_state = ego_traj[i]
        other_state = other_traj[i]
        # solve for the next iteration

        ego_solve_start = time()
        ego_control = mpc_step(ego_ip, ego_state, other_state, ibr_iterations=ibr_iterations, leader=leader)
        ego_solve_end = time()
        ego_solve_times[i] = ego_solve_end - ego_solve_start

        other_solve_start = time()
        other_control = mpc_step(other_ip, other_state, ego_state, ibr_iterations=ibr_iterations, leader=leader)
        other_solve_end = time()
        other_solve_times[i] = other_solve_end - other_solve_start

        ego_state = step(ego_dyn, ego_state, ego_control)
        other_state = step(other_dyn, other_state, other_control)

        ego_traj[i+1] = ego_state
        other_traj[i+1] = other_state
        ego_controls[i] = ego_control
        other_controls[i] = other_control

    end

    # cast vector of vectors to matrix for easier plotting
    ego_traj = vector_of_vectors_to_matrix(ego_traj)
    ego_controls = vector_of_vectors_to_matrix(ego_controls)
    other_traj = vector_of_vectors_to_matrix(other_traj)
    other_controls = vector_of_vectors_to_matrix(other_controls)

    ego_traj, ego_controls, other_traj, other_controls, (ego_solve_times, other_solve_times)
end

function simulate(ego_ip::InteractionPlanner, other_ip::InteractionPlanner, sim_horizon::Int64, constant_velo_agents::ConstantVeloAgent...; ibr_iterations=3::Int64, leader="ego"::String)
    # Given the IP problem setup of the ego agent and other agent
    # initialize matrices for saving the paths

    ego_dyn = ego_ip.ego_planner.incon.hps.dynamics
    other_dyn = other_ip.ego_planner.incon.hps.dynamics

    velo_agents = collect(constant_velo_agents)

    dt = ego_ip.ego_planner.incon.hps.dynamics.dt

    ego_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    ego_controls = Vector{Vector{Float64}}(undef, sim_horizon)
    other_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    other_controls = Vector{Vector{Float64}}(undef, sim_horizon)

    ego_traj[1] = ego_ip.ego_planner.incon.opt_params.initial_state
    other_traj[1] = other_ip.ego_planner.incon.opt_params.initial_state

    N_velo_agents = length(constant_velo_agents)
    copied_constant_velo_agents = deepcopy(constant_velo_agents)

    # Uses MPC function to simulate to a given time horizon
    for i in 1:(sim_horizon)

        ego_state = ego_traj[i]
        other_state = other_traj[i]
        # solve for the next iteration

        ego_control = mpc_step(ego_ip, ego_state, other_state, velo_agents, ibr_iterations=ibr_iterations, leader=leader)
        other_control = mpc_step(other_ip, other_state, ego_state, velo_agents, ibr_iterations=ibr_iterations, leader=leader)

        ego_state = step(ego_dyn, ego_state, ego_control)
        other_state = step(other_dyn, other_state, other_control)
        for j in 1:N_velo_agents
            copied_constant_velo_agents[j].pos .+= copied_constant_velo_agents[j].velo * dt
        end

        ego_traj[i+1] = ego_state
        other_traj[i+1] = other_state
        ego_controls[i] = ego_control
        other_controls[i] = other_control

    end

    # cast vector of vectors to matrix for easier plotting
    ego_traj = vector_of_vectors_to_matrix(ego_traj)
    ego_controls = vector_of_vectors_to_matrix(ego_controls)
    other_traj = vector_of_vectors_to_matrix(other_traj)
    other_controls = vector_of_vectors_to_matrix(other_controls)

    ego_traj, ego_controls, other_traj, other_controls
end


function simulate_human_social_forces(ego_ip, other_dyn::DoubleIntegrator2D, other_initial_state, other_goal_state, sim_horizon::Int64; ibr_iterations=3::Int64, leader="ego"::String, p=2., q=2., τ=2., ψ=pi/6, c=0.3)
    # for Ego IP, the other_planner must be unicycle
    print(typeof(ego_ip.other_planner.incon.hps.dynamics))
    if typeof(ego_ip.other_planner.incon.hps.dynamics) != Unicycle{Float64}
        throw(TypeError(simulate_human_social_forces, "Incorrect dynamics type for 'other_planner' in 'ego_ip", Unicycle{Float64}, typeof(ego_ip.other_planner.incon.hps.dynamics)))
    end

    ego_dyn = ego_ip.ego_planner.incon.hps.dynamics

    ego_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    ego_controls = Vector{Vector{Float64}}(undef, sim_horizon)
    other_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    other_controls = Vector{Vector{Float64}}(undef, sim_horizon)

    ego_traj[1] = ego_ip.ego_planner.incon.opt_params.previous_states[1]
    other_traj[1] = other_initial_state

     # Uses MPC function to simulate to a given time horizon
    for i in 1:(sim_horizon)
        ego_state = ego_traj[i]
        other_state = other_traj[i]
        # solve for the next iteration
        other_heading = atan(other_state[4], other_state[3])
        other_state_ = [other_state[1]; other_state[2]; other_heading]

        ego_control = mpc_step(ego_ip, ego_state, other_state_, ibr_iterations=ibr_iterations, leader=leader)
        other_control = social_forces(other_dyn, other_state, other_goal_state[1:2], [[ego_dyn, ego_state]], other_dyn.velocity_max, p=p, q=q, τ=τ, ψ=ψ, c=c)
        # other_control = [0.;0.]
        
        # break
        ego_state = step(ego_dyn, ego_state, ego_control)
        other_state = step(other_dyn, other_state, other_control)

        ego_traj[i+1] = ego_state
        other_traj[i+1] = other_state

        # println(ego_control)
        ego_controls[i] = ego_control
        other_controls[i] = other_control

    end

    ego_traj = vector_of_vectors_to_matrix(ego_traj)
    ego_controls = vector_of_vectors_to_matrix(ego_controls)
    other_traj = vector_of_vectors_to_matrix(other_traj)
    other_controls = vector_of_vectors_to_matrix(other_controls)

    ego_traj, ego_controls, other_traj, other_controls
end
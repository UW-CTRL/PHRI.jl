include("planner.jl")
using Random

function simulate(ego_ip::InteractionPlanner, other_ip::InteractionPlanner, sim_horizon::Int64; ibr_iterations=3::Int64, leader="ego"::String, seed=010100000111001001101111011000010110001101110100011010010111011001100101010010000101001001001001)
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
        Random.seed!(seed + i)

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

        other_noisy_control = other_control .* (1 .+ randn(2) * 0.05)

        ego_state = step(ego_dyn, ego_state, ego_control)
        other_state = step(other_dyn, other_state, other_noisy_control)

        ego_traj[i+1] = ego_state
        other_traj[i+1] = other_state
        ego_controls[i] = ego_control
        other_controls[i] = other_noisy_control

    end

    # cast vector of vectors to matrix for easier plotting
    ego_traj = vector_of_vectors_to_matrix(ego_traj)
    ego_controls = vector_of_vectors_to_matrix(ego_controls)
    other_traj = vector_of_vectors_to_matrix(other_traj)
    other_controls = vector_of_vectors_to_matrix(other_controls)

    ego_traj, ego_controls, other_traj, other_controls, (ego_solve_times, other_solve_times)
end

function simulate(ego_ip::InteractionPlanner, other_ip::InteractionPlanner, sim_horizon::Int64, constant_velo_agents::ConstantVeloAgent...; ibr_iterations=3::Int64, leader="ego"::String, seed=010100000111001001101111011000010110001101110100011010010111011001100101010010000101001001001001)
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

    ego_solve_times = Vector{Float64}(undef, sim_horizon)
    other_solve_times = Vector{Float64}(undef, sim_horizon)

    N_velo_agents = length(constant_velo_agents)
    copied_constant_velo_agents = deepcopy(constant_velo_agents)

    # Uses MPC function to simulate to a given time horizon
    for i in 1:(sim_horizon)
        Random.seed!(seed + i)

        ego_state = ego_traj[i]
        other_state = other_traj[i]
        # solve for the next iteration

        ego_solve_start = time()
        ego_control = mpc_step(ego_ip, ego_state, other_state, velo_agents, ibr_iterations=ibr_iterations, leader=leader)
        ego_solve_end = time()
        ego_solve_times[i] = ego_solve_end - ego_solve_start

        other_solve_start = time()
        other_control = mpc_step(other_ip, other_state, ego_state, velo_agents, ibr_iterations=ibr_iterations, leader=leader)
        other_solve_end = time()
        other_solve_times[i] = other_solve_end - other_solve_start

        other_noisy_control = other_control .* (1 .+ randn(2) * 0.05)

        ego_state = step(ego_dyn, ego_state, ego_control)
        other_state = step(other_dyn, other_state, other_noisy_control)
        for j in 1:N_velo_agents
            copied_constant_velo_agents[j].pos .+= copied_constant_velo_agents[j].velo * dt
        end

        ego_traj[i+1] = ego_state
        other_traj[i+1] = other_state
        ego_controls[i] = ego_control
        other_controls[i] = other_noisy_control

    end

    # cast vector of vectors to matrix for easier plotting
    ego_traj = vector_of_vectors_to_matrix(ego_traj)
    ego_controls = vector_of_vectors_to_matrix(ego_controls)
    other_traj = vector_of_vectors_to_matrix(other_traj)
    other_controls = vector_of_vectors_to_matrix(other_controls)

    ego_traj, ego_controls, other_traj, other_controls, (ego_solve_times, other_solve_times)
end


function simulate_human_social_forces(ego_dyn::DoubleIntegrator2D, other_ip, ego_initial_state, ego_goal_state, sim_horizon::Int64; ibr_iterations=3::Int64, leader="ego"::String, p=2., q=2., τ=2., ψ=pi/6, c=0.3, seed=010100000111001001101111011000010110001101110100011010010111011001100101010010000101001001001001)
    # for Ego IP, the other_planner must be unicycle
    print(typeof(other_ip.other_planner.incon.hps.dynamics))
    if typeof(other_ip.other_planner.incon.hps.dynamics) != Unicycle{Float64}
        throw(TypeError(simulate_human_social_forces, "Incorrect dynamics type for 'other_planner' in 'other_ip", Unicycle{Float64}, typeof(other_ip.other_planner.incon.hps.dynamics)))
    end

    other_dyn = other_ip.ego_planner.incon.hps.dynamics

    other_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    other_controls = Vector{Vector{Float64}}(undef, sim_horizon)
    ego_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    ego_controls = Vector{Vector{Float64}}(undef, sim_horizon)

    other_traj[1] = other_ip.ego_planner.incon.opt_params.previous_states[1]
    ego_traj[1] = ego_initial_state

     # Uses MPC function to simulate to a given time horizon
    for i in 1:(sim_horizon)
        Random.seed!(seed + i)

        ego_state = ego_traj[i]
        other_state = other_traj[i]
        # solve for the next iteration
        ego_heading = atan(ego_state[4], ego_state[3])
        ego_state_ = [ego_state[1]; ego_state[2]; ego_heading]

        other_control = mpc_step(other_ip, other_state, ego_state_, ibr_iterations=ibr_iterations, leader=leader)
        ego_control = social_forces(ego_dyn, ego_state, ego_goal_state[1:2], [[other_dyn, other_state]], ego_dyn.velocity_max, p=p, q=q, τ=τ, ψ=ψ, c=c)
        # other_control = [0.;0.]

        other_noisy_control = other_control .* (1 .+ randn(2) * 0.05)
        
        # break
        ego_state = step(ego_dyn, ego_state, ego_control)
        other_state = step(other_dyn, other_state, other_noisy_control)

        ego_traj[i+1] = ego_state
        other_traj[i+1] = other_state

        # println(ego_control)
        ego_controls[i] = ego_control
        other_controls[i] = other_noisy_control

    end

    ego_traj = vector_of_vectors_to_matrix(ego_traj)
    ego_controls = vector_of_vectors_to_matrix(ego_controls)
    other_traj = vector_of_vectors_to_matrix(other_traj)
    other_controls = vector_of_vectors_to_matrix(other_controls)

    ego_traj, ego_controls, other_traj, other_controls
end

function simulate_human_social_forces(ego_dyn::DynamicallyExtendedUnicycle, other_ip::InteractionPlanner, ego_initial_state, ego_goal_state, sim_horizon::Int64; ibr_iterations=3::Int64, leader="ego"::String, p=2., q=2., τ=2., ψ=pi/6, c=0.3, seed=010100000111001001101111011000010110001101110100011010010111011001100101010010000101001001001001)
    # for Ego IP, the other_planner must be unicycle
    print(typeof(other_ip.other_planner.incon.hps.dynamics))
    if typeof(other_ip.other_planner.incon.hps.dynamics) != DynamicallyExtendedUnicycle{Float64}
        throw(TypeError(simulate_human_social_forces, "Incorrect dynamics type for 'other_planner' in 'other_ip", DynamicallyExtendedUnicycle{Float64}, typeof(other_ip.other_planner.incon.hps.dynamics)))
    end

    other_dyn = other_ip.ego_planner.incon.hps.dynamics

    other_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    other_controls = Vector{Vector{Float64}}(undef, sim_horizon)
    ego_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    ego_controls = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    ego_controls[1] = [0., 0.]

    other_traj[1] = other_ip.ego_planner.incon.opt_params.previous_states[1]
    ego_traj[1] = ego_initial_state

     # Uses MPC function to simulate to a given time horizon
    for i in 1:(sim_horizon)
        Random.seed!(seed + i)

        ego_state = ego_traj[i]
        other_state = other_traj[i]
        # solve for the next iteration

        other_control = mpc_step(other_ip, other_state, ego_state, ibr_iterations=ibr_iterations, leader=leader)
        ego_forces = social_forces(ego_dyn, ego_state, ego_goal_state[1:2], [[other_dyn, other_state]], ego_dyn.velocity_max, p=p, q=q, τ=τ, ψ=ψ, c=c)
        ego_control = accel_to_dynamically_extended_unicycle(ego_forces[1:2], ego_state[3], get_velocity(ego_dyn, ego_state, ego_controls[i])[1:2])

        other_noisy_control = other_control .* (1 .+ randn(2) * 0.05)

        # break
        ego_state = step(ego_dyn, ego_state, ego_control)
        other_state = step(other_dyn, other_state, other_noisy_control)


        ego_traj[i+1] = ego_state
        other_traj[i+1] = other_state

        # println(ego_control)
        ego_controls[i+1] = ego_control
        other_controls[i] = other_noisy_control

    end

    ego_traj = vector_of_vectors_to_matrix(ego_traj)
    ego_controls = vector_of_vectors_to_matrix(ego_controls[2:end])
    other_traj = vector_of_vectors_to_matrix(other_traj)
    other_controls = vector_of_vectors_to_matrix(other_controls)

    ego_traj, ego_controls, other_traj, other_controls
end

function simulate_hj(ego_hps::PlannerHyperparameters, other_ip::InteractionPlanner, ego_initial_state::Vector{Float64}, ego_goal_state::Vector{Float64}, sim_horizon::Int64; ibr_iterations=3::Int64, leader="ego"::String, verbose=false::Bool, seed=010100000111001001101111011000010110001101110100011010010111011001100101010010000101001001001001)
    # Given the IP problem setup of the ego agent and other agent
    # initialize matrices for saving the paths

    # # HJIdata = matread("../hj_cache/DynamicallyExtendedUnicycle_VO_40_40_10_12_12.mat")
    # HJIdata = matread("../hj_cache/DynamicallyExtendedUnicycle_VO_50_50_10_20_20.mat")

    # V_mat = HJIdata["V"]
    # V_mat = [V_mat;;;V_mat[:,:,1:1,:,:]]
    # grid_knots = tuple((x -> convert(Vector{Float32}, vec(x))).(HJIdata["grid_knots"])...)
    # push!(grid_knots[3], -grid_knots[3][1])
    # global V = interpolate(Float32, Float32, grid_knots, V_mat, Gridded(Linear()));

    ego_dyn = ego_hps.dynamics
    other_dyn = other_ip.ego_planner.incon.hps.dynamics

    ego_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    ego_controls = Vector{Vector{Float64}}(undef, sim_horizon)
    other_traj = Vector{Vector{Float64}}(undef, sim_horizon + 1)
    other_controls = Vector{Vector{Float64}}(undef, sim_horizon)

    ego_traj[1] = ego_initial_state
    other_traj[1] = other_ip.ego_planner.incon.opt_params.initial_state

    ego_solve_times = Vector{Float64}(undef, sim_horizon)
    other_solve_times = Vector{Float64}(undef, sim_horizon)

    VO_QP = construct_VO_QP_base(ego_dyn, [0.; 0.])

    # Uses MPC function to simulate to a given time horizon
    for i in 1:(sim_horizon)
        Random.seed!(seed + i)

        ego_state = ego_traj[i]
        other_state = other_traj[i]
        # solve for the next iteration

        ego_opt_params_ = PlannerOptimizerParams(ego_dyn, ego_hps, ego_state, ego_goal_state, "ECOS")
        ego_ideal_planner_ = IdealProblem(ego_dyn, ego_hps, ego_opt_params_)
        solve(ego_ideal_planner_)
        ego_desired_control = ego_ideal_planner_.opt_params.previous_controls[1, :][1]

        other_position = get_position(other_dyn, other_state)
        if i == 1
            other_velocity = get_velocity(other_dyn, other_state, zeros(Float64, other_dyn.ctrl_dim))[:]
        else
            other_velocity = get_velocity(other_dyn, other_state, other_controls[i-1])[:]
        end
        
        rel_state = relative_state(ego_dyn, ego_state, other_position, other_velocity)

        ego_control, ϵ, ∇V, f0, B, V_min= minimally_invasive_velocity_obstacles(VO_QP, 
        ego_dyn,
        ego_state, 
        ego_desired_control,
        [other_position],
        [other_velocity]
)

        other_control = mpc_step(other_ip, other_state, ego_state, ibr_iterations=ibr_iterations, leader=leader)

        other_noisy_control = other_control .* (1 .+ randn(2) * 0.05)

        ego_state = step(ego_dyn, ego_state, ego_control)
        other_state = step(other_dyn, other_state, other_noisy_control)

        ego_traj[i+1] = ego_state
        other_traj[i+1] = other_state
        ego_controls[i] = ego_control
        other_controls[i] = other_noisy_control

    end

    # cast vector of vectors to matrix for easier plotting
    ego_traj = vector_of_vectors_to_matrix(ego_traj)
    ego_controls = vector_of_vectors_to_matrix(ego_controls)
    other_traj = vector_of_vectors_to_matrix(other_traj)
    other_controls = vector_of_vectors_to_matrix(other_controls)

    ego_traj, ego_controls, other_traj, other_controls
end

# optimal controller simulation w/o ibr
function simulate_oc(ego_ip::InteractionPlanner, other_ip::InteractionPlanner, sim_horizon::Int64; ego_ibr_iterations=3::Int64, other_ibr_iterations=3::Int64, leader="ego"::String, seed=010100000111001001101111011000010110001101110100011010010111011001100101010010000101001001001001)
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
        Random.seed!(seed + i)

        ego_state = ego_traj[i]
        other_state = other_traj[i]
        # solve for the next iteration

        ego_solve_start = time()
        ego_control = mpc_step(ego_ip, ego_state, other_state, ibr_iterations=ego_ibr_iterations, leader=leader)
        ego_solve_end = time()
        ego_solve_times[i] = ego_solve_end - ego_solve_start

        other_solve_start = time()
        other_control = mpc_step(other_ip, other_state, ego_state, ibr_iterations=other_ibr_iterations, leader=leader)
        other_solve_end = time()
        other_solve_times[i] = other_solve_end - other_solve_start

        other_noisy_control = other_control .* (1 .+ randn(2) * 0.05)

        ego_state = step(ego_dyn, ego_state, ego_control)
        other_state = step(other_dyn, other_state, other_noisy_control)

        ego_traj[i+1] = ego_state
        other_traj[i+1] = other_state
        ego_controls[i] = ego_control
        other_controls[i] = other_noisy_control

    end

    # cast vector of vectors to matrix for easier plotting
    ego_traj = vector_of_vectors_to_matrix(ego_traj)
    ego_controls = vector_of_vectors_to_matrix(ego_controls)
    other_traj = vector_of_vectors_to_matrix(other_traj)
    other_controls = vector_of_vectors_to_matrix(other_controls)

    ego_traj, ego_controls, other_traj, other_controls, (ego_solve_times, other_solve_times)
end
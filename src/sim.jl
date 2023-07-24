include("planner.jl")

function Sim(ego_ip::InteractionPlanner, other_ip::InteractionPlanner, sim_horizon::Int64; ibr_iterations=3::Int64, leader="ego"::String)
    # Given the IP problem setup of the ego agent and other agent
    # initialize matrices for saving the paths

    ego_path = Vector{Vector{Float64}}(undef, sim_horizon)
    ego_controls = Vector{Vector{Float64}}(undef, sim_horizon - 1)
    other_path = Vector{Vector{Float64}}(undef, sim_horizon)
    other_controls = Vector{Vector{Float64}}(undef, sim_horizon - 1)

    # Uses MPC function to simulate to a given time horizon
    for i in 1:(sim_horizon)
        # update state of each agent
        ego_path[i] = ego_ip.ego_planner.incon.opt_params.previous_states[1]
        if i != sim_horizon ego_controls[i] = ego_ip.ego_planner.incon.opt_params.previous_controls[1] end
        other_path[i] = other_ip.ego_planner.incon.opt_params.previous_states[1]
        if i != sim_horizon other_controls[i] = other_ip.ego_planner.incon.opt_params.previous_controls[1] end

        ego_state = step(ego_ip.ego_planner.incon.hps.dynamics, ego_ip.ego_planner.incon.opt_params.previous_states[1], ego_ip.ego_planner.incon.opt_params.previous_controls[1])
        other_state = step(other_ip.ego_planner.incon.hps.dynamics, other_ip.ego_planner.incon.opt_params.previous_states[1], other_ip.ego_planner.incon.opt_params.previous_controls[1])

        # solve for the next iteration
        mpc_step(ego_ip, ego_state, other_state, ibr_iterations=ibr_iterations, leader=leader)
        mpc_step(other_ip, other_state, ego_state, ibr_iterations=ibr_iterations, leader=leader)
    end

    # cast vector of vectors to matrix for easier plotting
    ego_path = vector_of_vectors_to_matrix(ego_path)
    ego_controls = vector_of_vectors_to_matrix(ego_controls)
    other_path = vector_of_vectors_to_matrix(other_path)
    other_controls = vector_of_vectors_to_matrix(other_controls)

    ego_path, ego_controls, other_path, other_controls
end
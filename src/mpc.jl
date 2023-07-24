function mpc_step(ip::InteractionPlanner, ego_actual_state::Vector{T}, other_actual_state::Vector{T}; ibr_iterations::Int64, leader::String) where {T}
    ip.ego_planner.ideal.opt_params.initial_state = ego_actual_state
    ip.other_planner.ideal.opt_params.initial_state = other_actual_state
    _, _, ego_control = IteratedBestResponseMPC(ip, ibr_iterations, leader)
    # MPC Step function that takes inputs of current ego state and current state of other agent
    # Returns optimal ego control 
    return ego_control[1]
end
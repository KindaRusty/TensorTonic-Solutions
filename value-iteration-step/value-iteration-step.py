def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states = len(values)
    num_actions = len(transitions[0])
    new_values = []

    for s in range(num_states):
        max_q = float('-inf')
        for a in range(num_actions):
            expected_future_value = 0
            for s_prime in range(num_states):
                expected_future_value += transitions[s][a][s_prime] * values[s_prime]
            q_s_a = rewards[s][a] + gamma * expected_future_value
            
            if q_s_a > max_q:
                max_q = q_s_a
        new_values.append(float(max_q))
        
    return new_values
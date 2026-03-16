import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    v_new = np.array(V, dtype=float)
    td_error = r + gamma * v_new[s_next] - v_new[s]
    
    v_new[s] = v_new[s] + alpha * td_error

    return v_new
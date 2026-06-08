import numpy as np
def mc_policy_evaluation(episodes, gamma, n_states):
    # Khởi tạo mảng giá trị V và danh sách chứa returns cho từng state
    V = np.zeros(n_states)
    returns = {s: [] for s in range(n_states)}
    # Duyệt qua từng episode
    for episode in episodes:
        G = 0
        # Sử dụng dict để ghi đè/lưu giá trị G của lần xuất hiện đầu tiên (tính từ đầu)
        # Vì duyệt ngược, giá trị được cập nhật cuối cùng của state s chính là First-Visit
        visited_in_episode = {}
        # Duyệt ngược từ cuối episode về đầu để tính G hiệu quả
        for state, reward in reversed(episode):
            G = reward + (gamma * G)
            visited_in_episode[state] = G  # Ghi đè liên tục, giữ lại giá trị ở "First-visit"
        # Sau khi kết thúc episode, nạp các giá trị First-visit vào returns tổng
        for state, g_val in visited_in_episode.items():
            returns[state].append(g_val)
    # Tính trung bình cộng return cho mỗi state
    for s in range(n_states):
        if returns[s]:
            V[s] = np.mean(returns[s])
    return V
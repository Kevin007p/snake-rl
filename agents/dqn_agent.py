class DQNAgent:
    def __init__(self, obs_shape, n_actions, **hp):
        # build policy and target networks
        # init optimizer, replay buffer
        pass

    def select_action(self, state, epsilon):
        # epsilon-greedy
        pass

    def store_transition(self, s,a,r,s_,done):
        pass

    def update(self):
        # sample batch, compute loss, backward()
        pass

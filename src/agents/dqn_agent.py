class DQNAgent:
    def __init__(self, config):
        self.config = config

    def act(self, state):
        raise NotImplementedError

    def learn(self, transition):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

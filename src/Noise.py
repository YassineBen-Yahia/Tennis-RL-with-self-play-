import numpy as np

class OUNoise:
    
    """
    Ornstein-Uhlenbeck process to generate noise, which is temporally correlated.
    """
    
    def __init__(self, outpu_size, seed=1, mu=0., theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process.
        Params
        ======
            size (int): dimension of the noise, which should be the dimension of the action
            seed (int): random seed
            mu (float): the long-running mean
            theta (float): the speed of mean reversion
            sigma (float): the volatility parameter
        """
        self.mu = mu * np.ones(outpu_size)
        self.theta = theta
        self.sigma = sigma
        np.random.seed(seed)
        self.reset()
    
    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.state = np.copy(self.mu)
    
    def sample(self):
        """
        Update internal state and return it as a noise sample.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
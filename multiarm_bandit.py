2# Barakovic Ksenija 
# RA27/2020

from random import Random
from typing import Iterable
from tqdm import trange

import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

random = Random()

class Bandit:
    '''Represents an instance of a bandit 
    
    Attributes: 
        - mean : average reward returned by the bandit
        - span : deviation of the reward returned by the badnit'''

    def __init__(self, mean : float, span : float) -> None:
        self.mean = mean
        self.span = span
    
    def pull(self) -> float:
        '''Simulates a lever pull.
        Returns a reward'''
        reward = self.mean + 2*self.span * (random.random() - 0.5)
        return reward

class Environment():
    '''Contains all bandits and configurations
    
    Attributes: 
        - bandits        : list of available bandits
        - penalty        : punishment for choosing a non-existant action
        - change_iter    : how often the environment changes, every change_iter iterations'''

    def __init__(self, bandits: Iterable[Bandit], change_iter:int, penalty=1000) -> None:
        self.bandits = list(bandits)
        self.num_of_bandits = len(self.bandits)
        self.penalty = penalty
        self.change_iter = change_iter

    def take_action(self, bandit) -> float:
        '''Performs an action for the specified bandit.
        Returns a reward.'''
        if bandit < 0 or bandit >= self.num_of_bandits:
            return -self.penalty
        else:
            return self.bandits[bandit].pull()

    def change_bandits(self):
        '''Changes mean and span values for all bandits in a 
        predefined range of random values.'''
        for bandit in self.bandits:
            bandit.mean = random.randint(1,50)*(random.random()-0.5)
            bandit.span = random.randint(1,10)*random.random()

def greedy_action(q:list[float]) -> int:
    '''Returns the bandit with the best reward aproximation'''
    return np.argmax(q)

def random_action(num_of_bandits:int) -> int:
    '''Returns a random bandit'''
    return random.randint(0, num_of_bandits-1)

def eps_greedy_action(q:list[float], eps:float) -> int:
    '''Returns a bandit based on epsilon-greedy policy'''
    if random.random() > eps:
        return greedy_action(q)
    else:
        return random_action(len(q))

def loop(env:Environment, alpha:float, q, eps:float, iter_num:int=5000, test:bool=False) -> list[tuple]:
    rewards = []
    q_values = []

    for i in trange(1, iter_num):
        bandit = eps_greedy_action(q, eps)
        r = env.take_action(bandit)

        if not test:
            q[bandit] += alpha * (r-q[bandit])
        # (mean aproximation for bandit, bandit number, real mean for bandit)
        q_values.append((q[bandit], bandit, env.bandits[bandit].mean))

        if i % (env.change_iter+2) == 0:
            env.change_bandits()

        rewards.append(r)

    plt.subplot(1, 2, 1)

    plt.scatter(range(len(q)), q, marker=".", label="estimated mean")
    plt.scatter(range(len(q)), [b.mean for b in env.bandits], marker="x", label="actual mean")
    plt.legend()

    plt.subplot(1, 2, 2)

    g = np.cumsum(rewards)
    max_reward = max([b.mean for b in env.bandits])
    plt.title(f"eps={eps}")
    plt.plot(g, color="r", label="estimated gain")
    plt.plot(np.cumsum(max_reward * np.ones(len(g))), label="ideal gain", color="b")

    plt.legend()
    plt.show()

    return q_values

def first(env:Environment, eps:float, alpha:float, iter_num:int):
    """ 1. Experiment with the value of epsilon. 
    Try smaller values, and verify that the slope difference is smaller. 
    Verify also the oposite fact: increasing epsilon decreases the slope 
    of the plotted achieved gain. Explain."""
  
    for _ in trange(1, 6):
        q = [0 for _ in range(env.num_of_bandits)]
        loop(env, alpha, q, eps, iter_num)
        eps /= 2

    # Smaller eps -> More frequent choice of greedy bandit
    # Since no environment changes occured, greedy policy is the best policy

    # Larger eps -> More frequent choice of random bandit
    # Random choice of bandits leads to exploration and better learning,
    # but smaller overall rewards

def second(env:Environment, eps:float, alpha:float, iter_num:int):
    """2. Perform a training run to learn the q vector (in essence, 
    to estimate the mean value of the reward returned by each bandit). 
    Create a test loop, in which the learned q values remain fixed. 
    Compare performance of the agent to the ideal performance 
    (as it is done in the previous point).
    """

    q = [0 for _ in range(env.num_of_bandits)]
    loop(env, alpha, q, eps=0.9, iter_num=iter_num)
    # Large eps leads to better learning because bandits are mostly chosen by random
    # It results with a very good aproximation of mean rewards for each bandit, 
    # but a poor gain aproximation 

    loop(env, alpha, q, eps=0.1, test=True, iter_num=iter_num)
    # In order to test how good the learned aproximations are we choose a very small eps 
    # so to always choose the greedy bandit
    # It results with a very good overall gain aproximation

def third(env:Environment, eps:float, alpha:float, iter_num:int):
    """3. Plot the change of the estimated q values in time. 
    Show that with passing time the algorithm sucessfully 
    approximates mean rewards of all bandits."""

    q = [0 for _ in range(env.num_of_bandits)]
    q_values = loop(env, alpha, q, eps, iter_num)

    qs = list(map(lambda x: x[0], q_values))
    bs = list(map(lambda x: x[1], q_values))

    for index in range(env.num_of_bandits):
        # Index of q-values for the current num bandit
        indexes = [i for i, bandit in enumerate(bs) if bandit == index]

        # Q-values for that bandit
        values = [val for i, val in enumerate(qs) if i in indexes]

        plt.title(f"Bandit: {index}, Mean: {env.bandits[index].mean}, Final aprox.: {values[-1]}")
        plt.scatter(range(len(values)), values, marker=".")
        plt.plot(env.bandits[index].mean * np.ones(len(values)), linestyle="--", color="r")
        plt.show()

def fourth(env:Environment, eps:float, alpha:float, iter_num:int):
    """4. Modify the environment so that the mean values of 
    all bandits change in time. Repeat the training procedure.
    Evaluate its effectiveness.
    """

    q = [0 for _ in range(env.num_of_bandits)]
    q_values = loop(env, alpha, q, eps, iter_num)

    qs = list(map(lambda x: x[0], q_values))
    bs = list(map(lambda x: x[1], q_values))
    ms = list(map(lambda x: x[2], q_values))

    colors = [np.random.rand(3,) for _ in range(env.num_of_bandits)]

    # When the environment is changed too often there is no time to learn anything useful about the bandits, resulting in very chaotic and random choices
    # When eps is large, random policy, mean values are still aproximated pretty well because the space is explored after every change
    # When eps is small, and the policy greedy, we can no longer determine the best bandit
    for index in range(env.num_of_bandits):
        # Index of q-values for the current bandit
        indexes = [i for i, bandit in enumerate(bs) if bandit == index]
        
        # Approximated mean values for that bandit
        values = [val for i, val in enumerate(qs) if i in indexes]

        # Actual mean values for that bandit
        m_values = [val for i, val in enumerate(ms) if i in indexes]

        # Showcases how often each bandit was chosen during training
        print(f"Bandit {index} was chosen {len(indexes)} times")
        plt.scatter(indexes, values, marker="+", c=colors[index], label=f"bandit {index}")
        plt.scatter(indexes, m_values, marker="_", c=colors[index])

    plt.legend()
    plt.show()

    # Best results are achieved when eps is in the middle
    # When the environment experiences changes there is no point in stopping with the exploration faze, 
    # since everything that has been learned expires after some time
    
assignment_dict = {
    #   num_od_bandits, change_iter, eps, alpha, iteration_number,  function
    1 : (5,                5000,        1,  0.1,    5000,             first),
    2 : (5,                5000,      0.9,  0.1,    5000,            second),
    3 : (5,                5000,      0.9,  0.1,    5000,             third),
    4 : (5,                 500,      0.5,  0.1,    5000,            fourth)
}

if __name__ == "__main__":
    while 1:
        assignment = int(input("Izaberite zadatak za prikaz: "))
        number_of_bandits, change_iter, eps, alpha, iteration_num, assignment = assignment_dict[assignment]

        bandits = [Bandit(10*(random.random()-0.5), 5*random.random()) for _ in range(number_of_bandits)]
        env = Environment(bandits, change_iter)

        assignment(env, eps, alpha, iteration_num)

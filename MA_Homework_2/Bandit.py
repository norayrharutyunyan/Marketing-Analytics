"""
A/B Testing with Multi-Armed Bandits
=====================================
Implements Epsilon-Greedy and Thompson Sampling algorithms
for a 4-armed bandit problem with known reward probabilities.

:author: Student
:date: 2026
"""

############################### IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from loguru import logger


############################### CONSTANTS
BANDIT_REWARDS = [1, 2, 3, 4]
NUM_TRIALS = 20000
PRECISION = 1  # Known precision for Thompson Sampling (tau)


############################### BASE CLASS

class Bandit(ABC):
    """
    Abstract base class for multi-armed bandit algorithms.

    :param p: True reward (mean) for this bandit arm.
    :type p: float
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize bandit with true reward value.

        :param p: True expected reward.
        :type p: float
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return string representation of the bandit."""
        pass

    @abstractmethod
    def pull(self):
        """
        Simulate pulling the bandit arm.

        :return: Observed reward.
        :rtype: float
        """
        pass

    @abstractmethod
    def update(self, x):
        """
        Update internal estimates based on observed reward.

        :param x: Observed reward from a pull.
        :type x: float
        """
        pass

    @abstractmethod
    def experiment(self):
        """Run the full bandit experiment."""
        pass

    @abstractmethod
    def report(self):
        """
        Report results: store CSV, print cumulative reward and regret.
        """
        pass


############################### VISUALIZATION

class Visualization:
    """
    Visualization utilities for bandit experiment results.
    """

    def plot1(self, eg_rewards, ts_rewards, eg_best, ts_best):
        """
        Visualize the learning process for each algorithm on linear and log scale.

        :param eg_rewards: Cumulative rewards per trial for Epsilon-Greedy.
        :param ts_rewards: Cumulative rewards per trial for Thompson Sampling.
        :param eg_best: Optimal cumulative rewards for Epsilon-Greedy baseline.
        :param ts_best: Optimal cumulative rewards for Thompson Sampling baseline.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Learning Process per Algorithm", fontsize=16)

        trials = np.arange(1, len(eg_rewards) + 1)

        # E-Greedy linear
        axes[0, 0].plot(trials, eg_rewards, label="E-Greedy Cumulative Reward", color="blue")
        axes[0, 0].plot(trials, eg_best, label="Optimal", color="green", linestyle="--")
        axes[0, 0].set_title("Epsilon-Greedy (Linear Scale)")
        axes[0, 0].set_xlabel("Trial")
        axes[0, 0].set_ylabel("Cumulative Reward")
        axes[0, 0].legend()

        # E-Greedy log
        axes[0, 1].plot(trials, eg_rewards, label="E-Greedy Cumulative Reward", color="blue")
        axes[0, 1].plot(trials, eg_best, label="Optimal", color="green", linestyle="--")
        axes[0, 1].set_title("Epsilon-Greedy (Log Scale)")
        axes[0, 1].set_xlabel("Trial")
        axes[0, 1].set_ylabel("Cumulative Reward")
        axes[0, 1].set_xscale("log")
        axes[0, 1].legend()

        # Thompson linear
        axes[1, 0].plot(trials, ts_rewards, label="Thompson Cumulative Reward", color="orange")
        axes[1, 0].plot(trials, ts_best, label="Optimal", color="green", linestyle="--")
        axes[1, 0].set_title("Thompson Sampling (Linear Scale)")
        axes[1, 0].set_xlabel("Trial")
        axes[1, 0].set_ylabel("Cumulative Reward")
        axes[1, 0].legend()

        # Thompson log
        axes[1, 1].plot(trials, ts_rewards, label="Thompson Cumulative Reward", color="orange")
        axes[1, 1].plot(trials, ts_best, label="Optimal", color="green", linestyle="--")
        axes[1, 1].set_title("Thompson Sampling (Log Scale)")
        axes[1, 1].set_xlabel("Trial")
        axes[1, 1].set_ylabel("Cumulative Reward")
        axes[1, 1].set_xscale("log")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig("plot1_learning_process.png")
        plt.show()
        logger.info("plot1 saved as 'plot1_learning_process.png'")

    def plot2(self, eg_cum_rewards, ts_cum_rewards, eg_cum_regrets, ts_cum_regrets):
        """
        Compare E-Greedy vs Thompson Sampling on cumulative rewards and regrets.

        :param eg_cum_rewards: Cumulative rewards list for E-Greedy.
        :param ts_cum_rewards: Cumulative rewards list for Thompson Sampling.
        :param eg_cum_regrets: Cumulative regrets list for E-Greedy.
        :param ts_cum_regrets: Cumulative regrets list for Thompson Sampling.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("E-Greedy vs Thompson Sampling Comparison", fontsize=16)

        trials = np.arange(1, len(eg_cum_rewards) + 1)

        # Cumulative Rewards
        axes[0].plot(trials, eg_cum_rewards, label="Epsilon-Greedy", color="blue")
        axes[0].plot(trials, ts_cum_rewards, label="Thompson Sampling", color="orange")
        axes[0].set_title("Cumulative Rewards")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Cumulative Reward")
        axes[0].legend()

        # Cumulative Regrets
        axes[1].plot(trials, eg_cum_regrets, label="Epsilon-Greedy", color="blue")
        axes[1].plot(trials, ts_cum_regrets, label="Thompson Sampling", color="orange")
        axes[1].set_title("Cumulative Regrets")
        axes[1].set_xlabel("Trial")
        axes[1].set_ylabel("Cumulative Regret")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("plot2_comparison.png")
        plt.show()
        logger.info("plot2 saved as 'plot2_comparison.png'")


############################### EPSILON-GREEDY

class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit algorithm with decaying epsilon (1/t).

    :param p: True expected reward for this bandit arm.
    :type p: float
    """

    def __init__(self, p):
        """
        Initialize Epsilon-Greedy bandit.

        :param p: True expected reward.
        :type p: float
        """
        self.p = p               # true reward
        self.p_estimate = 0      # estimated reward
        self.N = 0               # number of pulls

    def __repr__(self):
        """
        Return string representation.

        :return: Description string.
        :rtype: str
        """
        return f"EpsilonGreedy(true_reward={self.p}, estimated_reward={self.p_estimate:.4f}, pulls={self.N})"

    def pull(self):
        """
        Pull the arm: sample from Gaussian with mean = true reward.

        :return: Sampled reward.
        :rtype: float
        """
        return np.random.randn() + self.p

    def update(self, x):
        """
        Update running estimate of the mean reward.

        :param x: Observed reward.
        :type x: float
        """
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N

    def experiment(self, bandits, num_trials):
        """
        Run Epsilon-Greedy experiment across all bandits.

        :param bandits: List of EpsilonGreedy bandit arms.
        :type bandits: list[EpsilonGreedy]
        :param num_trials: Total number of trials to run.
        :type num_trials: int
        :return: Tuple of (rewards list, chosen bandits list).
        :rtype: tuple[list, list]
        """
        rewards = []
        chosen_bandits = []

        for t in range(1, num_trials + 1):
            epsilon = 1 / t  # decaying epsilon

            if np.random.random() < epsilon:
                # Explore: pick random bandit
                chosen = np.random.randint(len(bandits))
            else:
                # Exploit: pick bandit with highest estimate
                chosen = np.argmax([b.p_estimate for b in bandits])

            reward = bandits[chosen].pull()
            bandits[chosen].update(reward)

            rewards.append(reward)
            chosen_bandits.append(chosen)

        return rewards, chosen_bandits

    def report(self, rewards, chosen_bandits, bandits, algorithm_name="EpsilonGreedy"):
        """
        Report results: save CSV, log cumulative reward and regret.

        :param rewards: List of rewards obtained per trial.
        :param chosen_bandits: List of bandit indices chosen per trial.
        :param bandits: List of bandit arm objects.
        :param algorithm_name: Label for the algorithm.
        :type algorithm_name: str
        """
        best_reward = max(b.p for b in bandits)
        cumulative_reward = sum(rewards)
        cumulative_regret = best_reward * len(rewards) - cumulative_reward

        # Save to CSV
        df = pd.DataFrame({
            "Bandit": [b + 1 for b in chosen_bandits],
            "Reward": rewards,
            "Algorithm": algorithm_name
        })
        filename = f"{algorithm_name}_results.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to '{filename}'")

        logger.info(f"[{algorithm_name}] Cumulative Reward: {cumulative_reward:.4f}")
        logger.info(f"[{algorithm_name}] Cumulative Regret: {cumulative_regret:.4f}")

        return cumulative_reward, cumulative_regret


############################### THOMPSON SAMPLING

class ThompsonSampling(Bandit):
    """
    Thompson Sampling bandit using Gaussian posterior with known precision.

    Assumes reward ~ N(mu, 1/tau) where tau (precision) is known.
    Prior: mu ~ N(m, 1/lambda).

    :param p: True expected reward for this bandit arm.
    :type p: float
    :param precision: Known precision (tau) of the reward distribution.
    :type precision: float
    """

    def __init__(self, p, precision=PRECISION):
        """
        Initialize Thompson Sampling bandit.

        :param p: True expected reward.
        :type p: float
        :param precision: Known precision of reward distribution.
        :type precision: float
        """
        self.p = p                   # true reward
        self.precision = precision   # known tau (precision of likelihood)
        self.m = 0                   # posterior mean
        self.lambda_ = 1             # posterior precision (starts at prior precision)
        self.N = 0                   # number of pulls
        self.sum_x = 0               # sum of observed rewards

    def __repr__(self):
        """
        Return string representation.

        :return: Description string.
        :rtype: str
        """
        return f"ThompsonSampling(true_reward={self.p}, posterior_mean={self.m:.4f}, pulls={self.N})"

    def pull(self):
        """
        Pull the arm: sample from Gaussian with mean = true reward.

        :return: Sampled reward.
        :rtype: float
        """
        return np.random.randn() + self.p

    def update(self, x):
        """
        Update Gaussian posterior parameters (conjugate update).

        :param x: Observed reward.
        :type x: float
        """
        self.N += 1
        self.sum_x += x
        # Conjugate Gaussian update:
        # new lambda = lambda_prior + N * tau
        # new m = (lambda_prior * m_prior + tau * sum_x) / new_lambda
        self.lambda_ = 1 + self.N * self.precision
        self.m = (self.precision * self.sum_x) / self.lambda_

    def sample(self):
        """
        Draw a sample from the current posterior distribution.

        :return: Sampled value of mu.
        :rtype: float
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def experiment(self, bandits, num_trials):
        """
        Run Thompson Sampling experiment across all bandits.

        :param bandits: List of ThompsonSampling bandit arms.
        :type bandits: list[ThompsonSampling]
        :param num_trials: Total number of trials to run.
        :type num_trials: int
        :return: Tuple of (rewards list, chosen bandits list).
        :rtype: tuple[list, list]
        """
        rewards = []
        chosen_bandits = []

        for _ in range(num_trials):
            # Sample from each bandit's posterior and pick the best
            chosen = np.argmax([b.sample() for b in bandits])
            reward = bandits[chosen].pull()
            bandits[chosen].update(reward)

            rewards.append(reward)
            chosen_bandits.append(chosen)

        return rewards, chosen_bandits

    def report(self, rewards, chosen_bandits, bandits, algorithm_name="ThompsonSampling"):
        """
        Report results: save CSV, log cumulative reward and regret.

        :param rewards: List of rewards obtained per trial.
        :param chosen_bandits: List of bandit indices chosen per trial.
        :param bandits: List of bandit arm objects.
        :param algorithm_name: Label for the algorithm.
        :type algorithm_name: str
        """
        best_reward = max(b.p for b in bandits)
        cumulative_reward = sum(rewards)
        cumulative_regret = best_reward * len(rewards) - cumulative_reward

        # Save to CSV
        df = pd.DataFrame({
            "Bandit": [b + 1 for b in chosen_bandits],
            "Reward": rewards,
            "Algorithm": algorithm_name
        })
        filename = f"{algorithm_name}_results.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to '{filename}'")

        logger.info(f"[{algorithm_name}] Cumulative Reward: {cumulative_reward:.4f}")
        logger.info(f"[{algorithm_name}] Cumulative Regret: {cumulative_regret:.4f}")

        return cumulative_reward, cumulative_regret


############################### COMPARISON

def comparison():
    """
    Run both algorithms, generate all plots, and compare performance visually.
    Stores results to CSV and logs cumulative reward and regret for both.
    """
    logger.info("Starting Epsilon-Greedy experiment...")
    eg_bandits = [EpsilonGreedy(p) for p in BANDIT_REWARDS]
    eg_instance = EpsilonGreedy(BANDIT_REWARDS[0])
    eg_rewards, eg_chosen = eg_instance.experiment(eg_bandits, NUM_TRIALS)
    eg_cum_rewards = np.cumsum(eg_rewards)
    eg_best = np.cumsum([max(BANDIT_REWARDS)] * NUM_TRIALS)
    eg_cum_regrets = eg_best - eg_cum_rewards
    eg_total_reward, eg_total_regret = eg_instance.report(eg_rewards, eg_chosen, eg_bandits)

    logger.info("Starting Thompson Sampling experiment...")
    ts_bandits = [ThompsonSampling(p) for p in BANDIT_REWARDS]
    ts_instance = ThompsonSampling(BANDIT_REWARDS[0])
    ts_rewards, ts_chosen = ts_instance.experiment(ts_bandits, NUM_TRIALS)
    ts_cum_rewards = np.cumsum(ts_rewards)
    ts_best = np.cumsum([max(BANDIT_REWARDS)] * NUM_TRIALS)
    ts_cum_regrets = ts_best - ts_cum_rewards
    ts_total_reward, ts_total_regret = ts_instance.report(ts_rewards, ts_chosen, ts_bandits)

    # Merge both CSVs into a single combined CSV
    eg_df = pd.read_csv("EpsilonGreedy_results.csv")
    ts_df = pd.read_csv("ThompsonSampling_results.csv")
    combined = pd.concat([eg_df, ts_df], ignore_index=True)
    combined.to_csv("combined_results.csv", index=False)
    logger.info("Combined results saved to 'combined_results.csv'")

    # Visualizations
    viz = Visualization()
    viz.plot1(eg_cum_rewards, ts_cum_rewards, eg_best, ts_best)
    viz.plot2(eg_cum_rewards, ts_cum_rewards, eg_cum_regrets, ts_cum_regrets)

    logger.info(
        f"Final Comparison | "
        f"EG Reward: {eg_total_reward:.2f}, EG Regret: {eg_total_regret:.2f} | "
        f"TS Reward: {ts_total_reward:.2f}, TS Regret: {ts_total_regret:.2f}"
    )


############################### MAIN

if __name__ == '__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    comparison()

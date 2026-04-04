# Homework 2 — Multi-Armed Bandit A/B Testing

## Overview
This project implements two multi-armed bandit algorithms — **Epsilon-Greedy** and **Thompson Sampling** — to simulate an A/B testing scenario with four advertisement options (bandits) with rewards `[1, 2, 3, 4]`.

## Algorithms
- **Epsilon-Greedy**: Explores with probability epsilon (decaying as `1/t`) and exploits the best known arm otherwise.
- **Thompson Sampling**: Uses a Gaussian posterior with known precision to balance exploration and exploitation via probabilistic sampling.

## Project Structure
```
├── Bandit.py                    # Main implementation
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── combined_results.csv         # Generated after running (both algorithms)
├── plot1_learning_process.png   # Generated after running
└── plot2_comparison.png         # Generated after running
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python Bandit.py
```

## Output
- **CSV files**: `EpsilonGreedy_results.csv`, `ThompsonSampling_results.csv`, `combined_results.csv` — each with columns `{Bandit, Reward, Algorithm}`
- **plot1_learning_process.png** — learning curve for each algorithm on linear and log scale
- **plot2_comparison.png** — side-by-side comparison of cumulative rewards and regrets
- **Console logs** — cumulative reward and regret for both algorithms printed via `loguru`

## Parameters
| Parameter | Value |
|-----------|-------|
| Bandit Rewards | [1, 2, 3, 4] |
| Number of Trials | 20,000 |
| Epsilon (initial) | 1/t (decaying) |
| Precision (tau) | 1 |

---

## Better Implementation Plan

### Current Limitations

The current implementation can be improved by replacing Gaussian Rewards with Bernoulli Rewards.

**Problem:** The current implementation samples rewards as `N(p, 1)` — a Gaussian centered at the true reward. In real A/B testing for advertisements, rewards are binary (click / no click), making a **Bernoulli distribution** far more realistic.

**Proposed change:**
```python
def pull(self):
    # More realistic: Bernoulli reward (click or no click)
    return np.random.binomial(1, self.p)
```

For Thompson Sampling with Bernoulli rewards, use the **Beta-Binomial conjugate model** instead of Gaussian:
```python
def update(self, x):
    self.alpha += x        # successes (clicks)
    self.beta += (1 - x)  # failures (no clicks)

def sample(self):
    return np.random.beta(self.alpha, self.beta)
```



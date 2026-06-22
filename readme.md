#  Mario PPO Agent

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-green)

An AI agent trained to play Super Mario Bros using Deep Reinforcement Learning with the PPO algorithm.

![Mario PPO Agent](mario_best.gif)

---

##  How it works

The agent learns by trial and error — it receives a reward when it moves forward and a penalty when it dies. Over time, it learns to run, jump over enemies, and avoid obstacles on its own.

The algorithm used is **PPO (Proximal Policy Optimization)**, a state-of-the-art RL algorithm that updates the agent's policy in small, stable steps to avoid forgetting what it already learned.

---

##  Tech Stack

| Tool | Role |
|------|------|
| `gym-super-mario-bros` | Mario environment |
| `stable-baselines3` | PPO implementation |
| `gymnasium` | Environment wrapper |
| `opencv` | Frame preprocessing |
| Google Colab T4 GPU | Training hardware |

---

##  Architecture

- **Observation space** : 84×84 grayscale frames, stacked 4 at a time
- **Action space** : SIMPLE_MOVEMENT (7 actions)
- **Policy** : CnnPolicy (convolutional neural network)
- **Frame skip** : 4 (1 action repeated over 4 frames)
- **Parallel envs** : 10 environments trained simultaneously

---

##  Training Config

```python
PPO(
    policy        = "CnnPolicy",
    learning_rate = 2.5e-4,
    n_steps       = 512,
    batch_size    = 64,
    n_epochs      = 4,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    n_envs        = 10       # parallel environments
)
```

---

##  Results

| Metric | Value |
|--------|-------|
| Total timesteps | 500,000 |
| Parallel environments | 10 |
| Best episode selected from | 300 runs |
| Behavior | Runs right, jumps Goombas, reaches flagpole |

The agent consistently moves right, jumps over Goombas, and reaches the flagpole on good runs.

---

##  Run it yourself

### 1. Open the notebook in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 2. Upload `mario_ppo_500k.zip` to Colab

### 3. Run all cells
- **Cell 1** — Install dependencies & apply patches
- **Cell 2** — Train the PPO agent (500k timesteps)
- **Cell 3** — Record the best episode as GIF + MP4

---

##  Files

```
mario-ppo-agent/
├── README.md
├── mario_agent.py        ← Python training script
├── mario_best.gif        ← demo (best episode)
├── mario_best.mp4        ← demo HD
├── mario_agent.ipynb     ← full notebook
└── mario_ppo_500k.zip    ← trained model
```

---

##  Author

**Imene Belmadoui** — AI & Data Science Engineer
[GitHub](https://github.com/imenei) · [LinkedIn](https://linkedin.com/in/imene-belmadoui)

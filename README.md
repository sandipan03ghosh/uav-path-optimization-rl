# UAV Path Optimization using Q-Learning

An implementation of a Reinforcement Learning agent (UAV/Drone) designed to navigate a 6x6 urban grid. The agent learns to avoid high-risk zones (Thunderstorms) and restricted airspace while minimizing battery consumption to reach a target destination.

## 🚀 The Challenge
* **Goal:** Reach $(5,5)$ from $(0,0)$.
* **Constraints:** * Thunderstorms: $-15$ penalty (End of Episode).
    * Restricted Airspace: $-5$ penalty.
    * Movement Cost: $-1$ per step (Battery Efficiency).

## 🧠 Algorithm
This project utilizes **Q-Learning** (Off-policy Temporal Difference Control). The update rule is based on the Bellman Equation:
$Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$

## 📈 Results
The agent successfully converged to an optimal policy after approximately 5,000 episodes.


## 🛠️ Requirements
- Python 3.x
- NumPy
- Matplotlib

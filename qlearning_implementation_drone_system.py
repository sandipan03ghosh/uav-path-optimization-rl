
import numpy as np
import random
import matplotlib.pyplot as plt

GRID_SIZE = 6
START_POS = (0, 0)
GOAL_POS = (5, 5)
STORM_ZONES = [(1, 1), (1, 2), (4, 4), (4, 3)] # High risk
RESTRICTED_AIRSPACE = [(0, 3), (2, 5), (3, 0)]  # Moderate risk

class DroneCity:
    """The Environment: A 6x6 City Grid"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_pos = START_POS
        return self.current_pos

    def get_reward(self, pos):
        if pos == GOAL_POS:
            return 20  # Big win
        if pos in STORM_ZONES:
            return -15 # Danger!
        if pos in RESTRICTED_AIRSPACE:
            return -5  # Fine/Penalty
        return -1      # Battery usage

    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        r, c = self.current_pos
        if action == 0: r = max(0, r - 1)
        elif action == 1: r = min(GRID_SIZE - 1, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(GRID_SIZE - 1, c + 1)

        self.current_pos = (r, c)
        reward = self.get_reward(self.current_pos)

        # End if Goal or Storm reached
        done = (self.current_pos == GOAL_POS or self.current_pos in STORM_ZONES)
        return self.current_pos, reward, done

class QDeliveryAgent:
    """The Brain: Q-Learning with Epsilon Decay"""
    def __init__(self, alpha=0.2, gamma=0.95, epsilon=1.0):
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        self.alpha = alpha     # Learning rate
        self.gamma = gamma     # Future discount
        self.epsilon = epsilon # Exploration rate
        self.decay = 0.9995    # Slowly stop being random
        self.reward_history = []

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3) # Explore
        return np.argmax(self.q_table[state[0], state[1]]) # Exploit

    def train(self, episodes=10000):
        env = DroneCity()
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)

                # Q-Learning Formula (The Bellman Update)
                old_value = self.q_table[state[0], state[1], action]
                next_max = np.max(self.q_table[next_state[0], next_state[1]])

                # Update rule
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                self.q_table[state[0], state[1], action] = new_value

                state = next_state
                total_reward += reward

            # Reduce randomness over time
            self.epsilon *= self.decay
            self.reward_history.append(total_reward)

    def visualize_policy(self):
        """Shows the best move for each square"""
        arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        print("\n--- OPTIMIZED DRONE FLIGHT PATH ---")
        for r in range(GRID_SIZE):
            row_str = "| "
            for c in range(GRID_SIZE):
                if (r, c) == GOAL_POS: row_str += "GOLD | "
                elif (r, c) in STORM_ZONES: row_str += "TRAP | "
                else:
                    best_action = np.argmax(self.q_table[r, c])
                    row_str += f" {arrows[best_action]}  | "
            print(row_str)

if __name__ == "__main__":
    agent = QDeliveryAgent()
    print("Training Drone for 10,000 deliveries...")
    agent.train(10000)

    # Plotting Results
    plt.figure(figsize=(10, 5))
    plt.plot(agent.reward_history)
    plt.title("Drone Learning Curve")
    plt.ylabel("Total Reward per Delivery")
    plt.xlabel("Delivery Attempt")
    plt.show()

    agent.visualize_policy()

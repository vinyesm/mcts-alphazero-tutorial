import heapq
import gymnasium as gym
import time
from copy import deepcopy
import numpy as np


class Node:
    def __init__(self, state, action, parent, cost, terminated, truncated, rng=None, elapsed_steps=0):
        self.state = state
        self.action = action
        self.parent = parent
        self.cost = cost
        self.terminated = terminated
        self.truncated = truncated
        self.rng = deepcopy(rng)  # Save the RNG state
        self.elapsed_steps = elapsed_steps  # Save the elapsed steps


    def heuristic(self):
        """
        Define the heuristic: prioritize states where the pole angle
        and angular velocity are minimized.
        """
        _, _, pole_angle, pole_ang_vel = self.state
        return abs(pole_angle) + abs(pole_ang_vel)
        

    def get_path(self):
        """
        Reconstruct the path of actions from the root to this node.
        """
        path = []
        current = self
        while current.parent is not None:
            path.append((current.action, current.state))
            current = current.parent
        return path[::-1]  # Reverse to get the correct order

    def __lt__(self, other):
        """
        Comparison method for the priority queue. Nodes with lower cost
        are given higher priority.
        """
        return self.cost < other.cost


def best_first_search(env, max_steps=200):
    """
    Perform Best-First Search to balance the CartPole as long as possible.

    Args:
        env: Gym environment for CartPole.
        max_steps: Maximum steps to explore.

    Returns:
        Longest sequence of actions leading to valid states.
    """
    # initial_state, _ = env.reset()
    initial_state = deepcopy(env.unwrapped.state)
    initial_rng = deepcopy(env.unwrapped.np_random)
    root = Node(
        initial_state, 
        None, 
        None, 
        cost=0.0, 
        terminated=False, 
        truncated=False, 
        rng=initial_rng
        )

    # Priority queue to explore nodes based on their heuristic cost
    priority_queue = []
    heapq.heappush(priority_queue, (root.cost, root))

    # Set to track visited states
    visited = set()

    # Variable to store the longest valid path found
    longest_path = []
    steps = 0

    while priority_queue and steps < max_steps:
        # Pop the node with the lowest heuristic cost
        _, current_node = heapq.heappop(priority_queue)

        # Reconstruct the current path
        current_path = current_node.get_path()

        # If the node is terminal, skip further exploration
        if current_node.terminated:
            print(f"Terminated... exploring next node.")
            continue

        if current_node.truncated:
            print(f"Truncated at step {len(current_node.get_path())}")
            return current_path

        # Update the longest path if the current path is longer
        if len(current_path) > len(longest_path):
            longest_path = current_path
            print(f"Step: {steps}, Longest path: {len(longest_path)}")

        # Generate child nodes by simulating actions (0 or 1)
        for action in [0, 1]:
            env.unwrapped.state = deepcopy(current_node.state)
            env.unwrapped.np_random = deepcopy(current_node.rng)
            setattr(env, "_elapsed_steps", current_node.elapsed_steps)

            new_state, _, terminated, truncated, _ = env.step(action)

            # Simplified state representation for the visited set
            state_tuple = tuple(round(s, 6) for s in new_state)  # Higher precision
            if state_tuple not in visited:
                visited.add(state_tuple)
                # Add the child node to the priority queue
                cost = abs(new_state[0]) + abs(new_state[2])  # Cart position + pole angle
                # cost = 0
                child_node = Node(
                    new_state, 
                    action, 
                    current_node, 
                    cost, 
                    terminated, 
                    truncated,
                    rng=deepcopy(env.unwrapped.np_random),
                    elapsed_steps=current_node.elapsed_steps + 1,
                    )
                heapq.heappush(priority_queue, (cost, child_node))
        steps += 1

    return longest_path


# def replay_solution(env, init_state, rng, actions, render=True, delay=0.05):
#     """
#     Replay the solution to verify its correctness.

#     Args:
#         env: Gym environment for CartPole.
#         actions: List of actions to replay.
#         render: Whether to render the environment.
#         delay: Delay between actions for visualization.
#     """
#     env.reset()
#     env.unwrapped.state = init_state
#     env.unwrapped.np_random = rng
#     # print(f"Restored RNG state: {env.unwrapped.np_random.bit_generator.state}")
#     total_reward = 0

#     for i, res in enumerate(actions):
#         action, bfs_state = res
#         if render:
#             env.render()
#         state, reward, terminated, truncated, _ = env.step(action)
#         print(f"state type: {type(state[0])}")
#         total_reward += reward
#         print(f"Step {i + 1}: Action: {action}, State: {state}, BFS_State: {bfs_state}, Reward: {reward}")

#         if terminated or truncated:
#             print(f"Episode ended at step {i + 1}. Total reward: {total_reward}")
#             break

#         time.sleep(delay)

#     if not (terminated or truncated):
#         print(f"Solution executed successfully for all {len(actions)} steps.")
#         print(f"Total reward: {total_reward}")

#     env.close()

# if __name__ == "__main__":
#     # Create the CartPole environment and wrap it with CloneableEnv
#     env = gym.make("CartPole-v1", render_mode="rgb_array")
#     env.reset()
#     env.unwrapped.state = np.array(env.unwrapped.state, dtype=np.float32)
#     init_state = deepcopy(env.unwrapped.state)
#     rng = deepcopy(env.unwrapped.np_random)
#     # print(f"initial state: {env.unwrapped.state}")
#     # print(f"Restored RNG state: {env.unwrapped.np_random.bit_generator.state}")

#     # Perform Best-First Search
#     actions = best_first_search(env, max_steps=100)
#     replay_solution(env, init_state, rng, actions, render=False)

#     # Display the results
#     if actions:
#         # print(f"Found solution with {len(actions)} actions: {actions}")
#         print(f"Found solution with {len(actions)} actions.")
#         print("Replaying solution...")
#         replay_solution(env, init_state, rng, actions, render=False)
#     else:
#         print("No solution found.")

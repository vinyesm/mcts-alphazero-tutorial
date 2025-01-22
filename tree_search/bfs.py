import heapq
import gymnasium as gym
import time
from copy import deepcopy


class Node:
    def __init__(self, state, action, parent, cost, done, rng=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.cost = cost
        self.done = done
        self.rng = deepcopy(rng)  # Save the RNG state


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
            path.append(current.action)
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
    initial_state, _ = env.reset()
    initial_rng = deepcopy(env.unwrapped.np_random)
    root = Node(initial_state, None, None, cost=0.0, done=False, rng=initial_rng)

    # Priority queue to explore nodes based on their heuristic cost
    priority_queue = []
    heapq.heappush(priority_queue, (root.cost, root))

    # Set to track visited states
    visited = set()

    # Variable to store the longest valid path found
    longest_path = []

    while priority_queue and len(visited) < max_steps:
        # Pop the node with the lowest heuristic cost
        _, current_node = heapq.heappop(priority_queue)

        # Restore the environment to the current node's state
        # env.reset()
        # env.unwrapped.state = deepcopy(current_node.state)
        # env.unwrapped.np_random = deepcopy(current_node.rng)

        # Reconstruct the current path
        current_path = current_node.get_path()

        # Update the longest path if the current path is longer
        if len(current_path) > len(longest_path):
            longest_path = current_path
            print(f"Step: {len(visited)}, Longest path: {len(longest_path)}")

        # If the node is terminal, skip further exploration
        if current_node.done:
            continue

        # Generate child nodes by simulating actions (0 or 1)
        for action in [0, 1]:
            # Save the current environment state
            env.reset()
            env.unwrapped.state = deepcopy(current_node.state)
            env.unwrapped.np_random = deepcopy(current_node.rng)
            # rng = deepcopy(env.unwrapped.np_random)
            # state = deepcopy(env.unwrapped.state)

            # Perform the action and observe the new state
            new_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # print(f"Step: {len(visited)}, Action: {action}, New state: {new_state}, Done: {done}")
            # print(f"current path {current_path}")
            # print(f"Step: {len(visited)}, Action: {action}, New state: {new_state}, Done: {done}")
            # print(f"Cart Position: {new_state[0]}, Pole Angle: {new_state[2]}")

            # Restore the environment state after the action
            # env.state = state
            # env.unwrapped.np_random = rng

            if done:
                print("DONE is reached!!!")

            # Simplified state representation for the visited set
            state_tuple = tuple(round(s, 6) for s in new_state)  # Higher precision
            if state_tuple not in visited:
                visited.add(state_tuple)
                # Add the child node to the priority queue
                cost = abs(new_state[0]) + abs(new_state[2])  # Cart position + pole angle
                child_node = Node(new_state, action, current_node, cost, done, rng=deepcopy(env.unwrapped.np_random))
                heapq.heappush(priority_queue, (cost, child_node))

    return longest_path


def replay_solution(env, actions, render=True, delay=0.05):
    """
    Replay the solution to verify its correctness.

    Args:
        env: Gym environment for CartPole.
        actions: List of actions to replay.
        render: Whether to render the environment.
        delay: Delay between actions for visualization.
    """
    state, _ = env.reset()
    total_reward = 0

    for i, action in enumerate(actions):
        if render:
            env.render()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {i + 1}. Total reward: {total_reward}")
            break

        time.sleep(delay)

    if not (terminated or truncated):
        print(f"Solution executed successfully for all {len(actions)} steps.")
        print(f"Total reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    # Create the CartPole environment and wrap it with CloneableEnv
    base_env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = base_env

    # Perform Best-First Search
    actions = best_first_search(env, max_steps=5000)

    # Display the results
    if actions:
        print(f"Found solution with {len(actions)} actions: {actions}")
        print("Replaying solution...")
        replay_solution(env, actions, render=False)
    else:
        print("No solution found.")

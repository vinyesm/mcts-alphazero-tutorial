from copy import deepcopy
import random
from collections import deque
import logging
from typing import List, Dict
import math

import fire
import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from tree_search.mcts import Policy_Player_MCTS, Node
from tree_search.mcts_alphazero import Node as Node_AlphaZero
from tree_search.mcts_alphazero import Policy_Player_AlphaZero, ReplayBuffer
from tree_search.bfs import best_first_search
from model.model import ValueModel, PolicyModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logger")

torch.set_default_dtype(torch.float64)


def load_config(config_file: str) -> Dict:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def get_policy_params(config: Dict) -> Dict:
    policy_name = config.get("policy_name")
    return config.get("policies", {}).get(policy_name, {})


def save_plot(rewards: List[float], moving_average: List[float], filename: str = "rewards_plot.png") -> None:
    plt.plot(rewards, label="Rewards")
    plt.plot(moving_average, label="Moving Average")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.legend()
    plt.savefig(filename)
    plt.clf()
    logger.info(f"Plot saved as {filename}")


def save_video(frames: List[np.ndarray], filename: str = 'output.mp4', fps: int = 30) -> None:
    """Create a video from the list of frames, ensuring frames are in RGB format."""
    processed_frames = []
    for frame in frames:
        frame = np.array(frame)
        if len(frame.shape) == 2:  # Grayscale image (H, W)
            frame = np.stack([frame] * 3, axis=-1)  # Convert to RGB (H, W, 3)
        elif len(frame.shape) == 3 and frame.shape[-1] not in [1, 3, 4]:
            raise ValueError(f"Unexpected frame shape: {frame.shape}. Ensure frames are valid images.")
        processed_frames.append(frame)
    imageio.mimsave(filename, processed_frames, fps=fps)
    logger.info(f"Video saved as {filename}")




def train(config_file: str) -> None:
    # Load configuration and initialize W&B
    config = load_config(config_file)

    game_name = config.get('game_name')
    render = config.get('render')
    episodes = config.get('episodes')
    policy_name = config.get('policy_name', 'random')
    policy_params = get_policy_params(config)

    # Initialize environment
    with gym.make(game_name, render_mode="rgb_array") as env:
        n_actions = env.action_space.n
        n_observation = env.observation_space.shape[0]
        print(f"game_name: {game_name} policy_name: {policy_name}")
        print(f"n_actions: {n_actions}, n_observation: {n_observation}")
        
        if policy_name == "alphazero":
            value_model = ValueModel(n_observation)
            policy_model = PolicyModel(n_observation, n_actions)
            value_optimizer = optim.Adam(value_model.parameters(), lr=0.001)
            policy_optimizer = optim.Adam(policy_model.parameters(), lr=0.001)
            buffer_size = policy_params.get("buffer_size", 1000)
            batch_size = policy_params.get("batch_size", 128)
            update_every = policy_params.get("update_every", 1)
            n_explore = policy_params.get("n_explore")
            max_reward = policy_params.get("max_reward", 1)
            c = policy_params.get("c", math.sqrt(2))
            replay_buffer = ReplayBuffer(buffer_size, batch_size)
        elif policy_name == "mcts":
            n_explore = policy_params.get("n_explore")
            max_reward = policy_params.get("max_reward", 1)
            c = policy_params.get("c", math.sqrt(2))
        elif policy_name == "bfs":
            n_explore = policy_params.get("n_explore", 100)
            c = policy_params.get("c", 0.0)

        if False:
            wandb.init(project="mcts_actor_critic", config=config, name=f"{game_name} {policy_name}")

        rewards = []
        moving_average = []
        v_losses = []
        p_losses = []
        frames = []

        for e in range(episodes):
            print(f"Episode {e + 1}/{episodes}")
            step = 0
            observation, _ = env.reset()
            done = False
            episode_reward = torch.tensor(0, dtype=torch.float64)

            # Create a new tree
            rng = env.unwrapped.np_random
            new_env = gym.make(env.spec.id)
            new_env.reset()
            new_env.unwrapped.state = deepcopy(env.unwrapped.state)
            new_env.unwrapped.np_random = deepcopy(rng)
            if policy_name == "alphazero":
                mytree = Node_AlphaZero(new_env, None, 0, observation, False, value_model, policy_model, c)
                obs, ps, p_obs = [], [], []
            elif policy_name == "mcts":
                mytree = Node(new_env, None, 0, observation, False, c)
            elif policy_name == "bfs":
                print(n_explore)
                actions = best_first_search(new_env, max_steps=n_explore)
                if actions:
                    print(f"Found solution with {len(actions)} actions.")
                else:
                    print("No solution found.")
            
            while not done:
                if policy_name == "random":
                    action = env.action_space.sample()
                elif policy_name == "mcts":
                    mytree, action = Policy_Player_MCTS(mytree, n_explore)
                elif policy_name == "alphazero":
                    mytree, action, ob, p, p_ob = Policy_Player_AlphaZero(mytree, n_explore)
                    obs.append(ob)
                    ps.append(p)
                    p_obs.append(p_ob)
                elif policy_name == "bfs":
                    if step >= len(actions):
                        print("No more actions, increase exploration.")
                        break
                    action, _ = actions[step]
                
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                print(f"Step {step}: Action: {action}, Reward: {reward}, Done: {done}")

                if done:
                    if policy_name == "alphazero":
                        for i in range(len(obs)):
                            replay_buffer.add(obs[i], episode_reward.item(), p_obs[i], ps[i])
                    new_env.close()
                    break


                step += 1
                if render:
                    frame = env.render()
                    frames.append(frame)

            rewards.append(episode_reward.item())
            moving_average.append(np.mean(rewards[-100:]))
            logger.info(f"PLAY: Episode {e + 1}/{episodes}: Reward: {episode_reward}")
            if False:
                wandb.log({
                    "episode_reward": episode_reward.item(),
                    "moving_average": moving_average[-1],
                    "steps_per_episode": step
                })

            # Training loop (only for AlphaZero)
            if policy_name == "alphazero":
                if (e + 1) % update_every == 0 and len(replay_buffer) > batch_size:
                    n_iter = len(replay_buffer) // batch_size
                    for i in range(n_iter):
                        experiences = replay_buffer.sample()
                        obs, r, p_obs, ps = zip(*experiences)
                        obs = np.array(obs)
                        obs_tensor = torch.tensor(obs, dtype=torch.float64)
                        r = torch.tensor(r, dtype=torch.float64).view(-1, 1)

                        # Update Value Model
                        loss_v = nn.MSELoss()(value_model(obs_tensor), r / max_reward)
                        value_optimizer.zero_grad()
                        loss_v.backward()
                        value_optimizer.step()
                        v_losses.append(loss_v.item())
                
                        for name, param in value_model.named_parameters():
                            if param.grad is not None:
                                wandb.log({f"gradient_norm_{name}": param.grad.norm().item()})

                        # Update Policy Model
                        p_obs = torch.tensor(np.array(p_obs), dtype=torch.float64)
                        ps = torch.tensor(np.array(ps), dtype=torch.float64)
                        loss_p = nn.CrossEntropyLoss()(policy_model(p_obs), ps.argmax(dim=-1))
                        policy_optimizer.zero_grad()
                        loss_p.backward()
                        policy_optimizer.step()
                        p_losses.append(loss_p.item())

                        logger.info(f"TRAIN: Episode {e + 1}/{episodes}: Reward: {episode_reward} loss_v: {loss_v:0.5f} loss_p: {loss_p:0.5f}")
                        wandb.log({
                            "value_loss": loss_v.item(),
                            "policy_loss": loss_p.item()
                        })

        save_plot(rewards, moving_average, f'{game_name}_{policy_name}_rewards.png')
        save_video(frames, filename=f'{game_name}_{policy_name}_simulation.mp4')

    # Log final results to W&B
    if False:
        wandb.log({
            "final_rewards_plot": wandb.Image(f'{game_name}_{policy_name}_rewards.png'),
            "final_video": wandb.Video(f'{game_name}_{policy_name}_simulation.mp4', fps=30, format="mp4")
        })
        wandb.finish()

if __name__ == "__main__":
    fire.Fire(train)
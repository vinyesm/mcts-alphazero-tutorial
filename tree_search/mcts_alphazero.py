from typing import Optional, Dict, Tuple
from collections import namedtuple, deque
from copy import deepcopy
import random
import math
import numpy as np

import torch
import gymnasium as gym

def Policy_Player_AlphaZero(mytree, n_explore=100):  
    '''
    Our strategy for using the MCTS is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to MCTS, is the best possible next action
    '''
    for i in range(n_explore):
        mytree.explore()  
    next_tree, next_action, obs, p, p_obs = mytree.next()
        
    next_tree.detach_parent()
    
    return next_tree, next_action, obs, p, p_obs


'''
The ReplayBuffer stores game plays that we will use for neural network training. It stores, in particular:
    - The observation (i.e. state) of the game environment
    - The target Value
    - The observation (i.e. state) of the game environment at the previous step
    - The target Policy according to visit counts 
'''

class ReplayBuffer:
  """Fixed-size buffer to store experience tuples."""
  def __init__(self, buffer_size, batch_size):
    """Initialize a ReplayBuffer object.

    Params
    ======
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
    """
    self.memory = deque(maxlen=buffer_size)  
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["obs", "v", "p_obs", "p"])

  def add(self, obs, v, p, p_obs):
    """Add a new experience to memory."""
    e = self.experience(obs, v, p, p_obs)
    self.memory.append(e)

  def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)
    return experiences

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)


class Node:
  '''
  The Node class represents a node of the MCTS tree. 
  It contains the information needed for the algorithm to run its search.
  '''
  def __init__(
    self, 
    game: gym.Env, 
    parent: Optional['Node'], 
    action: int, 
    observation, 
    done: bool, 
    value_model=None, 
    policy_model=None,
    c: float = math.sqrt(2)):
    self.game = game
    self.child = None # children nodes
    self.parent = parent # parent node
    self.action = action # action leading to this node 
    self.observation = observation # current state of the game
    self.N = 0 # visit count
    self.W = 0 # total reward
    self.done = done # won/loss/draw
    self.value_model = value_model # the value model
    self.policy_model = policy_model # the policy model
    self.nn_v = 0 # the value of the node according to nn
    self.nn_p = None # the next probabilities
    self.c = c


  def getUCBscore(self) -> float:
    '''
    This is the formula that gives a value to the node.
    The MCTS will pick the nodes with the highest value.        
    '''

    if self.N == 0:
      return float('inf')

    top_node = self
    if top_node.parent:
        top_node = top_node.parent

    exploit_term = self.W/self.N
    explore_term = self.c * math.sqrt(math.log(self.parent.N)/(self.N + 1))
    explore_term = self.parent.nn_p[0][self.action] * explore_term

    return exploit_term + explore_term


  def create_child(self) -> None:     
    '''
    We create one children for each possible action of the game, 
    then we apply such action to a copy of the current node enviroment 
    and create such child node with proper information returned from the action executed
    '''

    if self.done:
        return

    actions = []
    child = {}

    for action in range(self.game.action_space.n): 
        with gym.make(self.game.spec.id) as new_game:
            new_game.reset()
            new_game.unwrapped.state = deepcopy(self.game.unwrapped.state)
            new_game.unwrapped.np_random = self.game.unwrapped.np_random

            observation, reward, terminated, truncated, _ = new_game.step(action)
            done = terminated or truncated

            child[action] = Node(new_game, self, action, observation, done, self.value_model, self.policy_model, self.c)

    self.child = child


  def detach_parent(self) -> None:
    """
    Detach this node from its parent, effectively making it a new root node.
    """
    if self.parent:
      action_to_remove = None
      for action, child in self.parent.child.items():
        if child is self:  # Check if the current node is the child
            action_to_remove = action
            break

      if action_to_remove is not None:
        del self.parent.child[action_to_remove] 
    self.parent = None


  def explore(self) -> None:  
    '''
    The search along the tree is as follows:
    - from the current node, recursively pick the children which maximizes the value according to the MCTS formula
    - when a leaf is reached:
        - if it has never been explored before, do a rollout and update its current value
        - otherwise, expand the node creating its children, pick one child at random, do a rollout and update its value
    - backpropagate the updated statistics up the tree until the root: update both value and visit counts
    ''' 

    current = self

    while current.child:
      # pick children which maximize value according to MCTS
      children = current.child
      max_U = -float('inf')

      for action, child in children.items():
        U = child.getUCBscore()
        if U > max_U:
          max_U = U
          current = child

    if current.N < 1: #if never visited
      current.nn_v, current.nn_p = current.rollout()
      current.W = current.W + current.nn_v
    else:
      current.create_child()
      if current.child: #game not done yet
        current = random.choice(current.child)
      current.nn_v, current.nn_p = current.rollout()
      current.W = current.W + current.nn_v

    current.N += 1 

    parent = current
    while parent.parent:
      parent = parent.parent
      parent.N += 1
      parent.W += current.W


  def rollout(self, temperature: float = 1.0) -> None:
    '''
    The rollout is where we use the neural network estimations to approximate the Value and Policy of a given node.
    With the trained neural network, it will give us a good approximation even in large state spaces searches.
    '''
    if self.done:
        return 0, None
    
    obs = np.array([self.observation]) 
    obs_tensor = torch.tensor(obs, dtype=torch.float64)
    with torch.no_grad():
      v = self.value_model(obs_tensor)
      
      # p = self.policy_model(obs_tensor)
      logits = self.policy_model(obs_tensor)
      scaled_logits = logits / temperature  # Apply temperature
      p = torch.softmax(scaled_logits, dim=-1)

    return v, p


  def next(self) -> Tuple['Node', int]:
    ''' 
    Once we have done enough search in the tree, the values contained in it should be statistically accurate.
    We will at some point then ask for the next action to play from the current node, and this is what this function does.
    There may be different ways on how to choose such action, in this implementation the strategy is as follows:
    - pick at random one of the node which has the maximum visit count, as this means that it will have a good value anyway.
    '''
    if self.done:
            raise ValueError("game has ended")

    if not self.child:
        raise ValueError('no children found and game hasn\'t ended')
    
    children = self.child
    max_N = max(node.N for node in children.values())
    probs = [ node.N / max_N for node in children.values() ]
    probs /= np.sum(probs)

    next_child = np.random.choice(list(children.values()), p=probs)

    return next_child, next_child.action, next_child.observation, probs, self.observation
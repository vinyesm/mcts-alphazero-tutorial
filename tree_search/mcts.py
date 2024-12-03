from typing import Optional, Dict, Tuple
from copy import deepcopy
import random
import math
import numpy as np

import torch
import gymnasium as gym


def Policy_Player_MCTS(mytree, n_explore=100):  
    '''
    Our strategy for using the MCTS is quite simple:
    - in order to pick the best move from the current node:
        - explore the tree starting from that node for a certain number of iterations to collect reliable statistics
        - pick the node that, according to MCTS, is the best possible next action
    '''
    for i in range(n_explore):
        mytree.explore()  
    next_tree, next_action = mytree.next()
        
    # note that here we are detaching the current node and returning the sub-tree 
    # that starts from the node rooted at the choosen action.
    # The next search, hence, will not start from scratch but will already have collected information and statistics
    # about the nodes, so we can reuse such statistics to make the search even more reliable!
    next_tree.detach_parent()
    
    return next_tree, next_action


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
    c = math.sqrt(2)):
    self.game = game
    self.child = None # children nodes
    self.parent = parent # parent node
    self.action = action # action leading to this node 
    self.observation = observation # current state of the game
    self.N = 0 # visit count
    self.W = 0 # total reward
    self.done = done # won/loss/draw
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
    explore_term = self.c * math.sqrt(math.log(self.parent.N)/self.N)

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

            child[action] = Node(new_game, self, action, observation, done, self.c)

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

    if current.N < 1: # if never visited
      v = current.rollout()
      current.W = current.W + v
    else:
      current.create_child()
      if current.child: # game not done yet
        current = random.choice(current.child)
      v = current.rollout()
      current.W = current.W + v

    current.N += 1 

    parent = current
    while parent.parent:
      parent = parent.parent
      parent.N += 1
      parent.W += current.W


  def rollout(self) -> None:
    '''
    The rollout is a random play from a copy of the environment of the current node using random moves.
    This will give us a value for the current node.
    Taken alone, this value is quite random, but, the more rollouts we will do for such node,
    the more accurate the average of the value for such node will be. This is at the core of the MCTS algorithm.
    '''
    if self.done:
      return 0, None
     
    v = 0
    done = False
    with gym.make(self.game.spec.id) as new_game:     
        rng = self.game.unwrapped.np_random
        new_game.reset()
        new_game.unwrapped.state = deepcopy(self.game.unwrapped.state)
        new_game.unwrapped.np_random = rng  # Use the same RNG
        while not done:
            action = new_game.action_space.sample()
            _, reward, terminated, truncated, _ = new_game.step(action)
            done = terminated or truncated
            v += reward
    return v

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
    max_N = -1
    max_children = []

    for child in self.child.values():
        if child.N > max_N:
            max_N = child.N
            max_children = [child]
        elif child.N == max_N:
            max_children.append(child)

    if len(max_children) == 0:
      print("error zero length ", max_N) 

    probs = None      
    next_child = random.choice(max_children)

    return next_child, next_child.action
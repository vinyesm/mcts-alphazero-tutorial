import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueModel(nn.Module):
    '''
    The Value Neural Network will approximate the Value of the node, given a State of the game.
    '''
    def __init__(self, in_features, hidden_states=64):
        super(ValueModel, self).__init__()
        self.hidden_states = hidden_states
        self.in_features = in_features
        self.dense1 = nn.Linear(in_features=in_features, out_features=hidden_states)
        self.dense2 = nn.Linear(hidden_states, hidden_states)
        self.v_out = nn.Linear(hidden_states, 1)

    def forward(self, x):
        x =  F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.v_out(x)
        return x


class PolicyModel(nn.Module):
    '''
    The Policy Neural Network will approximate the MCTS policy for the choice of nodes, given a State of the game.
    '''
    def __init__(self, in_features, n_actions, hidden_states=64):
        super(PolicyModel, self).__init__()
        self.hidden_states = hidden_states
        self.in_features = in_features
        self.dense1 = nn.Linear(in_features=in_features, out_features=hidden_states)  
        self.dense2 = nn.Linear(hidden_states, hidden_states)
        self.p_out = nn.Linear(hidden_states, n_actions)

    def forward(self, x):
        x =  F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.softmax(self.p_out(x), dim=-1)
        return x
import torch.nn as nn
import torch

class MLP(nn.Module):
  def __init__(self, n_gaussians, n_features, aud_in, emo_in):
    super(MLP, self).__init__()
    self.n_gaussians = n_gaussians
    
    self.a_input = nn.Linear(aud_in, 500)
    self.e_input = nn.Linear(emo_in, 500)
    
    self.fc1 = nn.Linear(500, 5000)
    self.fc2 = nn.Linear(5000, 10000)
    self.fc3 = nn.Linear(10000, n_gaussians*n_features)
    
    self.relu = nn.ReLU()

  def forward(self, audio_feats, emo_feats):
    
    a = self.relu(self.a_input(audio_feats).flatten())
    e = self.relu(self.e_input(emo_feats).flatten())
    
    c = a + e
    
    c = self.relu(self.fc1(c))
    c = self.relu(self.fc2(c))
    c = self.relu(self.fc3(c))
    
    return c

class MLP_Less_Params(nn.Module):
  def __init__(self, n_gaussians, n_features, aud_in, emo_in):
    super(MLP_Less_Params, self).__init__()
    self.n_gaussians = n_gaussians
    
    self.a_input = nn.Linear(aud_in, 100)
    self.e_input = nn.Linear(emo_in, 100)
    
    self.fc1 = nn.Linear(100, 500)
    self.fc2 = nn.Linear(500, 2000)
    self.fc3 = nn.Linear(2000, n_gaussians*n_features)
    
    self.relu = nn.ReLU()

  def forward(self, audio_feats, emo_feats):
    
    a = self.relu(self.a_input(audio_feats).flatten())
    e = self.relu(self.e_input(emo_feats).flatten())
    
    c = a + e
    
    c = self.relu(self.fc1(c))
    c = self.relu(self.fc2(c))
    c = self.relu(self.fc3(c))
    
    return c
import torch.nn as nn
import torch

# FLATTEN SHOULD NOT BE NEEDED

class DeltaMLP(nn.Module):
  def __init__(self, aud_in, emo_in):
    super(DeltaMLP, self).__init__()

    self.a_input = nn.Linear(aud_in, 500)
    self.e_input = nn.Linear(emo_in, 500)

    self.fc1 = nn.Linear(500, 200)

    self.fc_xyz = nn.Linear(200, 3)
    self.fc_opa = nn.Linear(200, 1)
    self.fc_ftr = nn.Linear(200, 48)
    self.fc_sca = nn.Linear(200, 3)
    self.fc_rot = nn.Linear(200, 4)

    self.relu = nn.ReLU()
  
  def forward(self, audio_feats, emo_feats):

    a = self.relu(self.a_input(audio_feats).flatten())
    e = self.relu(self.e_input(emo_feats).flatten())
    
    c = a + e
    c = self.relu(self.fc1(c))

    deform_xyz = self.fc_xyz(c)
    deform_opa = self.fc_opa(c)
    deform_ftr = self.fc_ftr(c)
    deform_sca = self.fc_sca(c)
    deform_rot = self.fc_rot(c)

    return {
      'deform_xyz': deform_xyz,
      'deform_opa': deform_opa,
      'deform_ftr': deform_ftr,
      'deform_sca': deform_sca,
      'deform_rot': deform_rot,
    }


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

class MLP_New(nn.Module):
  def __init__(self, n_features, aud_in, emo_in):
    super(MLP_New, self).__init__()

    self.a_input = nn.Linear(aud_in, 500)
    self.e_input = nn.Linear(emo_in, 500)
    
    self.fc1 = nn.Linear(500, 100)
    self.fc2 = nn.Linear(100, n_features)
    
    self.relu = nn.ReLU()

  def forward(self, audio_feats, emo_feats):
    
    a = self.relu(self.a_input(audio_feats))
    e = self.relu(self.e_input(emo_feats))
    
    c = a + e
    
    c = self.relu(self.fc1(c))
    c = self.relu(self.fc2(c))
    
    return c
import torch
import torchvision
import numpy as np

def update_last_layer(model, out_features=1):
    '''changes the last fully connected layer to a (_, out_features) fc layer. 
       set out_features to 1 for regression.'''
    n_features = model.fc.in_features
    model.fc = nn.linear(n_features, out_features)
    return model


def truncated_normal(t, mean=0.0, std=0.01):
    '''samples from normal distribution of mean _mean_ and std _std_, 
       adapted from tf.truncated_normal, used in (Yeh & al., 2020)'''
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
      cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
      if not torch.sum(cond):
        break
      t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
    return t

    
def update_first_layer(model, in_channels, weights_init='random', scaling=1.):
    '''changes the first Conv2d layer to take in iputs as many _in_channels_
       and init its weights. 
       
       In case of additional channels, this functions offers
       two _weights_init_ methods : 
       - 'random' for a random initialization
       - 'average' for a pre-trained RGB-weights averaged initialization
       
       _scaling_ can lower the initial weights to avoid biased image representation.
       Especially recommended when shifting domains.'''
   ref_layer = model.conv1
   new_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
   # For now, all weights of added input channels are set to 0. 
   with torch.no_grad():
       n_additional_channels = in_channels - ref_layer.weight.shape[2]
       # F = 7 for ResNet18 -> need to generalize this 
       additional_weights = np.zeros((7, 7, n_additional_channels, 64))
       if weights_init == 'random':
           rgb_mean = np.mean(ref_layer.weight)
           rgb_std = np.std(ref_layer.weight)
           additional_weights = truncated_normal( additional_weights.to_tensor(), mean=rgb_mean, std=rgb_std )
       if weights_init == 'average':
          rgb_mean = rgb_weights.mean(axis=2, keepdims=True)  # shape [F, F, 1, 64]
          additional_weights = np.tile(rgb_mean, (1, 1, n_additional_channels, 1))
      else:
          raise ValueError(f'Unknown weight initialization method')
    
      new_layer.weight[:,:,in_channels-n_additional_channels:] = additional_weights
      # Scaling 
      new_layer.weight *= scaling
      model.conv1 = new_layer
    return model


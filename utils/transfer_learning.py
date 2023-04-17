'''
Utils functions to adapt pre-trained networks' layers and weights.
'''
import torch
import numpy as np

def update_last_layer(model, out_features=1):
    '''changes the last fully connected layer to a (_, out_features) fc layer. 
       set out_features to 1 for regression.'''
    n_features = model.fc.in_features
    model.fc = torch.nn.Linear(n_features, out_features)
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
        # Exception for single channel networks
        if n_additional_channels < 0:
            n_additional_channels = in_channels
        # F = 7 for ResNet18 -> need to generalize this 
        additional_weights = np.zeros((7, 7, n_additional_channels, 64))
        rgb_weights = ref_layer.weight
        if weights_init == 'random':
            rgb_mean = np.mean(rgb_weights)
            rgb_std = np.std(rgb_weights)
            additional_weights = truncated_normal( additional_weights.to_tensor(), mean=rgb_mean, std=rgb_std )
        if weights_init == 'average':
            rgb_mean = rgb_weights.mean(axis=1, keepdims=True)  # shape [F, F, 1, 64]
            additional_weights = np.tile(rgb_mean, (1, 1, n_additional_channels, 1))
        else:
            raise ValueError(f'Unknown weight initialization method')
        for i in range(n_additional_channels):
            new_layer.weight[:,(in_channels-n_additional_channels)+i:] = torch.from_numpy(additional_weights)
        # Scaling 
        new_layer.weight *= scaling
        model.conv1 = new_layer

    return model


def s2_to_landsat(model : torch.nn.Module) -> torch.nn.Module:
    """Updates the 13-bands sentinel-2 pretrained model to a 8-bands landsat one.

    Args:
        model (n.Module): model to be updated
    """
    conv1 = model.conv1
    new_layer = torch.nn.Conv2d(in_channels=7, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    s2_bands_to_keep = np.array([ 1, 2, 3, 7, 10, 11 ])
    with torch.no_grad():
        new_weights = np.zeros((64, 7, 7, 7))
        for band in range(len(s2_bands_to_keep)):
            new_layer.weight[:,band,:,:] = conv1.weight[:,s2_bands_to_keep[band],:,:]
    
    model.conv1 = new_layer
    return model
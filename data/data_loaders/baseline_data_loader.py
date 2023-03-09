import torch
import torchvision
import torchgeo


class BaselineDataLoader():
    
    def __init__(self, config):
        self.config = config
        self.train_loader = torch.utils.data.DataLoader()
        self.test_loader = torch.utils.data.DataLoader()




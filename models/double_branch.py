# @MDC, MARBEC, 2023

import torch


class DoubleBranchCNN( torch.nn.Module ):
    """Custom Double Branch Network, that takes two separate CNNs
       as inputs and performs a late fusion, combining them 
       at their last fully-connected layer.

    Args:
        b1 (torch.nn.Module): first branch cnn
        b2 (torch.nn.Module): second branch cnn
    """

    def __init__(self, b1, b2 , output_features : int):
        super(DoubleBranchCNN, self).__init__()
        self.b1 = b1
        self.b2 = b2
        b1_features = b1.fc.in_features
        b2_features = b2.fc.in_features
        self.b2.fc = torch.nn.Identity()
        self.b1.fc = torch.nn.Identity()
        self.fc = torch.nn.Linear(b1_features+b2_features, output_features)
        
    
    def forward(self, x1, x2):
        x1= self.b1(x1)
        x2= self.b2(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
    
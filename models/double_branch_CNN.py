# @MDC, MARBEC, 2023

import torch


class DoubleBranchCNN( torch.nn.Module ):
    """Custom Double Branch Network, that takes two separate CNNs
       as inputs and performs a late fusion, combining them 
       at their last fully-connected layer.

    Args:
        ms (torch.nn.Module): first branch cnn
        nl (torch.nn.Module): second branch cnn
    """

    def __init__(self, ms, nl , output_features : int):
        super(DoubleBranchCNN, self).__init__()
        self.ms = ms
        self.nl = nl
        ms_features = ms.fc.in_features
        nl_features = nl.fc.in_features
        self.nl.fc = torch.nn.Identity()
        self.ms.fc = torch.nn.Identity()
        self.fc = torch.nn.Linear(ms_features+nl_features, output_features)
        # self.fc = torch.nn.Linear(2, output_features)

        
    def forward(self, x1, x2):
        x1= self.ms(x1)
        x2= self.nl(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
    
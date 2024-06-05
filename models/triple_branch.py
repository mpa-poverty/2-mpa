# @MDC, MARBEC, 2023

import torch


class TripleBranch(torch.nn.Module):
    """Custom Triple Branch Network, that takes three separate CNNs
       as inputs and performs a late fusion, combining them 
       at their last fully-connected layer.

    Args:
        ms (torch.nn.Module): first branch cnn
        nl (torch.nn.Module): second branch cnn
        ts (torch.nn.Module): third branch cnn
    """

    def __init__(self, ms, nl, ts, output_features: int):
        super(TripleBranch, self).__init__()
        self.ms = ms
        self.nl = nl
        self.ts = ts

        total_features = ms.fc.in_features + nl.fc.in_features + ts.fc.in_features
        self.ms.fc = torch.nn.Identity()

        self.nl.fc = torch.nn.Identity()
        self.ts.fc = torch.nn.Identity()

        self.fc = torch.nn.Linear(total_features, output_features)

    def forward(self, x1, x2, x3):
        x1 = 0.9 * self.ms(x1)
        x2 = self.nl(x2)
        x3 = self.ts(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x

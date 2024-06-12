# @MDC, MARBEC, 2023

import torch


class TripleBranch(torch.nn.Module):
    """Custom Double Branch Network, that takes two separate CNNs
       as inputs and performs a late fusion, combining them
       at their last fully-connected layer.

    Args:
        ms (torch.nn.Module): first branch cnn
        nl (torch.nn.Module): second branch cnn
    """

    def __init__(self, branch_1, branch_2, branch_3, output_features: int, with_vit: bool = False):
        super(TripleBranch, self).__init__()
        self.branch_1 = branch_1
        self.branch_2 = branch_2
        self.branch_3 = branch_3

        if not with_vit:
            total_features = branch_1.fc.in_features + branch_2.fc.in_features + branch_3.fc.in_features
            self.branch_1.fc = torch.nn.Identity()
        else:
            total_features = branch_1.head.in_features + branch_2.fc.in_features + branch_3.fc.in_features
            self.branch_1.head = torch.nn.Identity()

        self.branch_2.fc = torch.nn.Identity()
        self.branch_3.fc = torch.nn.Identity()

        self.fc = torch.nn.Linear(total_features, 1)

    def forward(self, x1, x2, x3):
        x1 = 0.9 * self.branch_1(x1)
        x2 = self.branch_2(x2)
        x3 = self.branch_3(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x
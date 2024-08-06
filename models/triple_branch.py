# MODELS/RESULTS/TRIPLE_BRANCH.PY
#
# Description: Custom Triple Branch Network, that takes three separate CNNs
#
# @MDC, MARBEC, 2023

import torch


class TripleBranch(torch.nn.Module):
    """Custom Double Branch Network, that takes two separate CNNs
       as inputs and performs a late fusion, combining them
       at their last fully-connected layer.

    Args:
        branch_1 (torch.nn.Module): first branch cnn
        branch_2 (torch.nn.Module): second branch cnn
        branch_3 (torch.nn.Module): third branch fcn
        output_features (int): number of output features

    """

    def __init__(self, branch_1, branch_2, branch_3, output_features=1):
        super(TripleBranch, self).__init__()
        self.branch_1 = branch_1
        self.branch_2 = branch_2
        self.branch_3 = branch_3

        total_features = branch_1.fc.in_features + branch_2.fc.in_features + branch_3.fc.in_features
        self.branch_1.fc = torch.nn.Identity()

        self.branch_2.fc = torch.nn.Identity()
        self.branch_3.fc = torch.nn.Identity()

        self.fc = torch.nn.Linear(total_features, output_features)

    def forward(self, x1, x2, x3):
        x1 = 0.9 * self.branch_1(x1)
        x2 = self.branch_2(x2)
        x3 = self.branch_3(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc(x)
        return x

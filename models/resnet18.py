from ..utils import transfer_learning as tl


def resnet18_init_model(weights=ResNet18_Weights.DEFAULT):
    '''imports the resnet18 with pretrained weights by default.
       other weights can be set via the _weights_ parameter.'''
    resnet18 = torch.models.resnet18(weights=weights)
    return resnet18

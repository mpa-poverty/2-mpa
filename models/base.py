import logging
from abc import abstractmethod

class BaseModel:
    """
    This base class will contain the base functions to be overloaded by Model.
    """
    @abstractmethod
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Model")

    @abstractmethod
    def forward(self,x):
        """ 
        Main forward loop
        """
        raise NotImplementedError
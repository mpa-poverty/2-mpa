import logging


class BaseModel:
    """
    This base class will contain the base functions to be overloaded by Model.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Model")

    def forward(self,x):
        """ 
        Main forward loop
        """
        raise NotImplementedError
from abc import ABC, abstractmethod

class StyleTransferer(ABC):
    """Abstract class of a style transfer algorithm.
    
    """
    
    def raise_default_error(self):
        raise NotImplementedError("This needs to be implemented by child class.")

    @abstractmethod
    def transfer_single_style(self, style_variable, content_variable):
        self.raise_default_error()

    @abstractmethod
    def transfer_tensor_to_tensor(self, style_tensor, content_tensor):
        self.raise_default_error()
    

class AdaIN(StyleTransferer):
    """AdaIN style transfer method
    
    """
    def __init__(self, args):
        self.args = args
        
    
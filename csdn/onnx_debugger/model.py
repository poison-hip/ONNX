import torch
from types import MethodType 
from debugger import Debugger

debugger = Debugger()

class Model(torch.nn.Module): 
 
    def __init__(self): 
        super().__init__() 
        self.convs1 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1)) 
        self.convs2 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1)) 
        self.convs3 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1)) 
        self.convs4 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1), 
                                          torch.nn.Conv2d(3, 3, 3, 1, 1)) 
 
    def forward(self, x): 
        x = self.convs1(x) 
        x = self.convs2(x) 
        x = self.convs3(x) 
        x = self.convs4(x) 
        return x 
    
def new_forward(self, x): 
    x = self.convs1(x) 
    x = debugger.debug(x, 'x_0') 
    x = self.convs2(x) 
    x = debugger.debug(x, 'x_1') 
    x = self.convs3(x) 
    x = debugger.debug(x, 'x_2') 
    x = self.convs4(x) 
    x = debugger.debug(x, 'x_3') 
    return x 

torch_model = Model() 
torch_model.forward = MethodType(new_forward, torch_model)

dummy_input = torch.randn(1, 3, 10, 10) 
torch.onnx.export(torch_model, dummy_input, 'before_debug.onnx', input_names=['input']) 

debugger.extract_debug_model('before_debug.onnx', 'after_debug.onnx') 
debugger.run_debug_model({'input':dummy_input.numpy()}, 'after_debug.onnx') 
debugger.print_debug_result() 

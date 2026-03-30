import torch
import numpy as np
from torch.onnx.symbolic_registry import register_op
import onnxruntime 

def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input)

register_op('asinh', asinh_symbolic, '', 13)

class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, x): 
        return torch.asinh(x) 

model = Model() 
input = torch.rand(1, 3, 10, 10)
torch_output = model(input).detach().numpy() 
torch.onnx.export(model, input, 'asinh.onnx') 

sess = onnxruntime.InferenceSession('asinh.onnx')
ort_output = sess.run(None, {'onnx::Asinh_0':input.numpy()})[0]

assert np.allclose(torch_output, ort_output)
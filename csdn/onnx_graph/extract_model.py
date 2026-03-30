import torch 
import onnx
 
class Model(torch.nn.Module): 
 
    def __init__(self): 
        super().__init__() 
        self.convs1 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3)) 
        self.convs2 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3)) 
        self.convs3 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3)) 
        self.convs4 = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3), 
                                          torch.nn.Conv2d(3, 3, 3)) 
    def forward(self, x): 
        x = self.convs1(x) 
        x1 = self.convs2(x) 
        x2 = self.convs3(x) 
        x = x1 + x2 
        x = self.convs4(x) 
        return x 
 
model = Model() 
input = torch.randn(1, 3, 20, 20) 
 
torch.onnx.export(model, input, 'whole_model.onnx',opset_version=16) 

# 提取部分的onnx节点来获取部分信息作为我的输入
# 截取模型
onnx.utils.extract_model('whole_model.onnx', 'partial_model.onnx', ['22'], ['28'])
model = onnx.load('partial_model.onnx') 
# 增加模型输出
onnx.utils.extract_model('whole_model.onnx', 'submodel_1.onnx', ['22'], ['27', '31']) 
# 增加冗余输入
onnx.utils.extract_model('whole_model.onnx', 'submodel_2.onnx', ['22', 'input.1'], ['28']) 
# 输出冗余中间节点的值
onnx.utils.extract_model('whole_model.onnx', 'more_output_model.onnx', ['input.1'], ['31', '23', '25', '27'])
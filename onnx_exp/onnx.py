import torch.onnx
import torch.nn as nn
from torch.nn.functional import interpolate 

class NewInterpolate(torch.autograd.Function):

    @ staticmethod
    def symbolic(g, input, scales):
        return g.op("Resize",
                    input,
                    g.op("Constant",
                         value_t=torch.tensor([], dtype=torch.float32)),
                    scales,
                    coordinate_transformation_mode_s="pytorch_half_pixel",
                    cubic_coeff_a_f=-0.75,
                    mode_s='cubic',
                    nearest_mode_s='floor')

    @ staticmethod
    def forward(ctx, input, scales):
        scales = scales.tolist()[-2:]
        return interpolate(input,
                           scale_factor=scales,
                           mode='bicubic',
                           align_corners=False)

class SuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 原始固定值方法
        # self.img_upsampler = nn.Upsample(
        #     scale_factor=upscale_factor,
        #     mode="bicubic",
        #     align_corners=False
        # )

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x, upscale_factor):

        # x = self.img_upsampler(x)
        # x = interpolate(x, 
        #                 scale_factor=upscale_factor.item(), 
        #                 mode='bicubic', 
        #                 align_corners=False) 

        x = NewInterpolate.apply(x, upscale_factor)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = SuperResolutionNet()
ckpt = torch.load("/home/poison/桌面/ONNX/srcnn.pth", map_location="cpu")

if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

new_state_dict = {}
for old_key, value in state_dict.items():
    new_key = ".".join(old_key.split(".")[1:])
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict, strict=True)
model.eval()

factor = torch.tensor([1, 1, 3, 3], dtype=torch.float)
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    torch.onnx.export(
        model,
        (x,factor),
        "srcnn.onnx",
        opset_version=11,
        input_names=['input','factor'],
        output_names=['output'],
    )



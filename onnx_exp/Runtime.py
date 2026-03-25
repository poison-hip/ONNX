import torch
import numpy as np
import cv2
import onnxruntime

# input_img = np.random.randn(1, 3, 256, 256).astype(np.float32)
input_img = cv2.imread("/home/poison/桌面/ONNX/img.png")
input_img = input_img.astype(np.float32)
input_img = np.transpose(input_img, (2, 0, 1))
input_img = np.expand_dims(input_img, 0)
input_factor = np.array([1, 1, 5, 5], dtype=np.float32) 

ort_session = onnxruntime.InferenceSession("/home/poison/桌面/ONNX/srcnn.onnx")
ort_input = {'input' : input_img, 'factor' : input_factor}
ort_output = ort_session.run(['output'], ort_input)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1,2,0]).astype(np.uint8)
cv2.imwrite("factor_img5*5.png", ort_output)
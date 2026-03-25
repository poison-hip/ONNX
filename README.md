# ONNX Learning Notes

This directory records one afternoon of focused ONNX export experiments with PyTorch.

## What I Learned

### 1. Basic ONNX Export and ONNX Runtime Inference

From [onnx_exp/onnx.py](/home/poison/桌面/ONNX/onnx_exp/onnx.py) and [onnx_exp/Runtime.py](/home/poison/桌面/ONNX/onnx_exp/Runtime.py):

- `torch.onnx.export(model, args, path)` exports a PyTorch model to ONNX.
- `onnxruntime.InferenceSession(path)` loads the exported ONNX model for inference.
- ONNX Runtime input must be `numpy.ndarray`, not `torch.Tensor`.
- The input feed keys in `sess.run()` must match the ONNX graph input names, not Python variable names.
- `np.allclose(torch_output, ort_output)` is a practical way to verify PyTorch and ONNX Runtime outputs are numerically close.

### 2. PyTorch `interpolate` vs ONNX `Resize`

This was the main conceptual point.

- In PyTorch, `F.interpolate(..., scale_factor=3)` is valid for 2D image upsampling.
- In ONNX, the corresponding `Resize` op expects `scales` for all input dimensions.
- For `NCHW` input, a 3x resize is represented as `[1, 1, 3, 3]`.
- So a single scalar `3` is acceptable in PyTorch high-level API, but not sufficient as ONNX `Resize` scales input.

### 3. Custom Symbolic for Export

From [onnx_exp/onnx.py](/home/poison/桌面/ONNX/onnx_exp/onnx.py):

- A custom `torch.autograd.Function` can be used to bridge PyTorch behavior and ONNX graph generation.
- `forward()` runs during tracing/export and must produce a real tensor output.
- `symbolic()` defines how this operation appears in the ONNX graph.
- In the custom `NewInterpolate` example:
  - `symbolic()` emits an ONNX `Resize` node.
  - `forward()` converts ONNX-style `scales` like `[1, 1, 3, 3]` into PyTorch-style spatial scale factors like `[3, 3]`.

### 4. Why `tolist()` Warns

- `scales.tolist()` converts a tensor into a Python list.
- During export, this breaks the tensor data flow from PyTorch tracing's point of view.
- PyTorch therefore warns that the exported graph may treat the value as constant.
- This is usually acceptable for fixed-scale export.
- This is not ideal if the goal is a truly dynamic scale input.

### 5. Trace vs Script

From [onnx_trans/test.py](/home/poison/桌面/ONNX/onnx_trans/test.py):

- `torch.jit.trace(model, example_input)` records one execution path.
- `torch.jit.script(model)` captures model logic more explicitly.
- Different models or control-flow patterns can export differently under trace and script.

### 6. Dynamic Axes

From [onnx_trans/test_dynamic.py](/home/poison/桌面/ONNX/onnx_trans/test_dynamic.py):

- `dynamic_axes` controls which dimensions are dynamic in the exported ONNX model.
- Without `dynamic_axes`, the exported model usually has fixed input/output shapes.
- Dynamic batch and dynamic spatial dimensions are configured separately.
- Export behavior can be customized with `torch.onnx.is_in_onnx_export()`.

### 7. Registering Symbolics for Unsupported Operators

From [onnx_register/aten.py](/home/poison/桌面/ONNX/onnx_register/aten.py):

- If an operator is unsupported by default, a custom symbolic can be registered.
- Example: registering `asinh` to map to ONNX `Asinh`.
- The registered symbolic must match the opset version used during export.
- If symbolic is registered for opset 9 but export uses opset 13, registration will not take effect.

### 8. Registering Third-Party / Custom Ops

From [onnx_register/torch_scripts.py](/home/poison/桌面/ONNX/onnx_register/torch_scripts.py):

- Third-party operators such as `torchvision::deform_conv2d` can also be given custom symbolic mappings.
- `register_custom_op_symbolic()` is used for this case.
- `parse_args` helps define how operator arguments are parsed by the symbolic function.

## Practical Pitfalls I Hit

- Passing `(x, 3)` into `torch.onnx.export()` did not behave like directly calling `F.interpolate(..., scale_factor=3)`.
- ONNX graph input names can differ from the Python variable names used in code.
- Misspelling ONNX op attributes causes checker failures immediately.
- A registered symbolic does not help unless the export opset matches it.
- Converting tensors to Python scalars or lists during export often removes dynamic behavior.

## Final Takeaway

The key lesson is that ONNX export is not just "saving a model".
It is translating PyTorch execution semantics into ONNX graph semantics.

The most important things I learned are:

- how PyTorch ops map to ONNX ops,
- how inputs are represented differently across the two systems,
- how to verify exported models with ONNX Runtime,
- and how to patch unsupported export paths by writing custom symbolic functions.

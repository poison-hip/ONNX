from typing import Dict, Optional, Sequence, Union

import torch
import tensorrt as trt


TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
}


class TRTWrapper(torch.nn.Module):
    def __init__(
        self,
        engine: Union[str, trt.ICudaEngine],
        output_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            with open(self.engine, mode="rb") as f:
                engine_bytes = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.context = self.engine.create_execution_context()
        self._tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self._input_names = [
            name
            for name in self._tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self._output_names = list(output_names) if output_names is not None else [
            name
            for name in self._tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

    def _get_torch_dtype(self, tensor_name: str) -> torch.dtype:
        trt_dtype = self.engine.get_tensor_dtype(tensor_name)
        if trt_dtype not in TRT_TO_TORCH_DTYPE:
            raise TypeError(f"Unsupported TensorRT dtype for {tensor_name}: {trt_dtype}")
        return TRT_TO_TORCH_DTYPE[trt_dtype]

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        missing_inputs = set(self._input_names) - set(inputs.keys())
        if missing_inputs:
            raise KeyError(f"Missing TensorRT inputs: {sorted(missing_inputs)}")

        profile_id = 0
        active_inputs: Dict[str, torch.Tensor] = {}
        for input_name in self._input_names:
            input_tensor = inputs[input_name]
            profile = self.engine.get_tensor_profile_shape(input_name, profile_id)
            if input_tensor.dim() != len(profile[0]):
                raise ValueError(
                    f"Input dim mismatch for {input_name}: expected {len(profile[0])}, "
                    f"got {input_tensor.dim()}"
                )

            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape, profile[2]):
                if not s_min <= s_input <= s_max:
                    raise ValueError(
                        f"Input shape for {input_name} should be between {profile[0]} "
                        f"and {profile[2]}, but got {tuple(input_tensor.shape)}"
                    )

            if input_tensor.device.type != "cuda":
                raise TypeError(f"Input {input_name} must be on CUDA, got {input_tensor.device}")

            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()

            self.context.set_input_shape(input_name, tuple(input_tensor.shape))
            self.context.set_tensor_address(input_name, input_tensor.data_ptr())
            active_inputs[input_name] = input_tensor

        outputs: Dict[str, torch.Tensor] = {}
        for output_name in self._output_names:
            shape = tuple(self.context.get_tensor_shape(output_name))
            dtype = self._get_torch_dtype(output_name)
            output = torch.empty(size=shape, dtype=dtype, device=torch.device("cuda"))
            self.context.set_tensor_address(output_name, output.data_ptr())
            outputs[output_name] = output

        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execution failed")

        return outputs


model = TRTWrapper("model.engine", ["output"])
output = model({"input": torch.randn(1, 3, 224, 224, device="cuda")})
print(output)

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch

from polygraphy import cuda as pcuda
from polygraphy.backend.trt import util as trt_util


def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


torch_to_numpy_dtype_dict = {torch.float16: np.float16,
                             torch.float32: np.float32}
numpy_to_torch_dtype_dict = {np.float16: torch.float16,
                             np.float32: torch.float32,
                             np.int32: torch.int32}


class TensorRTModel:
    def __init__(
        self,
        trt_engine_path,
        **kwargs,
    ):
        cuda.init()
        stream = pcuda.Stream() 
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        trt_runtime = trt.Runtime(TRT_LOGGER)
        engine = load_engine(trt_runtime, trt_engine_path)
        context = engine.create_execution_context()
        self.stream = stream
        self.context = context
        self.engine = engine

        if 'shape_list' in kwargs.keys():
            shape_list = kwargs['shape_list']
        else:
            shape_list = None
        self.allocate_buffers(shape_list=shape_list)

    def allocate_buffers(self, device="cuda", shape_list=None):
        self.tensors = {}
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            shape = self.engine.get_binding_shape(binding)
            print(f"get_binding_shape: {shape}")
            if shape_list is not None and idx < len(shape_list):
                shape = shape_list[idx]
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            print(f"idx: {idx}, binding: {binding}, dtype: {dtype}, shape: {shape}")
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor
    
    
    def __call__(self, **kwargs):
        context = self.context
        stream = self.stream

        feed_dict = kwargs
        for name, buf in feed_dict.items():
            # print(f"name: {name}, buf: {buf}")
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        context.execute_async_v3(stream_handle=stream.ptr)

        stream.synchronize()

        return self.tensors

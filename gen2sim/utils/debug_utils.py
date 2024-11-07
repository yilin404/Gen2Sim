import torch
import numpy as np

def debug_shape_info(data: torch.Tensor | np.ndarray | int, name: str):
    if isinstance(data, torch.Tensor):
        print(f"{name}.size is: ", data.size())
    elif isinstance(data, np.ndarray):
        print(f"{name}.shape is: ", data.shape)
    elif isinstance(data, int):
        print(f"{name} is: ", data)
    else:
        raise NotImplementedError()

def debug_value_info(data: torch.Tensor | np.ndarray, name: str):
    if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
        print(f"{name} is: ", data)
    else:
        raise NotImplementedError()
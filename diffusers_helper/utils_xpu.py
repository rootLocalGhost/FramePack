## File: diffusers_helper/utils_xpu.py
from diffusers_helper.utils import *
import torch

def print_free_mem():
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()
        try:
            free_mem, total_mem = torch.xpu.mem_get_info(0)
            free_mem_mb = free_mem / (1024 ** 2)
            total_mem_mb = total_mem / (1024 ** 2)
            print(f"Free memory: {free_mem_mb:.2f} MB")
            print(f"Total memory: {total_mem_mb:.2f} MB")
        except:
            print("Could not retrieve exact XPU memory info via mem_get_info")
            print(f"Reserved: {torch.xpu.memory_reserved(0) / (1024**2):.2f} MB")
    else:
        print("XPU not available.")
    return
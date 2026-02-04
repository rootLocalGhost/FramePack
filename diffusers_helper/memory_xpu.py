## File: diffusers_helper/memory_xpu.py
import torch

cpu = torch.device('cpu')

# Detect XPU
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    gpu = torch.device('xpu')
    print(f"XPU detected: {torch.xpu.get_device_name(0)}")
else:
    print("WARNING: XPU not detected. Fallback to CPU (This will be extremely slow).")
    gpu = torch.device('cpu')

gpu_complete_modules = []

class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })
        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return

def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return
    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return

def get_xpu_free_memory_gb(device=None):
    if device is None:
        device = gpu
    
    if device.type == 'cpu':
        return 32.0 # Fake return for CPU fallback

    # PyTorch XPU implementation of mem_get_info or equivalent
    try:
        # Try standard mem_get_info if implemented for XPU in your version
        bytes_free, bytes_total = torch.xpu.mem_get_info(device)
        return bytes_free / (1024 ** 3)
    except:
        # Fallback calculation using stats
        props = torch.xpu.get_device_properties(device)
        total_memory = props.total_memory
        reserved_memory = torch.xpu.memory_reserved(device)
        # This is an approximation, as reserved doesn't mean used, but safe for swap logic
        free_approx = total_memory - reserved_memory
        return free_approx / (1024 ** 3)

def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')
    for m in model.modules():
        # Check memory before moving
        if target_device.type == 'xpu':
            if get_xpu_free_memory_gb(target_device) <= preserved_memory_gb:
                torch.xpu.empty_cache()
                return
        
        if hasattr(m, 'weight'):
            m.to(device=target_device)
    model.to(device=target_device)
    if target_device.type == 'xpu':
        torch.xpu.empty_cache()
    return

def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')
    for m in model.modules():
        if target_device.type == 'xpu':
            if get_xpu_free_memory_gb(target_device) >= preserved_memory_gb:
                torch.xpu.empty_cache()
                return
        
        if hasattr(m, 'weight'):
            m.to(device=cpu)
    model.to(device=cpu)
    if target_device.type == 'xpu':
        torch.xpu.empty_cache()
    return

def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')
    gpu_complete_modules.clear()
    if gpu.type == 'xpu':
        torch.xpu.empty_cache()
    return

def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()
    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')
    gpu_complete_modules.append(model)
    return
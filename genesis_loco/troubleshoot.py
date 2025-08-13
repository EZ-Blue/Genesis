# In your conda/venv that runs Genesis:
import torch, os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda.is_available ->", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device 0:", torch.cuda.get_device_name(0))


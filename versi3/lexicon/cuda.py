import torch
print("CUDA available:", torch.cuda.is_available())
print("torch.cuda.version:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"CUDA:{i} -> {torch.cuda.get_device_name(i)}")

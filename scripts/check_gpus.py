import os, torch, subprocess

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(i, props.name, f"{props.total_memory/1024**3:.1f} GB")
print("torch.cuda.current_device():", torch.cuda.current_device())

# optional: nvidia-smi
try:
    print("\nNVIDIA-SMI (short):")
    print(subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,name,memory.free","--format=csv,noheader,nounits"]
    ).decode())
except Exception as e:
    print("nvidia-smi not available:", e)

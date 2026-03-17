import torch

print("=======================================")
print("          CUDA / GPU TEST              ")
print("=======================================")

# 1. Check PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# 2. Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available:  {cuda_available}")

if cuda_available:
    # 3. Get detailed GPU info
    print(f"Device Count:    {torch.cuda.device_count()}")
    print(f"Current Device:  {torch.cuda.current_device()}")
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version:    {torch.version.cuda}")
    print("\nSUCCESS: PyCharm is properly utilizing the Jetson GPU!")
else:
    print("\nFAILURE: CUDA is NOT available.")
    print("PyTorch is falling back to the CPU, which will be incredibly slow.")

print("\n=======================================")
print("          YOLO ENGINE TEST             ")
print("=======================================")
try:
    import ultralytics
    # This runs YOLO's built-in system check
    ultralytics.checks()
except ImportError:
    print("Ultralytics YOLO is not installed in this environment.")
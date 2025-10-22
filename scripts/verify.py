import torch

print("\n" + "=" * 40)
print("✅ Project Initialization Successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
print("=" * 40)
print(f"\nTo activate this environment, run:")

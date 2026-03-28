"""
Quick test script to verify depth-only training setup
"""
import sys
import os

# Test 1: Check imports
print("=" * 50)
print("Test 1: Checking imports...")
print("=" * 50)

try:
    from models import DepthOnlyResNet34U_f, DepthOnlyResNet34U_f_EMAEncoderOnly
    print("✓ Models imported successfully")
except ImportError as e:
    print(f"✗ Model import failed: {e}")
    sys.exit(1)

try:
    from data import DepthOnlyDataset
    print("✓ DepthOnlyDataset imported successfully")
except ImportError as e:
    print(f"✗ Dataset import failed: {e}")
    sys.exit(1)

try:
    from utils.build_dataset_depth_only import build_dataset_depth_only, DepthOnlyImageFolderDataset
    print("✓ Dataset builder imported successfully")
except ImportError as e:
    print(f"✗ Dataset builder import failed: {e}")
    sys.exit(1)

# Test 2: Check model instantiation
print("\n" + "=" * 50)
print("Test 2: Model instantiation...")
print("=" * 50)

try:
    import torch
    stu_model = DepthOnlyResNet34U_f(num_classes=1)
    tea_model = DepthOnlyResNet34U_f_EMAEncoderOnly(num_classes=1)
    print(f"✓ Student model created: {stu_model.__class__.__name__}")
    print(f"✓ Teacher model created: {tea_model.__class__.__name__}")
except Exception as e:
    print(f"✗ Model instantiation failed: {e}")
    sys.exit(1)

# Test 3: Forward pass
print("\n" + "=" * 50)
print("Test 3: Forward pass test...")
print("=" * 50)

try:
    # Simulate depth input (B, 1, H, W)
    depth_input = torch.randn(2, 1, 320, 320)
    
    stu_output = stu_model(depth_input)
    print(f"✓ Student forward pass: input {depth_input.shape} → output {stu_output.shape}")
    
    tea_output = tea_model(depth_input)
    print(f"✓ Teacher forward pass: input {depth_input.shape} → output {tea_output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 4: Parameter count
print("\n" + "=" * 50)
print("Test 4: Model info...")
print("=" * 50)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

stu_params = count_params(stu_model)
tea_params = count_params(tea_model)

print(f"Student model params: {stu_params:,}")
print(f"Teacher model params: {tea_params:,}")

# Test 5: Config file
print("\n" + "=" * 50)
print("Test 5: Config file...")
print("=" * 50)

config_path = 'cfg/depthOnly.yaml'
if os.path.exists(config_path):
    print(f"✓ Config file exists: {config_path}")
else:
    print(f"✗ Config file not found: {config_path}")
    sys.exit(1)

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)
print("\nNext steps:")
print("1. Prepare your depth data in .npy format")
print("2. Update cfg/depthOnly.yaml with your dataset paths")
print("3. Run: python depthOnlyTrain.py --config cfg/depthOnly.yaml")

#!/usr/bin/env python3
"""
Test script for Monocular Depth Estimation Implementation
This script tests all components to ensure they work correctly.
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.monodepth import MonodepthModel
from models.encoder import ResNetEncoder
from models.decoder import DepthDecoder
from utils.dataset import MonocularDepthDataset
from utils.losses import DepthLoss, BerHuLoss, SSIMLoss
from utils.metrics import compute_depth_metrics, visualize_depth

def test_encoder():
    """Test the ResNet encoder."""
    print("Testing ResNet Encoder...")
    
    # Test different ResNet depths
    for num_layers in [18, 34, 50]:
        encoder = ResNetEncoder(num_layers=num_layers, pretrained=False)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 192, 640)
        features = encoder(dummy_input)
        
        # Check output
        assert len(features) == 5, f"Expected 5 feature maps, got {len(features)}"
        assert features[0].shape[1] == encoder.num_ch_enc[0], f"Feature channels mismatch"
        
        print(f"  ResNet-{num_layers}: ✓")
    
    print("Encoder tests passed!")

def test_decoder():
    """Test the depth decoder."""
    print("Testing Depth Decoder...")
    
    # Test with different encoder configurations
    num_ch_enc = [64, 64, 128, 256, 512]
    decoder = DepthDecoder(num_ch_enc=num_ch_enc, scales=range(4), use_skips=True)
    
    # Test forward pass
    dummy_features = [torch.randn(2, ch, 192 // (2**i), 640 // (2**i)) 
                     for i, ch in enumerate(num_ch_enc)]
    outputs = decoder(dummy_features)
    
    # Check outputs
    expected_scales = ['disp_0', 'disp_1', 'disp_2', 'disp_3']
    for scale in expected_scales:
        assert scale in outputs, f"Missing output scale: {scale}"
        assert outputs[scale].shape[1] == 1, f"Depth should have 1 channel"
    
    print("Decoder tests passed!")

def test_complete_model():
    """Test the complete monodepth model."""
    print("Testing Complete Model...")
    
    # Test different configurations
    configs = [
        {'num_layers': 18, 'scales': range(4), 'use_skips': True},
        {'num_layers': 34, 'scales': range(3), 'use_skips': False},
    ]
    
    for config in configs:
        model = MonodepthModel(**config, pretrained=False)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 192, 640)
        outputs = model(dummy_input)
        
        # Check outputs
        expected_scales = [f'disp_{i}' for i in config['scales']]
        for scale in expected_scales:
            assert scale in outputs, f"Missing output scale: {scale}"
        
        print(f"  Config {config}: ✓")
    
    print("Complete model tests passed!")

def test_losses():
    """Test the loss functions."""
    print("Testing Loss Functions...")
    
    # Test BerHu loss
    berhu_loss = BerHuLoss()
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    loss = berhu_loss(pred, target)
    assert loss.item() >= 0, "Loss should be non-negative"
    
    # Test SSIM loss
    ssim_loss = SSIMLoss()
    loss = ssim_loss(pred, target)
    assert loss.item() >= 0, "SSIM loss should be non-negative"
    
    # Test combined depth loss
    depth_loss = DepthLoss()
    pred_depths = {
        'disp_0': torch.randn(2, 1, 64, 64),
        'disp_1': torch.randn(2, 1, 32, 32),
        'disp_2': torch.randn(2, 1, 16, 16),
    }
    target_depth = torch.randn(2, 1, 64, 64)
    input_image = torch.randn(2, 3, 64, 64)
    
    loss_dict = depth_loss(pred_depths, target_depth, input_image)
    assert 'total' in loss_dict, "Loss dict should contain 'total'"
    assert loss_dict['total'].item() >= 0, "Total loss should be non-negative"
    
    print("Loss function tests passed!")

def test_metrics():
    """Test the evaluation metrics."""
    print("Testing Evaluation Metrics...")
    
    # Create dummy predictions and targets
    pred = torch.randn(1, 1, 64, 64)
    target = torch.randn(1, 1, 64, 64)
    
    # Test depth metrics
    metrics = compute_depth_metrics(pred, target)
    expected_keys = ['rmse', 'mae', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], (int, float)), f"Metric {key} should be numeric"
    
    print("Metrics tests passed!")

def test_dataset():
    """Test the dataset classes."""
    print("Testing Dataset Classes...")
    
    # Create a temporary dataset structure
    os.makedirs("test_data/train/images", exist_ok=True)
    os.makedirs("test_data/train/depth", exist_ok=True)
    os.makedirs("test_data/val/images", exist_ok=True)
    
    # Create dummy images
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_depth = np.random.rand(480, 640).astype(np.float32)
    
    cv2.imwrite("test_data/train/images/test1.jpg", dummy_image)
    cv2.imwrite("test_data/train/images/test2.jpg", dummy_image)
    np.save("test_data/train/depth/test1.npy", dummy_depth)
    np.save("test_data/train/depth/test2.npy", dummy_depth)
    
    # Test dataset
    try:
        dataset = MonocularDepthDataset(
            data_path="test_data",
            split='train',
            load_depth=True
        )
        
        # Test dataset length
        assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
        
        # Test sample loading
        sample = dataset[0]
        assert 'image' in sample, "Sample should contain 'image'"
        assert 'depth' in sample, "Sample should contain 'depth'"
        assert sample['image'].shape[0] == 3, "Image should have 3 channels"
        
        print("Dataset tests passed!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")
        raise
    
    finally:
        # Clean up
        import shutil
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")

def test_inference_pipeline():
    """Test the complete inference pipeline."""
    print("Testing Inference Pipeline...")
    
    # Create model
    model = MonodepthModel(num_layers=18, pretrained=False)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 192, 640)
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
    
    # Test depth extraction
    depth = model.get_depth(dummy_input, scale=0)
    assert depth.shape[1] == 1, "Depth should have 1 channel"
    
    print("Inference pipeline tests passed!")

def test_visualization():
    """Test visualization functions."""
    print("Testing Visualization Functions...")
    
    # Create dummy depth map
    depth_map = torch.randn(1, 1, 64, 64)
    
    # Test visualization
    try:
        fig = visualize_depth(depth_map)
        assert fig is not None, "Visualization should return a figure"
        plt.close(fig)
        print("Visualization tests passed!")
    except Exception as e:
        print(f"Visualization test failed: {e}")

def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Monocular Depth Estimation Tests")
    print("=" * 50)
    
    tests = [
        test_encoder,
        test_decoder,
        test_complete_model,
        test_losses,
        test_metrics,
        test_dataset,
        test_inference_pipeline,
        test_visualization,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print("=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1) 
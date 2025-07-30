"""
Flexible weight loading utility for eqxvision models.
Uses name-based parameter mapping instead of positional alignment for better robustness.
Similar to HuggingFace -> Flax conversion approaches.
"""

import logging
import os
import sys
import warnings
from typing import Dict, Any, Optional
import re

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

try:
    import torch
except ImportError:
    warnings.warn("PyTorch is required for loading pre-trained weights.")

_TEMP_DIR = "/tmp/.eqx"


def get_nested_attr(obj, path: str):
    """Get nested attribute using dot notation and array indexing"""
    parts = path.split('.')
    current = obj
    
    for part in parts:
        if '[' in part and ']' in part:
            # Handle array indexing like layers[0]
            attr_name, indices = part.split('[', 1)
            indices = indices.rstrip(']')
            current = getattr(current, attr_name)
            
            # Handle multiple indices like [0][1]
            for idx in indices.split(']['):
                current = current[int(idx)]
        else:
            current = getattr(current, part)
    
    return current


def create_resnet_parameter_mapping():
    """
    Create explicit mapping between PyTorch and Equinox parameter names for ResNet models.
    Format: {pytorch_name: equinox_path}
    """
    mapping = {}
    
    # Root level parameters
    mapping["conv1.weight"] = "conv1.weight"
    mapping["bn1.weight"] = "bn1.weight"
    mapping["bn1.bias"] = "bn1.bias" 
    mapping["fc.weight"] = "fc.weight"
    mapping["fc.bias"] = "fc.bias"
    
    # Layer mappings - ResNet architectures have different block counts
    # ResNet50: [3, 4, 6, 3], ResNet101: [3, 4, 23, 3], etc.
    # We'll generate mappings for the maximum needed (ResNet152: [3, 8, 36, 3])
    layer_configs = [(1, 8), (2, 8), (3, 36), (4, 8)]  # Generous upper bounds
    
    for layer_idx, max_blocks in layer_configs:
        for block_idx in range(max_blocks):
            # PyTorch uses dot notation: layer1.0.conv1.weight
            # Equinox uses: layer1.layers[0].conv1.weight
            pt_prefix = f"layer{layer_idx}.{block_idx}"
            eqx_prefix = f"layer{layer_idx}.layers[{block_idx}]"
            
            # Convolution and BatchNorm parameters
            for conv_idx in range(1, 4):  # conv1, conv2, conv3
                # Convolution weights
                mapping[f"{pt_prefix}.conv{conv_idx}.weight"] = f"{eqx_prefix}.conv{conv_idx}.weight"
                
                # BatchNorm trainable parameters
                mapping[f"{pt_prefix}.bn{conv_idx}.weight"] = f"{eqx_prefix}.bn{conv_idx}.weight"
                mapping[f"{pt_prefix}.bn{conv_idx}.bias"] = f"{eqx_prefix}.bn{conv_idx}.bias"
            
            # Downsample layers (when present)
            mapping[f"{pt_prefix}.downsample.0.weight"] = f"{eqx_prefix}.downsample.layers[0].weight"
            mapping[f"{pt_prefix}.downsample.1.weight"] = f"{eqx_prefix}.downsample.layers[1].weight"
            mapping[f"{pt_prefix}.downsample.1.bias"] = f"{eqx_prefix}.downsample.layers[1].bias"
    
    return mapping


def create_resnet_bn_stats_mapping():
    """
    Create mapping for BatchNorm running statistics.
    Maps PyTorch running_mean/running_var to Equinox ema_state_index.
    """
    mapping = {}
    
    # Root level BN
    mapping["bn1.running_mean"] = "bn1.ema_state_index.init[0]"
    mapping["bn1.running_var"] = "bn1.ema_state_index.init[1]"
    
    # Layer BN stats
    layer_configs = [(1, 8), (2, 8), (3, 36), (4, 8)]  # Generous upper bounds
    
    for layer_idx, max_blocks in layer_configs:
        for block_idx in range(max_blocks):
            pt_prefix = f"layer{layer_idx}.{block_idx}"
            eqx_prefix = f"layer{layer_idx}.layers[{block_idx}]"
            
            for conv_idx in range(1, 4):  # bn1, bn2, bn3
                mapping[f"{pt_prefix}.bn{conv_idx}.running_mean"] = f"{eqx_prefix}.bn{conv_idx}.ema_state_index.init[0]"
                mapping[f"{pt_prefix}.bn{conv_idx}.running_var"] = f"{eqx_prefix}.bn{conv_idx}.ema_state_index.init[1]"
            
            # Downsample BN stats - need to handle the Sequential structure differently
            # The downsample Sequential has [Conv2d, BatchNorm], so the BN is at index [1]
            mapping[f"{pt_prefix}.downsample.1.running_mean"] = f"{eqx_prefix}.downsample.layers[1].ema_state_index.init[0]"
            mapping[f"{pt_prefix}.downsample.1.running_var"] = f"{eqx_prefix}.downsample.layers[1].ema_state_index.init[1]"
    
    return mapping


def flexible_load_torch_weights(
    model: eqx.Module,
    torch_weights: str = None,
    verbose: bool = True
) -> eqx.Module:
    """
    Load PyTorch weights into Equinox model using flexible name-based mapping.
    
    This approach is much more robust than positional alignment and follows
    industry best practices (similar to HuggingFace -> Flax conversions).
    
    **Arguments:**
    
    - `model`: An `eqx.Module` for which parameters will be loaded
    - `torch_weights`: A string pointing to PyTorch weights on disk or download URL
    - `verbose`: Whether to print loading progress. Defaults to `True`
    
    **Returns:**
        The model with weights loaded from the PyTorch checkpoint.
    """
    if "torch" not in sys.modules:
        raise RuntimeError(
            "Torch package not found! Weight loading requires the torch package."
        )
    
    if torch_weights is None:
        raise ValueError("torch_weights parameter cannot be empty!")
    
    # Download weights if needed
    if not os.path.exists(torch_weights):
        global _TEMP_DIR
        filepath = os.path.join(_TEMP_DIR, os.path.basename(torch_weights))
        if os.path.exists(filepath):
            if verbose:
                logging.info(f"Using cached file at {filepath}")
        else:
            os.makedirs(_TEMP_DIR, exist_ok=True)
            if verbose:
                print(f"Downloading weights from {torch_weights}")
            torch.hub.download_url_to_file(torch_weights, filepath)
    else:
        filepath = torch_weights
    
    # Load PyTorch state dict
    pytorch_state_dict = torch.load(filepath, map_location="cpu")
    
    if verbose:
        print("=== Flexible Weight Loading ===")
    
    # Get parameter mappings
    param_mapping = create_resnet_parameter_mapping()
    bn_stats_mapping = create_resnet_bn_stats_mapping()
    
    # Filter trainable parameters (exclude running stats)
    trainable_pytorch = {name: weight for name, weight in pytorch_state_dict.items() 
                        if "running" not in name and "num_batches" not in name}
    
    if verbose:
        print(f"PyTorch trainable parameters: {len(trainable_pytorch)}")
    
    # Load trainable parameters by name
    successful_loads = 0
    failed_loads = []
    
    for pytorch_name, equinox_path in param_mapping.items():
        if pytorch_name in trainable_pytorch:
            try:
                # Get PyTorch weight
                pytorch_weight = trainable_pytorch[pytorch_name]
                pytorch_array = jnp.asarray(pytorch_weight.detach().numpy())
                
                # Get current Equinox parameter to check shape compatibility
                try:
                    current_param = get_nested_attr(model, equinox_path)
                except (AttributeError, IndexError, KeyError):
                    # Parameter path doesn't exist in this model (e.g., different architecture)
                    continue
                
                # Check shape compatibility
                if pytorch_array.shape == current_param.shape:
                    # Replace the parameter using eqx.tree_at
                    where_fn = lambda m: get_nested_attr(m, equinox_path)
                    model = eqx.tree_at(where_fn, model, pytorch_array)
                    
                    if verbose:
                        print(f"✅ {pytorch_name} -> {equinox_path} {pytorch_array.shape}")
                    successful_loads += 1
                else:
                    if verbose:
                        print(f"❌ Shape mismatch: {pytorch_name} {pytorch_array.shape} vs {equinox_path} {current_param.shape}")
                    failed_loads.append((pytorch_name, equinox_path, "shape_mismatch"))
                    
            except Exception as e:
                if verbose:
                    print(f"❌ Error loading {pytorch_name} -> {equinox_path}: {e}")
                failed_loads.append((pytorch_name, equinox_path, str(e)))
    
    # Load BatchNorm running statistics
    if verbose:
        print(f"\n=== Loading BatchNorm Running Statistics ===")
    
    bn_stats_loaded = 0
    
    for pytorch_name, equinox_path in bn_stats_mapping.items():
        if pytorch_name in pytorch_state_dict:
            try:
                # Check if the path exists in the model
                try:
                    current_param = get_nested_attr(model, equinox_path)
                except (AttributeError, IndexError, KeyError):
                    # Path doesn't exist in this model
                    continue
                
                pytorch_weight = pytorch_state_dict[pytorch_name]
                pytorch_array = jnp.asarray(pytorch_weight.detach().numpy())
                
                # Load the running statistic
                where_fn = lambda m: get_nested_attr(m, equinox_path)
                model = eqx.tree_at(where_fn, model, pytorch_array)
                
                if verbose:
                    print(f"✅ {pytorch_name} -> {equinox_path}")
                bn_stats_loaded += 1
                
            except Exception as e:
                if verbose:
                    print(f"❌ Failed to load {pytorch_name}: {e}")
    
    if verbose:
        print(f"\n=== Loading Results ===")
        print(f"✅ Trainable parameters loaded: {successful_loads}")
        print(f"✅ BatchNorm statistics loaded: {bn_stats_loaded}")
        print(f"❌ Failed loads: {len(failed_loads)}")
        
        if failed_loads and len(failed_loads) <= 5:
            print("\nFailed loads:")
            for pt_name, eqx_path, reason in failed_loads:
                print(f"  {pt_name} -> {eqx_path}: {reason}")
    
    return model


# Legacy function for backward compatibility
def load_torch_weights_legacy(model: eqx.Module, torch_weights: str = None) -> eqx.Module:
    """
    Legacy weight loading function using positional alignment.
    
    ⚠️ DEPRECATED: Use flexible_load_torch_weights instead for better robustness.
    This function is kept for backward compatibility but may be removed in future versions.
    """
    warnings.warn(
        "load_torch_weights_legacy is deprecated. Use flexible_load_torch_weights for better robustness.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import the old implementation from utils.py
    from .utils import load_torch_weights as old_load_torch_weights
    return old_load_torch_weights(model, torch_weights) 
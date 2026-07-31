#!/usr/bin/env python3
"""
Latent Space Analysis & Plotting Script

This script demonstrates how to enable the latent token hooks and extract
the `class_token` and `patch_tokens` from a trained model for PCA visualization.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from umap import UMAP  # Uncomment if you want to use UMAP

# Import your model initialization here
# from weathergen.train.trainer import Trainer
# from omegaconf import OmegaConf

def plot_pca(class_tokens, patch_tokens, out_file="latent_space_pca.png"):
    """
    Plots the PCA of the latent tokens.
    
    Args:
        class_tokens (np.ndarray): Shape (batch, steps, dim)
        patch_tokens (np.ndarray): Shape (batch, steps, num_patches, dim)
        out_file (str): Path to save the plot
    """
    print("Performing PCA on latent space...")
    
    # Flatten tokens for PCA
    # Shape: (batch * steps, dim)
    ct_flat = class_tokens.reshape(-1, class_tokens.shape[-1])
    
    # Shape: (batch * steps * num_patches, dim)
    pt_flat = patch_tokens.reshape(-1, patch_tokens.shape[-1])
    
    # Fit PCA on patch tokens to understand the overall variance
    pca = PCA(n_components=2)
    pca.fit(pt_flat)
    
    pt_pca = pca.transform(pt_flat)
    ct_pca = pca.transform(ct_flat)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot patch tokens in the background (light gray/blue)
    plt.scatter(pt_pca[:, 0], pt_pca[:, 1], alpha=0.1, s=2, c='tab:blue', label='Patch Tokens (Local)')
    
    # Plot class tokens prominently
    plt.scatter(ct_pca[:, 0], ct_pca[:, 1], alpha=0.8, s=50, c='tab:red', edgecolor='black', label='Class Tokens (Global)')
    
    plt.title("Latent Space of WeatherGenerator (PCA)")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    args = parser.parse_args()
    
    print("This is a template script. To run it fully, you need to instantiate your model and dataloader.")
    
    # =========================================================================
    # 1. LOAD CONFIG & ENABLE HOOK
    # =========================================================================
    # cf = OmegaConf.load(args.config)
    
    # *** CRITICAL STEP ***
    # Tell the model to extract the tokens for EVERY autoregressive step!
    # cf.return_latent_tokens = True 
    
    # =========================================================================
    # 2. LOAD MODEL & DATALOADER
    # =========================================================================
    # model, _ = init_model_and_shard(...)
    # model.load_state_dict(torch.load(args.checkpoint))
    # model.eval()
    
    # =========================================================================
    # 3. RUN INFERENCE & EXTRACT TOKENS
    # =========================================================================
    # with torch.no_grad():
    #     output = model(model_params, batch)
    #
    #     class_tokens = []
    #     patch_tokens = []
    #
    #     # Iterate through all autoregressive steps
    #     for step in range(len(output.latent)):
    #         latent_state = output.get_latent_prediction(step)["latent_state"]
    #         class_tokens.append(latent_state.class_token.cpu().numpy())
    #         patch_tokens.append(latent_state.patch_tokens.cpu().numpy())
    #
    #     # Stack to (batch, steps, dim)
    #     class_tokens = np.stack(class_tokens, axis=1)
    #     patch_tokens = np.stack(patch_tokens, axis=1)
    #
    #     plot_pca(class_tokens, patch_tokens)

    # --- DUMMY DATA FOR DEMONSTRATION ---
    print("Generating dummy data to demonstrate the plot...")
    batch_size, steps, num_patches, dim = 2, 4, 1035, 256
    dummy_class = np.random.randn(batch_size, steps, dim) * 2
    dummy_patches = np.random.randn(batch_size, steps, num_patches, dim)
    
    plot_pca(dummy_class, dummy_patches)

if __name__ == "__main__":
    main()

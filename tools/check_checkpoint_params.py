import torch
import os

CHECKPOINT_PATH = "model/checkpoints/latest_checkpoint.pth"

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

state_dict = ckpt["model_state"]

total_params = sum(p.numel() for p in state_dict.values())

print("Checkpoint parameter count:", total_params)
"""Checkpointing utilities for training resumption.

This module provides checkpoint management for long-running training,
allowing training to be interrupted and resumed from saved snapshots.
"""

import os
import json
import torch
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """Manages training checkpoints for resumable training.

    Saves training state periodically and provides methods to resume
    training from the last checkpoint.

    Usage:
        # Initialize checkpoint manager
        ckpt_manager = CheckpointManager(checkpoint_dir="./checkpoints")

        # Check for existing checkpoint
        if ckpt_manager.has_checkpoint("my_experiment"):
            state = ckpt_manager.load_checkpoint("my_experiment")
            start_epoch = state["epoch"] + 1
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
        else:
            start_epoch = 0

        # During training, save checkpoints
        for epoch in range(start_epoch, num_epochs):
            # ... training code ...
            ckpt_manager.save_checkpoint(
                "my_experiment",
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                metrics={"loss": loss, "accuracy": acc}
            )

        # Clean up after successful completion
        ckpt_manager.delete_checkpoint("my_experiment")
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints", keep_last_n: int = 2):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            keep_last_n: Number of recent checkpoints to keep (default: 2)
        """
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _get_checkpoint_path(self, experiment_name: str, suffix: str = "") -> str:
        """Get path for a checkpoint file."""
        name = f"{experiment_name}{suffix}.ckpt"
        return os.path.join(self.checkpoint_dir, name)

    def _get_metadata_path(self, experiment_name: str) -> str:
        """Get path for checkpoint metadata."""
        return os.path.join(self.checkpoint_dir, f"{experiment_name}_meta.json")

    def has_checkpoint(self, experiment_name: str) -> bool:
        """Check if a checkpoint exists for the given experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            True if checkpoint exists
        """
        checkpoint_path = self._get_checkpoint_path(experiment_name)
        return os.path.exists(checkpoint_path)

    def save_checkpoint(
        self,
        experiment_name: str,
        epoch: int,
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
        mask: Optional[torch.Tensor] = None,
        mask_optimizer_state_dict: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Save a training checkpoint.

        Args:
            experiment_name: Name of the experiment
            epoch: Current epoch number
            model_state_dict: Model state dictionary
            optimizer_state_dict: Optimizer state dictionary (optional)
            scheduler_state_dict: Learning rate scheduler state (optional)
            mask: DiET mask tensor (optional)
            mask_optimizer_state_dict: Mask optimizer state (optional)
            metrics: Training metrics (optional)
            extra_state: Any additional state to save (optional)
            **kwargs: Additional key-value pairs to save

        Returns:
            Path to the saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "timestamp": datetime.now().isoformat(),
        }

        if optimizer_state_dict is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state_dict

        if scheduler_state_dict is not None:
            checkpoint["scheduler_state_dict"] = scheduler_state_dict

        if mask is not None:
            checkpoint["mask"] = mask.detach().cpu()

        if mask_optimizer_state_dict is not None:
            checkpoint["mask_optimizer_state_dict"] = mask_optimizer_state_dict

        if metrics is not None:
            checkpoint["metrics"] = metrics

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        # Add any additional kwargs
        checkpoint.update(kwargs)

        # Save checkpoint
        checkpoint_path = self._get_checkpoint_path(experiment_name)
        torch.save(checkpoint, checkpoint_path)

        # Save metadata
        metadata = {
            "experiment_name": experiment_name,
            "epoch": epoch,
            "timestamp": checkpoint["timestamp"],
            "metrics": metrics or {},
        }
        with open(self._get_metadata_path(experiment_name), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Checkpoint saved: {experiment_name} (epoch {epoch})")
        return checkpoint_path

    def load_checkpoint(
        self, experiment_name: str, device: str = "cpu"
    ) -> Optional[Dict[str, Any]]:
        """Load a checkpoint.

        Args:
            experiment_name: Name of the experiment
            device: Device to load tensors to

        Returns:
            Checkpoint dictionary or None if not found
        """
        checkpoint_path = self._get_checkpoint_path(experiment_name)

        if not os.path.exists(checkpoint_path):
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            print(f"Loaded checkpoint: {experiment_name} (epoch {checkpoint['epoch']})")
            return checkpoint
        except Exception as e:
            print(f"Failed to load checkpoint {experiment_name}: {e}")
            return None

    def delete_checkpoint(self, experiment_name: str) -> bool:
        """Delete a checkpoint after successful training completion.

        Args:
            experiment_name: Name of the experiment

        Returns:
            True if deleted successfully
        """
        checkpoint_path = self._get_checkpoint_path(experiment_name)
        metadata_path = self._get_metadata_path(experiment_name)

        deleted = False
        for path in [checkpoint_path, metadata_path]:
            if os.path.exists(path):
                os.remove(path)
                deleted = True

        if deleted:
            print(f"Deleted checkpoint: {experiment_name}")

        return deleted

    def get_checkpoint_info(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint metadata without loading full checkpoint.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = self._get_metadata_path(experiment_name)

        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def list_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """List all available checkpoints.

        Returns:
            Dictionary mapping experiment names to their metadata
        """
        checkpoints = {}

        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith("_meta.json"):
                experiment_name = filename[:-10]  # Remove "_meta.json"
                info = self.get_checkpoint_info(experiment_name)
                if info:
                    checkpoints[experiment_name] = info

        return checkpoints


def get_default_checkpoint_manager(
    output_dir: str = "./outputs/xai_experiments",
) -> CheckpointManager:
    """Get a checkpoint manager with default settings.

    Args:
        output_dir: Base output directory

    Returns:
        CheckpointManager instance
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    return CheckpointManager(checkpoint_dir=checkpoint_dir)

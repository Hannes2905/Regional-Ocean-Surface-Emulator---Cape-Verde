"""
Simple embedder for ocean surface variable encoding/decoding.

Handles:
- Concatenation of current state and previous state along the channel dimension

The embedder automatically adjusts input/output channel counts.
"""

import torch
import torch.nn as nn


class SimpleEmbedder(nn.Module):
    """
    Encodes ocean state variables into a single tensor for the UNet backbone,
    and decodes the backbone output back.

    encode: [B, n_vars, H, W] + optional [B, n_vars, H, W] → [B, C_in, H, W]
    decode: [B, n_vars, H, W] → [B, n_vars, H, W] (identity)

    Args:
        n_vars: Number of physical variables
        load_prev: Whether the previous timestep is concatenated as input
    """

    def __init__(self, n_vars: int = 1, load_prev: bool = True):
        super().__init__()
        self.n_vars = n_vars
        self.load_prev = load_prev

    @property
    def n_input_channels(self) -> int:
        """Number of input channels for the UNet backbone."""
        return self.n_vars * (2 if self.load_prev else 1)

    @property
    def n_output_channels(self) -> int:
        """Number of output channels from the UNet backbone."""
        return self.n_vars

    def encode(self, state: torch.Tensor, prev_state: torch.Tensor = None) -> torch.Tensor:
        """
        Encode state (and optional prev_state) into backbone input tensor.

        Args:
            state: Current state [B, n_vars, H, W]
            prev_state: Previous state [B, n_vars, H, W] or None

        Returns:
            x: Encoded input [B, C_in, H, W]
        """
        if self.load_prev and prev_state is not None:
            x = torch.cat([state, prev_state], dim=1)  # [B, 2*n_vars, H, W]
        else:
            x = state
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode backbone output back to state format.

        Args:
            x: Backbone output [B, n_vars, H, W]

        Returns:
            state: Decoded state [B, n_vars, H, W]
        """
        return x

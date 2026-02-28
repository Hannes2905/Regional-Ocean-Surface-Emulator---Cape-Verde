"""
Dataloader for CMEMS ocean reanalysis data stored in single NetCDF files.

Handles:
- Single .nc file with dimensions (time, latitude, longitude)
- NaN-based land/ocean mask derivation
- Time-based train/test split (80/20 by default)
- Separate validation dataset from a different file (planned)
- Extensible variable list (thetao now, uo/vo later)
- Previous timestep loading
- Z-score normalization (optional)

Returns dict batches:
    state:          [n_vars, H, W]   current timestep
    next_state:     [n_vars, H, W]   target (t + lead_time)
    prev_state:     [n_vars, H, W]   (if load_prev) previous timestep
    future_states:  [multistep, n_vars, H, W]  for multistep training
    mask:           [1, H, W]        ocean=1, land=0
    timestamp:      int64 scalar     seconds since epoch
"""

import numpy as np
import torch
import xarray as xr
from typing import List, Optional, Dict


class CMEMSDataset(torch.utils.data.Dataset):
    """
    Args:
        path: Path to the .nc file
        variables: List of variable names to load (e.g., ["thetao"])
        domain: One of "train", "test", "all". Controls time-based splitting.
        train_frac: Fraction of timesteps used for training (default 0.8)
        lead_time_hours: Hours between input and target timestep
        multistep: Number of future steps to load for multistep training
        load_prev: Whether to also load the previous timestep
        norm_scheme: Normalization scheme ("z-score" or None)
        data_mean: Dict of variable means for z-score normalization
        data_std: Dict of variable stds for z-score normalization
    """

    def __init__(
        self,
        path: str,
        variables: List[str] = None,
        domain: str = "train",
        train_frac: float = 0.8,
        lead_time_hours: int = 24,
        multistep: int = 1,
        load_prev: bool = True,
        norm_scheme: Optional[str] = None,
        data_mean: Optional[Dict[str, float]] = None,
        data_std: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        if variables is None:
            variables = ["thetao"]

        self.variables = variables
        self.n_vars = len(variables)
        self.domain = domain
        self.train_frac = train_frac
        self.lead_time_hours = lead_time_hours
        self.multistep = multistep
        self.load_prev = load_prev
        self.norm_scheme = norm_scheme
        self.path = path

        # ── Load dataset ──
        ds = xr.open_dataset(path)

        # Validate variables exist
        for var in variables:
            if var not in ds.data_vars:
                available = list(ds.data_vars)
                raise ValueError(
                    f"Variable '{var}' not found in dataset. Available: {available}"
                )

        # ── Handle coordinate names ──
        # Standardize latitude/longitude dimension names
        for dim in ds.dims:
            if dim.lower() in ("latitude", "lat"):
                self.lat_dim = dim
            elif dim.lower() in ("longitude", "lon"):
                self.lon_dim = dim

        # ── Derive land/ocean mask from NaN values ──
        # Use first timestep and union across all variables
        first_slice = ds.isel(time=0)
        mask_np = np.ones(
            (first_slice.sizes[self.lat_dim], first_slice.sizes[self.lon_dim]),
            dtype=np.float32,
        )
        for var in variables:
            var_data = first_slice[var].values
            mask_np[np.isnan(var_data)] = 0.0

        self.mask = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        # ── Load all data into memory as float32 tensors ──
        # Shape per variable: [T, H, W]
        self.data = {}
        for var in variables:
            arr = ds[var].values.astype(np.float32)
            # Replace NaN with 0 (masked regions)
            arr = np.nan_to_num(arr, nan=0.0)
            self.data[var] = torch.from_numpy(arr)  # [T, H, W]

        # ── Timestamps ──
        self.timestamps = ds.time.values  # numpy datetime64 array
        total_T = len(self.timestamps)

        # ── Compute timedelta between consecutive steps ──
        dt_ns = (self.timestamps[1] - self.timestamps[0]).astype("timedelta64[s]").astype(int)
        self.timedelta_hours = dt_ns / 3600
        if lead_time_hours % self.timedelta_hours != 0:
            raise ValueError(
                f"lead_time ({lead_time_hours}h) is not divisible by data resolution ({self.timedelta_hours}h)."
            )
        self.index_jump = int(lead_time_hours / self.timedelta_hours)

        # ── Time-based train/test split ──
        split_idx = int(total_T * train_frac)
        if domain == "train":
            self.time_indices = list(range(0, split_idx))
        elif domain == "test":
            self.time_indices = list(range(split_idx, total_T))
        elif domain == "val":
            # Validation in future from separate file — for now fall back to test split
            self.time_indices = list(range(split_idx, total_T))
        elif domain == "all":
            self.time_indices = list(range(0, total_T))
        else:
            raise ValueError(f"Unknown domain '{domain}'. Use 'train', 'test', 'val', or 'all'.")

        # ── Trim indices to account for prev/future buffers ──
        buffer_start = self.index_jump if self.load_prev else 0
        buffer_end = self.multistep * self.index_jump

        # Filter valid indices
        self.valid_indices = [
            t for t in self.time_indices
            if (t - buffer_start) >= 0 and (t + buffer_end) < total_T
        ]

        if len(self.valid_indices) == 0:
            raise ValueError(
                f"No valid samples for domain='{domain}'. "
                f"Total timesteps={total_T}, split_idx={split_idx}, "
                f"buffer_start={buffer_start}, buffer_end={buffer_end}"
            )

        # ── Normalization stats ──
        if norm_scheme == "z-score" and data_mean is not None and data_std is not None:
            self.data_mean = {k: torch.tensor(v).float() for k, v in data_mean.items()}
            self.data_std = {k: torch.tensor(v).float() for k, v in data_std.items()}
        else:
            if norm_scheme == "z-score":
                print("Warning: z-score requested but no mean/std provided. Normalization disabled.")
            self.norm_scheme = None
            self.data_mean = None
            self.data_std = None

        ds.close()

        print(f"CMEMSDataset initialized: domain={domain}, "
              f"variables={variables}, samples={len(self.valid_indices)}, "
              f"spatial=({self.mask.shape[1]}, {self.mask.shape[2]})")

    def __len__(self):
        return len(self.valid_indices)

    def _get_state(self, time_idx: int) -> torch.Tensor:
        """
        Get the state tensor for a given absolute time index.
        Returns: [n_vars, H, W]
        """
        tensors = [self.data[var][time_idx] for var in self.variables]
        return torch.stack(tensors, dim=0)  # [n_vars, H, W]

    def __getitem__(self, i: int) -> dict:
        t = self.valid_indices[i]
        out = {}

        # Current state
        out["state"] = self._get_state(t)  # [n_vars, H, W]

        # Mask (same for all samples)
        out["mask"] = self.mask.clone()  # [1, H, W]

        # Timestamp
        ts_seconds = self.timestamps[t].astype("datetime64[s]").astype(np.int64)
        out["timestamp"] = torch.tensor(ts_seconds, dtype=torch.int64)

        # Lead time
        out["lead_time_hours"] = torch.tensor(
            self.lead_time_hours * self.multistep, dtype=torch.int32
        )

        # Previous state
        if self.load_prev:
            out["prev_state"] = self._get_state(t - self.index_jump)

        # Future states for future multistep training
        future_states = []
        for k in range(1, self.multistep + 1):
            future_states.append(self._get_state(t + k * self.index_jump))
        out["future_states"] = torch.stack(future_states, dim=0)  # [multistep, n_vars, H, W]

        # next_state is the first future state (backward compatibility)
        out["next_state"] = out["future_states"][0]

        # Normalize if configured
        if self.norm_scheme:
            out = self.normalize(out)

        return out

    def normalize(self, sample: dict) -> dict:
        """Apply z-score normalization to state tensors."""
        out = {}
        for k, v in sample.items():
            if "state" in k and isinstance(v, torch.Tensor):
                mean_t = self._get_norm_tensor(self.data_mean)
                std_t = self._get_norm_tensor(self.data_std)
                if v.dim() == 3:  # [n_vars, H, W]
                    out[k] = (v - mean_t[:, None, None]) / std_t[:, None, None]
                elif v.dim() == 4:  # [multistep, n_vars, H, W]
                    out[k] = (v - mean_t[None, :, None, None]) / std_t[None, :, None, None]
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    def denormalize(self, sample: dict) -> dict:
        """Reverse z-score normalization on state tensors."""
        out = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor) and any(
                x in k for x in ["state", "prediction", "output"]
            ):
                mean_t = self._get_norm_tensor(self.data_mean)
                std_t = self._get_norm_tensor(self.data_std)
                if v.dim() == 3:
                    out[k] = v * std_t[:, None, None] + mean_t[:, None, None]
                elif v.dim() == 4:
                    out[k] = v * std_t[None, :, None, None] + mean_t[None, :, None, None]
                else:
                    out[k] = v
            else:
                out[k] = v
        return out

    def _get_norm_tensor(self, stats_dict: dict) -> torch.Tensor:
        """Build [n_vars] tensor from stats dict, matching self.variables order."""
        return torch.stack([stats_dict[var] for var in self.variables])

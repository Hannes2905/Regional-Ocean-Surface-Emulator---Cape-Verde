"""
Evaluation utilities for the ROSE-CV ocean surface emulator.

Provides three visualisation functions used in the model evaluation notebook:
- plot_training_history: step-wise and epoch-wise loss curves from CSV logs
- display_final_metrics: formatted table of final test-set metrics
- plot_model_predictions: side-by-side ground truth / prediction / error maps
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import numpy as np


def plot_training_history(csv_path):
    """Plot step-wise and epoch-wise training/validation loss curves.

    Args:
        csv_path: Path to a Lightning CSV log file containing columns
                  'step', 'train/loss_step', 'epoch', 'train/loss_epoch',
                  and 'val/loss'.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Apply a clean seaborn theme
    sns.set_theme(style="whitegrid", context="talk")
    # Adjusted to 2 subplots instead of 3, and resized the figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # -------------------------------------------------------------
    # 1. Step-wise Training Loss (with smoothing)
    # -------------------------------------------------------------
    # Drop NaNs specifically for the step loss column and sort
    train_step_df = df[['step', 'train/loss_step']].dropna().sort_values('step')
    
    if not train_step_df.empty:
        # Plot raw noisy loss
        axs[0].plot(train_step_df['step'], train_step_df['train/loss_step'], 
                    alpha=0.3, color='steelblue', label='Raw Train Loss')
        
        # Calculate and plot a rolling average for readability
        smoothed = train_step_df['train/loss_step'].rolling(window=10, min_periods=1).mean()
        axs[0].plot(train_step_df['step'], smoothed, color='darkblue', 
                    linewidth=2.5, label='Smoothed (Window=10)')
        
    axs[0].set_title("Training Loss per Step", fontweight='bold')
    axs[0].set_xlabel("Global Step")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # -------------------------------------------------------------
    # 2. Epoch-wise Training vs Validation Loss
    # -------------------------------------------------------------
    # Group by epoch and take the mean to compress multiple logging rows per epoch
    epoch_train_df = df[['epoch', 'train/loss_epoch']].dropna().groupby('epoch').mean().reset_index()
    epoch_val_df = df[['epoch', 'val/loss']].dropna().groupby('epoch').mean().reset_index()

    if not epoch_train_df.empty:
        axs[1].plot(epoch_train_df['epoch'], epoch_train_df['train/loss_epoch'], 
                    marker='o', linewidth=2.5, color='steelblue', label='Train Loss')
    if not epoch_val_df.empty:
        axs[1].plot(epoch_val_df['epoch'], epoch_val_df['val/loss'], 
                    marker='s', linewidth=2.5, color='darkorange', label='Val Loss')

    axs[1].set_title("Loss (Epoch-Level)", fontweight='bold')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def display_final_metrics(csv_path):
    """Print a formatted table of final evaluation metrics.

    Args:
        csv_path: Path to a two-column CSV with 'metric' and 'value' columns.
    """
    # Load the metrics
    df = pd.read_csv(csv_path)
    
    # Format the values: 4 decimal places for floats, integers for whole numbers
    df['value'] = df['value'].apply(
        lambda x: f"{x:.4f}" if isinstance(x, float) and x % 1 != 0 else str(int(x))
    )
    
    # Rename columns for a cleaner presentation
    df.columns = ['Metric', 'Value']
    
    # Print a formatted text table
    print("=" * 45)
    print(f"{'FINAL EVALUATION METRICS':^45}")
    print("=" * 45)
    for _, row in df.iterrows():
        # Left-align the metric name, right-align the value
        print(f"{row['Metric']:<25} | {row['Value']:>15}")
    print("=" * 45)



def plot_model_predictions(nc_path, original_dataset_path=None):
    """Plot ground truth, model prediction, and error maps for two test samples.

    Args:
        nc_path: Path to a NetCDF file containing 'gt_thetao', 'pred_thetao',
                 'mask', and 'time' variables.
        original_dataset_path: Optional path to the original CMEMS dataset
                               used as a fallback for geographic coordinates.
    """
    # Load the evaluation output
    ds = xr.open_dataset(nc_path)
    
    # Extract variables
    y_true = ds['gt_thetao'].values
    y_pred = ds['pred_thetao'].values
    mask = ds['mask'].values if 'mask' in ds.variables else np.ones_like(y_true)
    times = ds['time'].values
    
    # Handle potentially missing geographical coordinates in the prediction file
    if 'lon' in ds.coords and 'lat' in ds.coords:
        lons = ds['lon'].values
        lats = ds['lat'].values
    elif original_dataset_path:
        # Fallback to original dataset if coordinates weren't saved in predictions.nc
        orig_ds = xr.open_dataset(original_dataset_path)
        lons = orig_ds['longitude'].values
        lats = orig_ds['latitude'].values
    else:
        # Creating dummy coordinates to prevent immediate crashes, but the map will be misaligned
        lons = np.linspace(-32.50, -11.50, 512) 
        lats = np.linspace(5.00, 26.00, 512)

    # Apply mask (set land pixels to NaN)
    y_true = np.where(mask == 1, y_true, np.nan)
    y_pred = np.where(mask == 1, y_pred, np.nan)

    # Select two timesteps: one near the start, one near the end of the test set
    indices = np.array([10, -10])

    # Plotting setup
    fig, axs = plt.subplots(2, 3, figsize=(22, 14), 
                            subplot_kw={'projection': ccrs.PlateCarree()})

    for row, idx in enumerate(indices):
        target_snap = y_true[idx]
        pred_snap = y_pred[idx]
        diff_snap = target_snap - pred_snap
        
        # Calculate local 'robust' limits for this specific row
        v_min = np.nanpercentile(target_snap, 2)
        v_max = np.nanpercentile(target_snap, 98)
        
        sst_kwargs = {
            'cmap': cmocean.cm.thermal,
            'shading': 'auto',
            'transform': ccrs.PlateCarree(),
            'vmin': v_min,
            'vmax': v_max
        }

        # Column 0: Ground Truth
        im0 = axs[row, 0].pcolormesh(lons, lats, target_snap, **sst_kwargs)
        # Using numpy datetime handling to format the title cleanly
        date_str = np.datetime_as_string(times[idx], unit='D')
        axs[row, 0].set_title(f"Ground Truth (t)\n{date_str}", fontweight='bold')
        plt.colorbar(im0, ax=axs[row, 0], label=r'$\theta_o$ [$^\circ$C]', shrink=0.7)

        # Column 1: Model Prediction
        im1 = axs[row, 1].pcolormesh(lons, lats, pred_snap, **sst_kwargs)
        axs[row, 1].set_title(f"Model Prediction", fontweight='bold')
        plt.colorbar(im1, ax=axs[row, 1], label=r'$\theta_o$ [$^\circ$C]', shrink=0.7)

        # Column 2: Difference (Error Map)
        im2 = axs[row, 2].pcolormesh(lons, lats, diff_snap, 
                                    cmap='RdBu_r', 
                                    transform=ccrs.PlateCarree(), 
                                    vmin=-1.5, vmax=1.5,
                                    shading='auto')
        axs[row, 2].set_title(f"Error (GT - Prediction)", fontweight='bold')
        plt.colorbar(im2, ax=axs[row, 2], label=r'Error [$^\circ$C]', shrink=0.7)

    # Global Cartographic Enhancements
    for ax in axs.flat:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

    fig.suptitle("Vanilla U-Net Evaluation: Random Test Samples", fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_parity(y_true, y_pred, labels=None, output_path=None):
    labels = labels or ['dS_dt', 'dI_dt', 'dR_dt']
    n = y_true.shape[1] if hasattr(y_true, 'shape') and y_true.ndim > 1 else 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        yt = y_true[:, i] if n > 1 else y_true
        yp = y_pred[:, i] if n > 1 else y_pred
        ax.scatter(yt, yp, s=6, alpha=0.4)
        ax.set_title(f"{labels[i]}: true vs pred")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    return fig


def plot_trajectory(t, S, I, R, output_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, S, label='S', color='blue')
    ax.plot(t, I, label='I', color='red')
    ax.plot(t, R, label='R', color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fraction')
    ax.legend()
    ax.set_title('SIR Trajectory')
    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    return fig


def plot_vector_field(model, output_path=None):
    import torch
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    s_vals = np.linspace(0, 1, 20)
    i_vals = np.linspace(0, 1, 20)
    labels = ['dS_dt', 'dI_dt', 'dR_dt']
    for ax_idx, ax in enumerate(axes):
        Z = np.zeros((20, 20))
        for si, sv in enumerate(s_vals):
            for ii, iv in enumerate(i_vals):
                rv = max(0, 1 - sv - iv)
                x = torch.tensor([[sv, iv, rv]], dtype=torch.float32)
                with torch.no_grad():
                    out = model(x).numpy()
                Z[ii, si] = out[0, ax_idx]
        im = ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', aspect='auto')
        ax.set_title(labels[ax_idx])
        ax.set_xlabel('S')
        ax.set_ylabel('I')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    return fig


def plot_training_history(history, output_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Training History')
    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    return fig

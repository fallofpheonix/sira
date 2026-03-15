import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def create_summary_dashboard(data_path=None, model_path=None, output_path=None):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('SIRA Summary Dashboard', fontsize=16)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title('Data Distribution')
    ax1.text(0.5, 0.5, 'Load data to display', ha='center', va='center',
             transform=ax1.transAxes)

    if data_path and Path(data_path).exists():
        import pandas as pd
        df = pd.read_csv(data_path)
        ax1.hist(df['S'], bins=30, alpha=0.5, label='S')
        ax1.hist(df['I'], bins=30, alpha=0.5, label='I')
        ax1.legend()

    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    return fig

import torch
import pandas as pd
from src.training.metrics import compute_r2, compute_rmse, compute_mae


class ModelBenchmark:
    def evaluate(self, model, test_loader):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                preds.append(model(X_batch))
                targets.append(y_batch)
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        labels = ['dS_dt', 'dI_dt', 'dR_dt']
        results = {}
        for i, label in enumerate(labels):
            results[label] = {
                'r2': compute_r2(preds[:, i], targets[:, i]),
                'rmse': compute_rmse(preds[:, i], targets[:, i]),
                'mae': compute_mae(preds[:, i], targets[:, i]),
            }
        results['overall'] = {
            'r2': compute_r2(preds, targets),
            'rmse': compute_rmse(preds, targets),
            'mae': compute_mae(preds, targets),
        }
        return results

    def compare_models(self, models, test_loader):
        rows = []
        for name, model in models.items():
            metrics = self.evaluate(model, test_loader)
            rows.append({'model': name, **metrics['overall']})
        return pd.DataFrame(rows)


def evaluate_symbolic_recovery(sindy_results, true_beta, true_gamma):
    recovered = {}
    for eq, terms in sindy_results.items():
        recovered[eq] = {n: c for n, c in terms}
    return recovered

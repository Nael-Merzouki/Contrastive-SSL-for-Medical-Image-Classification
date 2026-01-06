import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_prob, n_classes):
  y_pred = np.argmax(y_prob, axis=1)
  acc = accuracy_score(y_true, y_pred)
  if n_classes == 2:
    auroc = roc_auc_score(y_true, y_prob[:,1])
  else:
    y_true_1hot = np.eye(n_classes)[y_true]

    try:
      auroc = roc_auc_score(y_true_1hot, y_prob, average='macro', multi_class='ovr')
    except Exception:
      auroc = np.nan

  return {'acc':acc,'auroc':auroc}

def reliability_diagram(y_true, y_prob, n_bins=10, save_path=None):
  confidences = np.max(y_prob, axis=1)
  preds = np.argmax(y_prob, axis=1)
  correct = (preds == y_true).astype(np.float32)
  bins = np.linspace(0,1,n_bins+1)
  ece = 0.0; xs = np.linspace(0.5/n_bins, 1-0.5/n_bins, n_bins)
  bin_accs = []; bin_confs = []

  for i in range(n_bins):
    lo, hi = bins[i], bins[i+1]
    mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)

    if mask.sum() == 0:
      bin_accs.append(0.0); bin_confs.append((lo+hi)/2); continue
    
    acc_i = correct[mask].mean(); conf_i = confidences[mask].mean(); frac_i = mask.mean()
    ece += abs(acc_i - conf_i) * frac_i
    bin_accs.append(acc_i); bin_confs.append(conf_i)

  if save_path is not None:
    plt.figure(); plt.plot([0,1],[0,1], linestyle='--');
    plt.bar(xs, bin_accs, width=1.0/n_bins, alpha=0.6, edgecolor='k');
    plt.plot(xs, bin_confs, marker='o');
    plt.xlabel('Confidence'); plt.ylabel('Accuracy');
    plt.title(f'Reliability Diagram (ECE={ece:.3f})');
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    
  return float(ece)

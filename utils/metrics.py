# -*- coding: utf-8 -*-
'''
=================================================
@File:      metrics.py
@Function:  Full metric suite for the per-subject best-epoch results.

Everything is computed from (y_true, y_prob) so the same numbers can be
reproduced offline from the saved .npz probability dumps.
=================================================
'''
import csv
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def _safe(fn, default=np.nan):
    """Metrics raise on degenerate inputs (a class missing from y_true, ...)."""
    try:
        return fn()
    except Exception:
        return default


def per_class_specificity(cm):
    """One-vs-rest specificity for every class."""
    total = cm.sum()
    out = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp          # predicted i but not actually i
        fn = cm[i, :].sum() - tp          # actually i but not predicted i
        tn = total - tp - fp - fn
        out.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    return out


def compute_metrics(y_true, y_prob, n_classes=None):
    """All metrics we can get from the label vector and the softmax output.

    ``y_true``: (N,) int labels.  ``y_prob``: (N, C) softmax probabilities.
    Returns a JSON-serialisable dict; values that cannot be computed are NaN.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim == 1:                       # (N,) -> single positive score
        y_prob = np.stack([1.0 - y_prob, y_prob], axis=1)
    if n_classes is None:
        n_classes = y_prob.shape[1]

    y_pred = y_prob.argmax(axis=1)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    binary = n_classes == 2
    if binary:
        pos = y_prob[:, 1]
        auc = _safe(lambda: roc_auc_score(y_true, pos))
        # Youden's J at the best threshold, plus the accuracy there
        fpr, tpr, thr = _safe(lambda: _roc_curve(y_true, pos), default=(None, None, None))
        if fpr is not None:
            j = tpr - fpr
            k = int(np.argmax(j))
            youden_j, youden_thr = float(j[k]), float(thr[k])
            acc_at_youden = float(((pos >= youden_thr).astype(int) == y_true).mean())
        else:
            youden_j = youden_thr = acc_at_youden = np.nan
    else:
        auc = _safe(lambda: roc_auc_score(y_true, y_prob, multi_class='ovr',
                                          average='macro', labels=labels))
        youden_j = youden_thr = acc_at_youden = np.nan

    spec = per_class_specificity(cm)
    sens = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    m = {
        # ---- dataset / bookkeeping ----
        'n_test': int(len(y_true)),
        'n_classes': int(n_classes),
        'class_support': [int((y_true == c).sum()) for c in labels],
        # ---- headline ----
        'acc': float(accuracy_score(y_true, y_pred) * 100),
        'bca': float(balanced_accuracy_score(y_true, y_pred) * 100),  # balanced accuracy
        'kappa': float(cohen_kappa_score(y_true, y_pred)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        # ---- F1 ----
        'f1_macro': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'f1_micro': float(f1_score(y_true, y_pred, labels=labels, average='micro', zero_division=0)),
        # ---- precision / recall ----
        'precision_macro': float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)),
        # ---- specificity (one-vs-rest), macro averaged ----
        'specificity_macro': float(np.nanmean(spec)),
        'specificity_weighted': float(np.nansum(np.array(spec) * cm.sum(axis=1) / max(cm.sum(), 1))),
        # ---- AUC / ranking ----
        'auc': float(auc) if auc == auc else np.nan,   # NaN-safe
        'youden_j': youden_j,
        'youden_threshold': youden_thr,
        'acc_at_youden': acc_at_youden,
        # ---- probabilistic calibration ----
        'log_loss': float(_safe(lambda: log_loss(y_true, y_prob, labels=labels))),
        'brier': float(_safe(lambda: np.mean(np.sum((y_prob - np.eye(n_classes)[y_true]) ** 2, axis=1)))),
        # ---- detail ----
        'confusion_matrix': cm.tolist(),          # rows = true, cols = predicted
        'per_class': {
            'precision': [float(v) for v in precision_score(y_true, y_pred, labels=labels,
                                                            average=None, zero_division=0)],
            'recall_sensitivity': [float(v) for v in sens],
            'specificity': [float(v) for v in spec],
            'f1': [float(v) for v in f1_score(y_true, y_pred, labels=labels,
                                              average=None, zero_division=0)],
            'support': [int((y_true == c).sum()) for c in labels],
        },
    }
    if binary:
        m['sensitivity'] = m['recall_macro']
        m['specificity'] = float(spec[1])          # the positive class
    return m


def _roc_curve(y_true, y_score):
    from sklearn.metrics import roc_curve
    return roc_curve(y_true, y_score)


def metrics_to_row(meta, metrics):
    """Flatten ``meta`` + ``metrics`` into an ordered dict for a CSV row.

    Sequence/mapping values (confusion matrix, per-class tables, class support)
    stay out of the CSV -- they live in the JSON detail file instead.
    """
    row = dict(meta)
    for k, v in metrics.items():
        if isinstance(v, (list, tuple, dict)):
            continue
        row[k] = v
    return row


def format_metrics(metrics, keys=('acc', 'bca', 'f1_macro', 'f1_weighted', 'auc',
                                  'precision_macro', 'recall_macro', 'specificity_macro',
                                  'kappa', 'mcc')):
    """One-line human-readable summary for the run log."""
    return ' | '.join(f'{k}={metrics.get(k, float("nan")):.4f}' for k in keys)


def dumps(metrics):
    return json.dumps(metrics, indent=2, default=_json_default)


# ======================================================================
#  Persistence
# ======================================================================

def run_paths(args):
    """Metric file paths for one run, derived from the run's log file name.

    Creates ``<result_dir>/metrics/`` and returns the CSV (one row per
    subject), the JSON (full detail incl. confusion matrices) and the
    probability directory.
    """
    base = os.path.splitext(os.path.basename(args.out_file.name))[0]
    d = os.path.join(args.result_dir, 'metrics')
    os.makedirs(d, exist_ok=True)
    return {
        'dir': d,
        'csv': os.path.join(d, base + '_subjects.csv'),
        'json': os.path.join(d, base + '_detail.json'),
        'prob_dir': os.path.join(d, base + '_probs'),
    }


def append_csv(path, row):
    """Append a row, creating the file and its header from the first row."""
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        with open(path) as f:
            header = f.readline().rstrip('\n').split(',')
    else:
        header = list(row.keys())
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerow(row)
    return path


def update_json(path, key, entry):
    """Add/replace one entry (``key``) in the JSON detail file."""
    data = {}
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[key] = entry
    with open(path, 'w') as f:
        f.write(dumps(data))
    return path


def save_probabilities(prob_dir, key, prob, true):
    """Keep the raw softmax output so metrics/ROC can be recomputed offline."""
    os.makedirs(prob_dir, exist_ok=True)
    path = os.path.join(prob_dir, key + '.npz')
    prob = np.asarray(prob)
    true = np.asarray(true)
    np.savez(path, prob=prob, true=true, pred=prob.argmax(axis=1))
    return path


def read_csv(path):
    """Read the per-subject CSV back as a list of dicts (str -> float)."""
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def summarise_csv(path, keys=('acc', 'bca', 'f1_macro', 'f1_weighted', 'auc',
                              'precision_macro', 'recall_macro', 'specificity_macro',
                              'kappa', 'mcc')):
    """Mean/std of the chosen columns across subjects (the 'final result')."""
    rows = read_csv(path)
    out = {}
    for k in keys:
        vals = []
        for r in rows:
            try:
                v = float(r.get(k, 'nan'))
            except (TypeError, ValueError):
                v = float('nan')
            if v == v:                      # skip NaN
                vals.append(v)
        if vals:
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not JSON serialisable: {type(o)}')

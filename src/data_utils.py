from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch


@dataclass
class FeaturePreprocessor:
    mode: str = "l2"
    clip_z: float = 3.0
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.mode in {"zscore_l2", "zscore_clip_minmax", "legacy_zscore_clip_minmax_l2"}:
            self.mean_ = x.mean(axis=0, keepdims=True)
            # The archived PyTorch pipeline used torch.std's sample standard
            # deviation (correction=1); preserve that only for the legacy mode.
            ddof = 1 if self.mode == "legacy_zscore_clip_minmax_l2" and len(x) > 1 else 0
            self.std_ = x.std(axis=0, keepdims=True, ddof=ddof)
            self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform_np(self, x):
        x = np.asarray(x, dtype=np.float32).copy()
        if self.mode == "none":
            return x
        if self.mode == "l2":
            n = np.linalg.norm(x, axis=1, keepdims=True)
            n[n < 1e-8] = 1.0
            return x / n
        if self.mode == "zscore_l2":
            z = (x - self.mean_) / self.std_
            n = np.linalg.norm(z, axis=1, keepdims=True)
            n[n < 1e-8] = 1.0
            return z / n
        if self.mode == "zscore_clip_minmax":
            z = (x - self.mean_) / self.std_
            z = np.clip(z, -self.clip_z, self.clip_z)
            return (z + self.clip_z) / (2.0 * self.clip_z)
        if self.mode == "legacy_zscore_clip_minmax_l2":
            # Exact preprocessing topology used by the archived 50-segment runs:
            # feature-wise z-score -> clip to [-3, 3] -> [0, 1] -> row-wise L2.
            z = (x - self.mean_) / self.std_
            z = np.clip(z, -self.clip_z, self.clip_z)
            scaled = (z + self.clip_z) / (2.0 * self.clip_z)
            n = np.linalg.norm(scaled, axis=1, keepdims=True)
            n[n < 1e-8] = 1.0
            return scaled / n
        raise ValueError(f"Unknown preprocessing mode: {self.mode}")

    def transform(self, x, device=None):
        return torch.as_tensor(self.transform_np(x), dtype=torch.float32, device=device)


def load_radar_csv(path, start=0, size=None, feature_cols=("RF", "PW", "PA", "DOA")):
    df = pd.read_csv(path)
    if size is None:
        part = df.iloc[start:].copy()
    else:
        part = df.iloc[start:start + size].copy()
    x = part[list(feature_cols)].values.astype(np.float32)
    y = part["Label"].values.astype(np.int64) if "Label" in part.columns else None
    return part.reset_index(drop=True), x, y


def stratified_holdout(df, test_ratio=0.3, seed=42, label_col="Label"):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, g in df.groupby(label_col):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return (
        df.loc[sorted(train_idx)].reset_index(drop=True),
        df.loc[sorted(test_idx)].reset_index(drop=True),
    )


def drop_pulses(df, rate, seed=42, label_col="Label"):
    """Stratified pulse loss: remove approximately rate of pulses from each real emitter."""
    if rate <= 0:
        return df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep = np.ones(len(df), dtype=bool)
    if label_col in df.columns:
        for _, g in df.groupby(label_col):
            idx = g.index.to_numpy()
            n_drop = int(round(rate * len(idx)))
            if n_drop:
                keep[rng.choice(idx, size=min(n_drop, len(idx)), replace=False)] = False
    else:
        n_drop = int(round(rate * len(df)))
        keep[rng.choice(len(df), size=n_drop, replace=False)] = False
    return df.loc[keep].reset_index(drop=True)


def add_spurious_pulses(df, rate, feature_cols=("RF", "PW", "PA", "DOA"), seed=42,
                        label_col="Label", spurious_label=-1, expansion=0.15):
    if rate <= 0:
        return df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n = max(1, int(round(rate * len(df))))
    fake = {}
    for c in feature_cols:
        lo, hi = float(df[c].min()), float(df[c].max())
        span = max(hi - lo, 1e-6)
        fake[c] = rng.uniform(lo - expansion * span, hi + expansion * span, size=n)
    if "TOA" in df.columns:
        toa_lo, toa_hi = float(df["TOA"].min()), float(df["TOA"].max())
        fake["TOA"] = rng.uniform(toa_lo, toa_hi, size=n)
    fake[label_col] = np.full(n, spurious_label, dtype=int)
    fake_df = pd.DataFrame(fake)
    for c in df.columns:
        if c not in fake_df.columns:
            fake_df[c] = np.nan
    fake_df = fake_df[df.columns]
    out = pd.concat([df, fake_df], ignore_index=True)
    if "TOA" in out.columns:
        out = out.sort_values("TOA").reset_index(drop=True)
    else:
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def add_measurement_noise(df, rel_std, feature_cols=("RF", "PW", "PA", "DOA"), seed=42):
    if rel_std <= 0:
        return df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in feature_cols:
        s = float(df[c].std())
        out[c] = df[c].to_numpy() + rng.normal(0.0, rel_std * max(s, 1e-8), size=len(df))
    return out.reset_index(drop=True)


def add_linear_drift(df, rel_std, feature_cols=("RF", "PW", "PA", "DOA"), direction="alternating"):
    if rel_std <= 0:
        return df.copy().reset_index(drop=True)
    out = df.copy()
    ramp = np.linspace(-1.0, 1.0, len(df), dtype=np.float32)
    for i, c in enumerate(feature_cols):
        sign = -1.0 if (direction == "alternating" and i % 2) else 1.0
        out[c] = df[c].to_numpy() + sign * rel_std * float(df[c].std()) * ramp
    return out.reset_index(drop=True)


def apply_combined_corruption(df, miss=0.0, spurious=0.0, noise=0.0, drift=0.0,
                              feature_cols=("RF", "PW", "PA", "DOA"), seed=42):
    out = drop_pulses(df, miss, seed=seed)
    out = add_measurement_noise(out, noise, feature_cols=feature_cols, seed=seed + 1)
    out = add_linear_drift(out, drift, feature_cols=feature_cols)
    out = add_spurious_pulses(out, spurious, feature_cols=feature_cols, seed=seed + 2)
    return out

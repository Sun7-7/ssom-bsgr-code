from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import math
import numpy as np
import torch

from data_utils import FeaturePreprocessor
from energy import EnergyStats


def _normalize_rows(w: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.norm(w, dim=1, keepdim=True)
    n = torch.where(n > 1e-8, n, torch.ones_like(n))
    return w / n


@dataclass
class ModelConfig:
    output_shape: tuple[int, int] = (20, 20)
    iterations: int = 600
    batch_size: int = 64
    T: int = 20
    V_th: float = 0.6
    U_rest: float = 0.5
    tau_m: float = 10.0
    R: float = 0.3
    lr0: float = 1.0
    sigma0: float = 1.0
    V_unit: float = 0.4
    trace_decay: float = 0.75
    trace_offset: float = 0.5
    theta_plus: float = 0.05
    theta_decay: float = 0.005
    stdp_dim: int = 20
    stdp_lr: float = 0.01
    stdp_wmax: float = 1.0
    encoding_type: str = "stdp"  # stdp/direct/poisson/latency
    preprocess: str = "l2"
    competition: str = "wta1"  # wta1/threshold_topk/threshold_topk_sparse
    topk: int = 3
    topk_margin: float = 0.08
    legacy_skip_t0: bool = True
    seed: int = 42
    reseed_on_init: bool = True
    som_update_rule: str = "legacy_batch_mean"  # legacy_batch_mean/mass_normalized/winner_only
    som_weight_constraint: str = "none"  # none/clamp01
    stdp_update_rule: str = "legacy_trace"  # legacy_trace/multiplicative_trace/oja


@dataclass
class TrainSchedule:
    """Explicit encoder/SSOM coupling schedule.

    random_frozen:
        No STDP updates. This is the exact semantic meaning of legacy
        stdp_pretrain_iteration=0 when the encoder is subsequently frozen.
    short_pretrain:
        A small number of STDP-only updates, then encoder freeze and SSOM training.
    joint:
        STDP and SSOM update on the same batches throughout training.
    ssom_warmup_weak_joint:
        First anchor the SSOM under a fixed encoder, then enable weak/decaying STDP
        for a short window, and freeze it again for the remaining SSOM iterations.
    """
    mode: str = "random_frozen"
    pretrain_steps: int = 0
    ssom_warmup_steps: int = 100
    joint_steps: int = 50
    joint_lr_scale: float = 0.10
    joint_lr_final_scale: float = 0.01
    encoder_update_every: int = 1


class STDPSSOM:
    def __init__(self, input_dim: int, config: ModelConfig | None = None, device=None):
        self.cfg = config or ModelConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.cfg.reseed_on_init:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        self.input_dim = int(input_dim)
        self.nn = int(self.cfg.output_shape[0] * self.cfg.output_shape[1])
        self.som_input_dim = self.cfg.stdp_dim if self.cfg.encoding_type == "stdp" else self.input_dim
        self.preprocessor = FeaturePreprocessor(self.cfg.preprocess)

        self.W = torch.rand((self.nn, self.som_input_dim, self.cfg.T), device=self.device)
        if self.cfg.encoding_type == "stdp":
            self.W_stdp = _normalize_rows(torch.rand((self.cfg.stdp_dim, self.input_dim), device=self.device))
            self.theta_stdp = torch.zeros(self.cfg.stdp_dim, device=self.device)
        else:
            self.W_stdp = None
            self.theta_stdp = None

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(self.cfg.output_shape[0], device=self.device),
                torch.arange(self.cfg.output_shape[1], device=self.device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        self.coords = coords.float()
        self.history: list[dict] = []
        self.training_stats: dict = {}

    # ------------------------------------------------------------------
    # Input preprocessing and encoders
    # ------------------------------------------------------------------
    def fit_preprocessor(self, x) -> None:
        x_np = np.asarray(x, dtype=np.float32)
        if self.preprocessor.mode == "legacy_zscore_clip_minmax_l2":
            # Match the archived implementation's CUDA float32 reductions.
            xt = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
            self.preprocessor.mean_ = xt.mean(dim=0, keepdim=True).cpu().numpy()
            self.preprocessor.std_ = xt.std(dim=0, keepdim=True, correction=1).cpu().numpy()
            self.preprocessor.std_[self.preprocessor.std_ < 1e-8] = 1.0
        else:
            self.preprocessor.fit(x_np)

    def _prepare(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float32)
        if self.preprocessor.mode == "legacy_zscore_clip_minmax_l2":
            xt = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
            mean = torch.as_tensor(self.preprocessor.mean_, dtype=torch.float32, device=self.device)
            std = torch.as_tensor(self.preprocessor.std_, dtype=torch.float32, device=self.device)
            z = torch.clamp((xt - mean) / std, -self.preprocessor.clip_z, self.preprocessor.clip_z)
            scaled = (z + self.preprocessor.clip_z) / (2.0 * self.preprocessor.clip_z)
            norms = torch.linalg.norm(scaled, dim=1, keepdim=True)
            norms = torch.where(norms > 1e-8, norms, torch.ones_like(norms))
            return scaled / norms
        return self.preprocessor.transform(x_np, device=self.device)

    def normalize_stdp_weights(self) -> None:
        with torch.no_grad():
            self.W_stdp.clamp_(0.0, self.cfg.stdp_wmax)
            self.W_stdp[:] = _normalize_rows(self.W_stdp)

    def _competition(self, u: torch.Tensor, adaptive_th: torch.Tensor) -> torch.Tensor:
        """Threshold-aware competition.

        Version A: threshold_topk
            Select at most K units only from neurons that actually exceed their
            own adaptive thresholds. No supra-threshold unit -> no spike.

        Version B: threshold_topk_sparse
            Same threshold constraint, then retain only candidates whose membrane
            potential is within topk_margin of the strongest eligible unit. This
            gives a variable 1..K spike count and avoids forced Top-K firing.
        """
        eligible = u >= adaptive_th
        B, M = u.shape
        out = torch.zeros_like(u)
        mode = self.cfg.competition

        if mode == "wta1":
            masked = u.masked_fill(~eligible, -torch.inf)
            has = eligible.any(dim=1)
            if has.any():
                idx = masked.argmax(dim=1)
                rows = torch.nonzero(has, as_tuple=True)[0]
                out[rows, idx[rows]] = 1.0
            return out

        if mode not in {"threshold_topk", "threshold_topk_sparse"}:
            raise ValueError(f"Unknown competition mode: {mode}")

        k = min(max(int(self.cfg.topk), 1), M)
        masked = u.masked_fill(~eligible, -torch.inf)
        vals, idx = torch.topk(masked, k=k, dim=1)
        finite = torch.isfinite(vals)

        if mode == "threshold_topk_sparse":
            top1 = vals[:, :1]
            # The strongest eligible neuron always survives. Lower-ranked units
            # must both exceed threshold and remain close to the strongest response.
            finite = finite & ((top1 - vals) <= self.cfg.topk_margin)

        rows = torch.arange(B, device=u.device).unsqueeze(1).expand_as(idx)
        out[rows[finite], idx[finite]] = 1.0
        return out

    def _stdp_encode(self, batch: torch.Tensor, update_stdp=False, lr_scale=1.0,
                     meter: Optional[EnergyStats] = None) -> torch.Tensor:
        x = batch
        B, D = x.shape
        M, T = self.cfg.stdp_dim, self.cfg.T
        spikes = torch.zeros((B, M, T), device=self.device)
        u = torch.zeros((B, M), device=self.device)
        pre_trace = torch.zeros((B, D), device=self.device)
        post_trace = torch.zeros((B, M), device=self.device)

        # During inference W_stdp is fixed, so the projection can be computed once.
        fixed_current = None
        if not update_stdp:
            fixed_current = torch.einsum("md,bd->bm", self.W_stdp, x)
            if meter is not None:
                meter.encoder_projection_macs += B * M * D

        for _t in range(T):
            if update_stdp:
                pre_trace = self.cfg.trace_decay * pre_trace + x
                input_current = torch.einsum("md,bd->bm", self.W_stdp, x)
            else:
                input_current = fixed_current

            du = (-u) / self.cfg.tau_m + self.cfg.R * input_current
            u = u + du
            adaptive_th = self.cfg.V_th + self.theta_stdp.unsqueeze(0)
            post = self._competition(u, adaptive_th)
            spikes[:, :, _t] = post
            u = torch.where(post.bool(), u - self.cfg.U_rest, u)

            if meter is not None:
                meter.encoder_state_updates += B * M
                meter.encoder_output_spikes += int(post.sum().item())

            if update_stdp:
                post_trace = self.cfg.trace_decay * post_trace + post
                ltp = torch.einsum("bm,bd->md", post, pre_trace) / B
                ltd = torch.einsum("bm,bd->md", post_trace, x) / B
                if self.cfg.stdp_update_rule == "legacy_trace":
                    delta = (self.cfg.stdp_lr * lr_scale) * (
                        self.cfg.V_unit * ltp - self.cfg.trace_offset * ltd
                    )
                elif self.cfg.stdp_update_rule == "multiplicative_trace":
                    # Weight-dependent STDP prevents additive saturation at the
                    # hard bounds while retaining the archived trace dynamics.
                    wfrac = torch.clamp(self.W_stdp / max(self.cfg.stdp_wmax, 1e-8), 0.0, 1.0)
                    delta = (self.cfg.stdp_lr * lr_scale) * (
                        self.cfg.V_unit * ltp * (1.0 - wfrac)
                        - self.cfg.trace_offset * ltd * wfrac
                    )
                elif self.cfg.stdp_update_rule == "oja":
                    # Local Hebbian/Oja alternative using only presynaptic input
                    # and postsynaptic spikes; row normalization is still applied.
                    hebb = torch.einsum("bm,bd->md", post, x) / B
                    post_activity = post.mean(dim=0, keepdim=False).unsqueeze(1)
                    delta = (self.cfg.stdp_lr * lr_scale) * (
                        hebb - post_activity * self.W_stdp
                    )
                else:
                    raise ValueError(f"Unknown STDP update rule: {self.cfg.stdp_update_rule}")
                with torch.no_grad():
                    self.W_stdp += delta
                    self.normalize_stdp_weights()
                    self.theta_stdp = (
                        (1.0 - self.cfg.theta_decay) * self.theta_stdp
                        + self.cfg.theta_plus * post.mean(dim=0)
                    )
        return spikes

    def _poisson_encode(self, batch: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(batch, 0.0, 1.0).unsqueeze(-1).expand(-1, -1, self.cfg.T)
        return (torch.rand_like(p) < p).float()

    def _latency_encode(self, batch: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(batch, 0.0, 1.0)
        B, D = x.shape
        spikes = torch.zeros((B, D, self.cfg.T), device=self.device)
        fire_steps = torch.round((1.0 - x) * (self.cfg.T - 1)).long()
        valid = x > 0
        if valid.any():
            bi, di = torch.nonzero(valid, as_tuple=True)
            ti = fire_steps[bi, di]
            spikes[bi, di, ti] = 1.0
        return spikes

    def encode_batch(self, batch: torch.Tensor, update_stdp=False, lr_scale=1.0,
                     meter: Optional[EnergyStats] = None) -> torch.Tensor:
        et = self.cfg.encoding_type
        if et == "stdp":
            return self._stdp_encode(batch, update_stdp=update_stdp, lr_scale=lr_scale, meter=meter)
        if et == "direct":
            return batch.unsqueeze(-1).repeat(1, 1, self.cfg.T)
        if et == "poisson":
            return self._poisson_encode(batch)
        if et == "latency":
            return self._latency_encode(batch)
        raise ValueError(et)

    # ------------------------------------------------------------------
    # SSOM dynamics
    # ------------------------------------------------------------------
    def _som_forward(self, x_spikes: torch.Tensor, meter: Optional[EnergyStats] = None):
        B, D, T = x_spikes.shape
        result_I = torch.einsum("ndt,bdt->bnt", self.W, x_spikes)
        U = torch.zeros((B, self.nn), device=self.device)
        out = torch.zeros((B, self.nn, T), device=self.device)
        start_t = 1 if self.cfg.legacy_skip_t0 else 0

        if meter is not None:
            meter.ssom_event_accums += int(x_spikes.sum().item()) * self.nn

        for t in range(start_t, T):
            U = U + (-U) / self.cfg.tau_m + self.cfg.R * result_I[:, :, t]
            s = (U >= self.cfg.V_th).float()
            out[:, :, t] = s
            U = torch.where(s.bool(), U - self.cfg.U_rest, U)
            if meter is not None:
                meter.ssom_state_updates += B * self.nn
                meter.ssom_output_spikes += int(s.sum().item())
        firing_rate = out.mean(dim=2)
        if meter is not None:
            meter.bmu_comparisons += B * max(self.nn - 1, 0)
        return firing_rate, out

    def _som_update(self, x_spikes: torch.Tensor, step: int):
        firing_rate, _ = self._som_forward(x_spikes)
        bmu = firing_rate.argmax(dim=1)
        eta = self.cfg.lr0 * math.exp(-step / max(self.cfg.iterations, 1))
        sigma = self.cfg.sigma0 * math.exp(-step / max(self.cfg.iterations, 1))
        sigma = max(sigma, 1e-5)

        bmu_coords = self.coords[bmu]
        dist2 = ((self.coords.unsqueeze(0) - bmu_coords.unsqueeze(1)) ** 2).sum(dim=2)
        h = torch.exp(-dist2 / (2.0 * sigma * sigma))
        if self.cfg.som_update_rule == "winner_only":
            h = torch.zeros_like(h).scatter_(1, bmu.unsqueeze(1), 1.0)
        diff = x_spikes.unsqueeze(1) - self.W.unsqueeze(0)
        weighted = h.unsqueeze(-1).unsqueeze(-1) * diff
        if self.cfg.som_update_rule in {"legacy_batch_mean", "winner_only"}:
            delta = eta * weighted.mean(dim=0)
        elif self.cfg.som_update_rule == "mass_normalized":
            mass = h.sum(dim=0)
            delta = eta * weighted.sum(dim=0) / mass.clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1)
            # Do not move nodes that received only negligible Gaussian support.
            delta = torch.where((mass >= 1e-3).unsqueeze(-1).unsqueeze(-1), delta, torch.zeros_like(delta))
        else:
            raise ValueError(f"Unknown SOM update rule: {self.cfg.som_update_rule}")
        with torch.no_grad():
            self.W += delta
            if self.cfg.som_weight_constraint == "clamp01":
                self.W.clamp_(0.0, 1.0)
            elif self.cfg.som_weight_constraint != "none":
                raise ValueError(f"Unknown SOM weight constraint: {self.cfg.som_weight_constraint}")
        return bmu, float(eta), float(sigma)

    @torch.no_grad()
    def extract_spike_features(self, x, batch_size=256):
        """Return cached spike-domain embeddings without raw-feature leakage.

        `encoder_counts` and `ssom_rates` are computed solely from emitted spikes;
        they can be reused by multiple consolidation backends without retraining.
        """
        xt = self._prepare(x)
        enc_rows, rate_rows, bmu_rows = [], [], []
        for s in range(0, len(xt), batch_size):
            spikes = self.encode_batch(xt[s:s + batch_size], update_stdp=False)
            rates, _ = self._som_forward(spikes)
            enc_rows.append(spikes.sum(dim=2).cpu())
            rate_rows.append(rates.cpu())
            bmu_rows.append(rates.argmax(dim=1).cpu())
        return {
            "encoder_counts": torch.cat(enc_rows).numpy().astype(np.float32),
            "ssom_rates": torch.cat(rate_rows).numpy().astype(np.float32),
            "bmu": torch.cat(bmu_rows).numpy().astype(np.int32),
        }

    def _sample_batch(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.randint(0, len(x), (self.cfg.batch_size,), device=self.device)
        return x[idx]

    def _pretrain_stdp(self, x: torch.Tensor, steps: int):
        if self.cfg.encoding_type != "stdp" or steps <= 0:
            return 0, 0.0
        cumulative_step_drift = 0.0
        for _ in range(int(steps)):
            batch = self._sample_batch(x)
            before = self.W_stdp.detach().clone()
            self.encode_batch(batch, update_stdp=True, lr_scale=1.0)
            cumulative_step_drift += float(
                torch.linalg.norm(self.W_stdp - before).item()
                / (torch.linalg.norm(before).item() + 1e-12)
            )
        return int(steps), cumulative_step_drift

    def _schedule_update(self, schedule: TrainSchedule, step: int):
        mode = schedule.mode
        if mode in {"random_frozen", "short_pretrain"}:
            return False, 0.0
        if mode == "joint":
            do = (step % max(schedule.encoder_update_every, 1)) == 0
            return do, float(schedule.joint_lr_scale)
        if mode == "ssom_warmup_weak_joint":
            a = int(schedule.ssom_warmup_steps)
            b = a + int(schedule.joint_steps)
            if a <= step < b and (step - a) % max(schedule.encoder_update_every, 1) == 0:
                p = (step - a) / max(schedule.joint_steps - 1, 1)
                start = max(schedule.joint_lr_scale, 1e-8)
                end = max(schedule.joint_lr_final_scale, 1e-8)
                scale = math.exp(math.log(start) * (1.0 - p) + math.log(end) * p)
                return True, float(scale)
            return False, 0.0
        raise ValueError(f"Unknown schedule mode: {mode}")

    def fit(self, x, schedule: TrainSchedule | None = None, log_every=25):
        schedule = schedule or TrainSchedule()
        self.fit_preprocessor(x)
        xt = self._prepare(x)
        encoder_start = None if self.W_stdp is None else self.W_stdp.detach().clone()
        cumulative_step_drift = 0.0
        encoder_updates = 0

        # Important semantic distinction: pretrain_steps=0 + freeze means a random
        # fixed spiking projection, not an STDP-learned encoder.
        if schedule.mode == "short_pretrain":
            encoder_updates, cumulative_step_drift = self._pretrain_stdp(xt, schedule.pretrain_steps)

        self.history = []
        for step in range(self.cfg.iterations):
            batch = self._sample_batch(xt)
            update_stdp, lr_scale = self._schedule_update(schedule, step)
            if update_stdp and self.W_stdp is not None:
                before = self.W_stdp.detach().clone()
            x_spikes = self.encode_batch(batch, update_stdp=update_stdp, lr_scale=lr_scale)
            if update_stdp and self.W_stdp is not None:
                denom = torch.linalg.norm(before).item() + 1e-12
                drift = torch.linalg.norm(self.W_stdp - before).item() / denom
                cumulative_step_drift += drift
                encoder_updates += 1
            else:
                drift = 0.0
            bmu, eta, sigma = self._som_update(x_spikes, step)

            if step % max(log_every, 1) == 0 or step == self.cfg.iterations - 1:
                self.history.append({
                    "step": step,
                    "eta": eta,
                    "sigma": sigma,
                    "encoder_update": int(update_stdp),
                    "stdp_lr_scale": lr_scale,
                    "encoder_drift": drift,
                    "encoder_spike_rate": float(x_spikes.mean().item()),
                    "active_bmu": int(torch.unique(bmu).numel()),
                })
        if encoder_start is None:
            endpoint_drift = 0.0
        else:
            endpoint_drift = float(
                torch.linalg.norm(self.W_stdp - encoder_start).item()
                / (torch.linalg.norm(encoder_start).item() + 1e-12)
            )
        self.training_stats = {
            "encoder_updates": int(encoder_updates),
            "cumulative_step_relative_drift": float(cumulative_step_drift),
            "endpoint_relative_drift": endpoint_drift,
        }
        return self

    # ------------------------------------------------------------------
    # Prediction, confidence, and adaptive horizon
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_bmu(self, x, batch_size=256, return_details=False,
                    mac_pj=4.6, ac_pj=0.9):
        xt = self._prepare(x)
        winners, top1s, margins = [], [], []
        total_meter = EnergyStats()
        for s in range(0, len(xt), batch_size):
            b = xt[s:s + batch_size]
            meter = EnergyStats(samples=len(b))
            spikes = self.encode_batch(b, update_stdp=False, meter=meter)
            fr, _ = self._som_forward(spikes, meter=meter)
            vals, idx = torch.topk(fr, k=min(2, self.nn), dim=1)
            winners.append(idx[:, 0])
            top1s.append(vals[:, 0])
            if vals.shape[1] > 1:
                margins.append(vals[:, 0] - vals[:, 1])
            else:
                margins.append(vals[:, 0])
            total_meter.add(meter)
        winners = torch.cat(winners)
        if not return_details:
            return winners
        return winners, {
            "top1_rate": torch.cat(top1s),
            "margin": torch.cat(margins),
            "energy": total_meter.per_sample(mac_pj=mac_pj, ac_pj=ac_pj),
            "energy_stats": total_meter,
        }

    @torch.no_grad()
    def predict_adaptive(self, x, batch_size=256, min_t=5, margin_rate=0.08,
                         min_output_spikes=2, patience=2, mac_pj=4.6, ac_pj=0.9):
        """Confidence-gated adaptive-T inference.

        A sample stops when the cumulative top-1 BMU remains unchanged for
        `patience` checks, has emitted at least `min_output_spikes`, and the
        cumulative top1-top2 spike-count gap divided by elapsed time exceeds
        `margin_rate`.
        """
        if self.cfg.encoding_type != "stdp":
            raise NotImplementedError("Adaptive-T script currently targets the STDP encoder.")

        xt = self._prepare(x)
        all_winners, all_t, all_margin = [], [], []
        total = EnergyStats()
        M, D, T = self.cfg.stdp_dim, self.input_dim, self.cfg.T
        start_t = 1 if self.cfg.legacy_skip_t0 else 0

        for s0 in range(0, len(xt), batch_size):
            xb = xt[s0:s0 + batch_size]
            B = len(xb)
            meter = EnergyStats(samples=B)
            # Fixed encoder projection is computed once per sample.
            enc_current = torch.einsum("md,bd->bm", self.W_stdp, xb)
            meter.encoder_projection_macs += B * M * D
            enc_u = torch.zeros((B, M), device=self.device)
            som_u = torch.zeros((B, self.nn), device=self.device)
            counts = torch.zeros((B, self.nn), device=self.device)
            active = torch.ones(B, dtype=torch.bool, device=self.device)
            stop_t = torch.full((B,), T, dtype=torch.long, device=self.device)
            stable = torch.zeros(B, dtype=torch.long, device=self.device)
            prev_w = torch.full((B,), -1, dtype=torch.long, device=self.device)
            final_margin = torch.zeros(B, device=self.device)

            for t in range(T):
                if not active.any():
                    break
                ai = torch.nonzero(active, as_tuple=True)[0]
                enc_u[ai] = enc_u[ai] + (-enc_u[ai]) / self.cfg.tau_m + self.cfg.R * enc_current[ai]
                adaptive_th = self.cfg.V_th + self.theta_stdp.unsqueeze(0)
                enc_sp = self._competition(enc_u[ai], adaptive_th.expand(len(ai), -1))
                enc_u[ai] = torch.where(enc_sp.bool(), enc_u[ai] - self.cfg.U_rest, enc_u[ai])
                meter.encoder_state_updates += len(ai) * M
                meter.encoder_output_spikes += int(enc_sp.sum().item())

                if t < start_t:
                    continue
                # Event-driven SSOM accumulation for this time index.
                It = torch.einsum("nd,bd->bn", self.W[:, :, t], enc_sp)
                meter.ssom_event_accums += int(enc_sp.sum().item()) * self.nn
                som_u[ai] = som_u[ai] + (-som_u[ai]) / self.cfg.tau_m + self.cfg.R * It
                ss = (som_u[ai] >= self.cfg.V_th).float()
                som_u[ai] = torch.where(ss.bool(), som_u[ai] - self.cfg.U_rest, som_u[ai])
                counts[ai] += ss
                meter.ssom_state_updates += len(ai) * self.nn
                meter.ssom_output_spikes += int(ss.sum().item())

                vals, idx = torch.topk(counts[ai], k=min(2, self.nn), dim=1)
                w = idx[:, 0]
                gap = vals[:, 0] - vals[:, 1] if vals.shape[1] > 1 else vals[:, 0]
                gap_rate = gap / float(t + 1)
                total_sp = vals[:, 0]
                same = w == prev_w[ai]
                stable[ai] = torch.where(same, stable[ai] + 1, torch.ones_like(stable[ai]))
                prev_w[ai] = w
                final_margin[ai] = gap_rate

                if t + 1 >= min_t:
                    done_local = (
                        (total_sp >= min_output_spikes)
                        & (gap_rate >= margin_rate)
                        & (stable[ai] >= patience)
                    )
                    if done_local.any():
                        di = ai[done_local]
                        stop_t[di] = t + 1
                        active[di] = False

            vals, idx = torch.topk(counts, k=min(2, self.nn), dim=1)
            meter.bmu_comparisons += B * max(self.nn - 1, 0)
            all_winners.append(idx[:, 0])
            all_t.append(stop_t)
            all_margin.append(final_margin)
            total.add(meter)

        return torch.cat(all_winners), {
            "stop_t": torch.cat(all_t),
            "margin_rate": torch.cat(all_margin),
            "energy": total.per_sample(mac_pj=mac_pj, ac_pj=ac_pj),
            "energy_stats": total,
        }

    def state_dict(self):
        return {
            "config": asdict(self.cfg),
            "input_dim": self.input_dim,
            "W": self.W.detach().cpu(),
            "W_stdp": None if self.W_stdp is None else self.W_stdp.detach().cpu(),
            "theta_stdp": None if self.theta_stdp is None else self.theta_stdp.detach().cpu(),
            "preprocess_mode": self.preprocessor.mode,
            "preprocess_mean": None if self.preprocessor.mean_ is None else torch.as_tensor(self.preprocessor.mean_),
            "preprocess_std": None if self.preprocessor.std_ is None else torch.as_tensor(self.preprocessor.std_),
            "history": self.history,
            "training_stats": self.training_stats,
        }

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device=None):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Compatibility with checkpoints produced by early project-local
            # revisions that stored NumPy preprocessing arrays.
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ckpt["config"])
        model = cls(ckpt["input_dim"], config=cfg, device=device)
        model.W = ckpt["W"].to(model.device)
        if ckpt["W_stdp"] is not None:
            model.W_stdp = ckpt["W_stdp"].to(model.device)
            model.theta_stdp = ckpt["theta_stdp"].to(model.device)
        model.preprocessor.mode = ckpt["preprocess_mode"]
        mean = ckpt["preprocess_mean"]
        std = ckpt["preprocess_std"]
        model.preprocessor.mean_ = None if mean is None else np.asarray(mean.cpu())
        model.preprocessor.std_ = None if std is None else np.asarray(std.cpu())
        model.history = ckpt.get("history", [])
        model.training_stats = ckpt.get("training_stats", {})
        return model

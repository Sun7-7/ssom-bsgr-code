from dataclasses import dataclass, asdict


@dataclass
class EnergyStats:
    """Operation-count proxy for event-driven inference.

    The implementation itself can still use dense PyTorch tensor operations for speed.
    These counters describe the corresponding event-driven algorithmic workload.
    """
    samples: int = 0
    encoder_projection_macs: int = 0
    encoder_state_updates: int = 0
    encoder_output_spikes: int = 0
    ssom_event_accums: int = 0
    ssom_state_updates: int = 0
    ssom_output_spikes: int = 0
    bmu_comparisons: int = 0

    def add(self, other: "EnergyStats") -> None:
        for key in asdict(self):
            setattr(self, key, getattr(self, key) + getattr(other, key))

    @property
    def ac_ops(self) -> int:
        return (
            self.encoder_state_updates
            + self.ssom_event_accums
            + self.ssom_state_updates
            + self.ssom_output_spikes
            + self.bmu_comparisons
        )

    def estimate_energy_uj(self, mac_pj: float = 4.6, ac_pj: float = 0.9) -> float:
        # 1 microjoule = 1e6 picojoules.
        total_pj = self.encoder_projection_macs * mac_pj + self.ac_ops * ac_pj
        return total_pj / 1e6

    def per_sample(self, mac_pj: float = 4.6, ac_pj: float = 0.9) -> dict:
        n = max(self.samples, 1)
        return {
            "samples": self.samples,
            "encoder_projection_macs": self.encoder_projection_macs / n,
            "encoder_state_updates": self.encoder_state_updates / n,
            "encoder_output_spikes": self.encoder_output_spikes / n,
            "ssom_event_accums": self.ssom_event_accums / n,
            "ssom_state_updates": self.ssom_state_updates / n,
            "ssom_output_spikes": self.ssom_output_spikes / n,
            "bmu_comparisons": self.bmu_comparisons / n,
            "ac_ops": self.ac_ops / n,
            "energy_uj": self.estimate_energy_uj(mac_pj=mac_pj, ac_pj=ac_pj) / n,
        }

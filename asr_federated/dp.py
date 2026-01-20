from typing import Dict, Any
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

class DPAccountant: 
    def __init__(
        self,
        sample_rate: float,
        noise_multiplier: float,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self._steps = 0
        
        self._accountant = RDPAccountant()
    
    def step(self, num_steps: int = 1) -> None:
        self._steps += num_steps
        
        if self._accountant is not None:
            for _ in range(num_steps):
                self._accountant.step(
                    noise_multiplier=self.noise_multiplier,
                    sample_rate=self.sample_rate,
                )
    
    def get_epsilon(self) -> float:
        if self._accountant is not None:
            try:
                return self._accountant.get_epsilon(delta=self.delta)
            except Exception as e:
                print(f"[DPAccountant] Error computing epsilon: {e}")
                return self.fallback_epsilon()
        else:
            return self.fallback_epsilon()
    
    def fallback_epsilon(self) -> float:
        
        import math
        if self._steps == 0 or self.noise_multiplier == 0:
            return float('inf')
        
        T = self._steps * self.sample_rate  
        sigma = self.noise_multiplier
        log_delta = math.log(1.0 / self.delta)
        
        epsilon = math.sqrt(2 * T * log_delta) / sigma
        return epsilon
    
    def get_privacy_spent(self) -> Dict[str, Any]:
        
        return {
            "epsilon": self.get_epsilon(),
            "delta": self.delta,
            "steps": self._steps,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "max_grad_norm": self.max_grad_norm,
        }
    
    def reset(self) -> None:
        self._steps = 0
        self._accountant = RDPAccountant()
    
    @staticmethod
    def compute_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        epochs: int,
        steps_per_epoch: int,
    ) -> float:
        
        return get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            epochs=epochs,
        )


def estimate_epsilon(
    steps: int,
    sample_rate: float,
    noise_multiplier: float,
    delta: float = 1e-5,
) -> float:
    
    accountant = DPAccountant(
        sample_rate=sample_rate,
        noise_multiplier=noise_multiplier,
        delta=delta,
    )
    accountant.step(steps)

    return accountant.get_epsilon()

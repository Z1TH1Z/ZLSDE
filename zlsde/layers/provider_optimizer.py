"""Feature 6: Dynamic Provider Cost-Quality Optimizer.

Wraps the FallbackChainManager to dynamically route label-generation
requests to the provider that currently offers the best cost/quality
trade-off, using a multi-armed bandit (UCB1) strategy.

Patent-relevant novelty:
- UCB1-based dynamic provider routing for LLM labelling
- Online cost-quality Pareto estimation per provider
- Exploration-exploitation balance for discovering cheaper high-quality routes
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from zlsde.models.data_models import PipelineConfig
from zlsde.providers.base import LLMProvider
from zlsde.providers.exceptions import ProviderError

logger = logging.getLogger(__name__)


@dataclass
class ProviderArm:
    """Bandit arm tracking for a single provider."""
    name: str
    pulls: int = 0
    total_reward: float = 0.0
    total_cost_ms: float = 0.0
    successes: int = 0
    failures: int = 0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0

    @property
    def avg_cost_ms(self) -> float:
        return self.total_cost_ms / self.pulls if self.pulls > 0 else float("inf")

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.0


class DynamicProviderOptimizer:
    """UCB1-based dynamic provider selection for label generation."""

    def __init__(self, config: PipelineConfig, providers: List[LLMProvider]):
        self.enabled = config.enable_provider_optimization
        self.quality_weight = config.provider_quality_weight
        self.exploration_rate = config.provider_exploration_rate

        # Filter to available providers
        self.providers = {p.get_provider_name(): p for p in providers if p.is_available()}
        self.arms: Dict[str, ProviderArm] = {
            name: ProviderArm(name=name) for name in self.providers
        }
        self._total_pulls = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_label(self, prompt: str, max_tokens: int = 20) -> str:
        """Select best provider via UCB1 and generate a label.

        Falls back to round-robin if optimisation is disabled.
        """
        if not self.enabled or not self.providers:
            return self._fallback_generate(prompt, max_tokens)

        selected = self._select_provider()
        provider = self.providers[selected]
        arm = self.arms[selected]

        start = time.time()
        try:
            label = provider.generate_label(prompt, max_tokens)
            elapsed_ms = (time.time() - start) * 1000

            # Reward = quality_weight * success_indicator + (1-quality_weight) * speed_bonus
            speed_bonus = max(0.0, 1.0 - elapsed_ms / 5000.0)
            reward = self.quality_weight * 1.0 + (1 - self.quality_weight) * speed_bonus

            arm.pulls += 1
            arm.successes += 1
            arm.total_reward += reward
            arm.total_cost_ms += elapsed_ms
            self._total_pulls += 1

            logger.debug(f"Provider {selected}: label='{label}', reward={reward:.3f}, cost={elapsed_ms:.0f}ms")
            return label

        except (ProviderError, Exception) as e:
            elapsed_ms = (time.time() - start) * 1000
            arm.pulls += 1
            arm.failures += 1
            arm.total_reward += 0.0
            arm.total_cost_ms += elapsed_ms
            self._total_pulls += 1

            logger.warning(f"Provider {selected} failed: {e}")

            # Try remaining providers in order
            for name, prov in self.providers.items():
                if name == selected:
                    continue
                try:
                    label = prov.generate_label(prompt, max_tokens)
                    self.arms[name].pulls += 1
                    self.arms[name].successes += 1
                    self._total_pulls += 1
                    return label
                except Exception:
                    self.arms[name].pulls += 1
                    self.arms[name].failures += 1
                    self._total_pulls += 1
                    continue

            raise RuntimeError("All providers failed in optimiser") from e

    def get_statistics(self) -> Dict[str, dict]:
        """Return per-provider statistics."""
        return {
            name: {
                "pulls": arm.pulls,
                "avg_reward": arm.avg_reward,
                "avg_cost_ms": arm.avg_cost_ms,
                "success_rate": arm.success_rate,
            }
            for name, arm in self.arms.items()
        }

    # ------------------------------------------------------------------
    # UCB1 selection
    # ------------------------------------------------------------------

    def _select_provider(self) -> str:
        """Select provider using UCB1 with exploration bonus."""
        # Ensure every arm is pulled at least once
        for name, arm in self.arms.items():
            if arm.pulls == 0:
                return name

        # UCB1 formula
        best_name = ""
        best_score = -float("inf")

        for name, arm in self.arms.items():
            exploitation = arm.avg_reward
            exploration = self.exploration_rate * math.sqrt(
                2 * math.log(self._total_pulls) / arm.pulls
            )
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_generate(self, prompt: str, max_tokens: int) -> str:
        """Simple sequential fallback when optimiser is disabled."""
        for name, provider in self.providers.items():
            try:
                return provider.generate_label(prompt, max_tokens)
            except Exception:
                continue
        raise RuntimeError("All providers failed (fallback mode)")

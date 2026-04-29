import torch
import torch.nn as nn
import torch.nn.functional as F


class TraditionalHopfieldNetwork(nn.Module):
    """Discrete Hopfield baseline for binary character memories.

    This class is intentionally simple: it stores bipolarized template patterns
    with the Hebbian rule, iteratively denoises a polluted query, then assigns
    the label of the closest stored template. It is used as a thesis baseline
    against the modern continuous Hopfield retrieval layer.
    """

    def __init__(
        self,
        memory_matrix,
        labels,
        threshold_offset=0.10,
        steps=6,
        center_patterns=True,
        retrieval_weight=0.35,
    ):
        super().__init__()
        if memory_matrix.dim() != 2:
            raise ValueError("memory_matrix must have shape [num_templates, feature_dim].")
        if labels.dim() != 1 or labels.shape[0] != memory_matrix.shape[0]:
            raise ValueError("labels must have shape [num_templates].")

        patterns = self._to_bipolar(memory_matrix.float(), threshold_offset, center=center_patterns)
        feature_dim = patterns.shape[1]
        weights = patterns.t().matmul(patterns) / max(1, feature_dim)
        weights.fill_diagonal_(0.0)

        self.register_buffer("patterns", patterns)
        self.register_buffer("patterns_norm", F.normalize(patterns, dim=-1))
        self.register_buffer("labels", labels.long())
        self.register_buffer("weights", weights)
        self.threshold_offset = float(threshold_offset)
        self.steps = int(steps)
        self.center_patterns = bool(center_patterns)
        self.retrieval_weight = float(retrieval_weight)

    @staticmethod
    def _to_bipolar(x, threshold_offset=0.10, center=True):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        binary = x > (mean + float(threshold_offset) * std)
        bipolar = binary.float() * 2.0 - 1.0
        if center:
            bipolar = bipolar - bipolar.mean(dim=-1, keepdim=True)
        return bipolar

    def retrieve(self, q, steps=None):
        state = self._to_bipolar(q.float(), self.threshold_offset, center=self.center_patterns).to(self.weights.device)
        step_count = self.steps if steps is None else int(steps)
        for _ in range(max(1, step_count)):
            updated = state.matmul(self.weights)
            state = torch.where(updated >= 0.0, torch.ones_like(state), -torch.ones_like(state))
            if self.center_patterns:
                state = state - state.mean(dim=-1, keepdim=True)
        return state

    def forward(self, q, steps=None):
        initial = self._to_bipolar(q.float(), self.threshold_offset, center=self.center_patterns).to(self.weights.device)
        restored = self.retrieve(q, steps=steps)
        initial_scores = F.normalize(initial, dim=-1).matmul(self.patterns_norm.t())
        restored_scores = F.normalize(restored, dim=-1).matmul(self.patterns_norm.t())
        scores = (1.0 - self.retrieval_weight) * initial_scores + self.retrieval_weight * restored_scores
        indices = torch.argmax(scores, dim=-1)
        return self.labels[indices], restored, scores

    def predict(self, q):
        pred, _, _ = self.forward(q)
        return pred

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

    def __init__(self, memory_matrix, labels, threshold_offset=0.10, steps=8):
        super().__init__()
        if memory_matrix.dim() != 2:
            raise ValueError("memory_matrix must have shape [num_templates, feature_dim].")
        if labels.dim() != 1 or labels.shape[0] != memory_matrix.shape[0]:
            raise ValueError("labels must have shape [num_templates].")

        patterns = self._to_bipolar(memory_matrix.float(), threshold_offset)
        feature_dim = patterns.shape[1]
        weights = patterns.t().matmul(patterns) / max(1, feature_dim)
        weights.fill_diagonal_(0.0)

        self.register_buffer("patterns", patterns)
        self.register_buffer("labels", labels.long())
        self.register_buffer("weights", weights)
        self.threshold_offset = float(threshold_offset)
        self.steps = int(steps)

    @staticmethod
    def _to_bipolar(x, threshold_offset=0.10):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        binary = x > (mean + float(threshold_offset) * std)
        return binary.float() * 2.0 - 1.0

    def retrieve(self, q, steps=None):
        state = self._to_bipolar(q.float(), self.threshold_offset).to(self.weights.device)
        step_count = self.steps if steps is None else int(steps)
        for _ in range(max(1, step_count)):
            updated = state.matmul(self.weights)
            state = torch.where(updated >= 0.0, torch.ones_like(state), -torch.ones_like(state))
        return state

    def forward(self, q, steps=None):
        restored = self.retrieve(q, steps=steps)
        scores = F.normalize(restored, dim=-1).matmul(F.normalize(self.patterns, dim=-1).t())
        indices = torch.argmax(scores, dim=-1)
        return self.labels[indices], restored, scores

    def predict(self, q):
        pred, _, _ = self.forward(q)
        return pred

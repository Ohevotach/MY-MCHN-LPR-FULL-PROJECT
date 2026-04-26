import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernHopfieldNetwork(nn.Module):
    """Continuous modern Hopfield retrieval layer.

    For the default dot metric, the update is equivalent to scaled attention:
        z = softmax(beta * q @ M.T) @ M
    """

    def __init__(self, memory_matrix, beta=25.0, metric="dot", normalize=True):
        super().__init__()
        if memory_matrix.dim() != 2:
            raise ValueError("memory_matrix must have shape [num_templates, feature_dim].")
        self.register_buffer("M", memory_matrix.float())
        self.beta = float(beta)
        self.metric = metric
        self.normalize = normalize
        self.num_templates = memory_matrix.shape[0]

    def _memory_for_similarity(self):
        if self.normalize and self.metric == "dot":
            return F.normalize(self.M, p=2, dim=-1)
        return self.M

    def _query_for_similarity(self, q):
        q = q.float()
        if self.normalize and self.metric == "dot":
            return F.normalize(q, p=2, dim=-1)
        return q

    def compute_similarity(self, q):
        q_sim = self._query_for_similarity(q)
        m_sim = self._memory_for_similarity()

        if self.metric == "dot":
            return torch.matmul(q_sim, m_sim.t())
        if self.metric == "manhattan":
            return -torch.cdist(q_sim, m_sim, p=1.0)
        if self.metric == "euclidean":
            return -torch.cdist(q_sim, m_sim, p=2.0)
        raise ValueError(f"Unsupported metric: {self.metric}")

    def forward(self, q, template_mask=None, return_attention=False):
        if q.dim() == 1:
            q = q.unsqueeze(0)

        sim_scores = self.compute_similarity(q)
        if template_mask is not None:
            mask = template_mask.to(device=sim_scores.device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            sim_scores = sim_scores.masked_fill(~mask, -1e9)

        attention_weights = F.softmax(self.beta * sim_scores, dim=-1)
        retrieved = torch.matmul(attention_weights, self.M)
        predicted_indices = torch.argmax(attention_weights, dim=-1)

        if return_attention:
            return retrieved, predicted_indices, attention_weights
        return retrieved, predicted_indices


if __name__ == "__main__":
    num_templates, feature_dim, batch_size = 34, 2048, 8
    memory = torch.rand((num_templates, feature_dim))
    model = ModernHopfieldNetwork(memory_matrix=memory, beta=25.0, metric="dot")
    query = torch.rand((batch_size, feature_dim))
    with torch.no_grad():
        reconstructed, preds, weights = model(query, return_attention=True)
    print("reconstructed:", tuple(reconstructed.shape))
    print("preds:", tuple(preds.shape))
    print("weights:", tuple(weights.shape))

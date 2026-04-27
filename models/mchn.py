import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernHopfieldNetwork(nn.Module):
    """Continuous modern Hopfield retrieval layer.

    For the default dot metric, the update is equivalent to scaled attention:
        z = softmax(beta * q @ M.T) @ M
    """

    def __init__(
        self,
        memory_matrix,
        beta=40.0,
        metric="dot",
        normalize=True,
        feature_mode="binary",
        image_shape=(64, 32),
    ):
        super().__init__()
        if memory_matrix.dim() != 2:
            raise ValueError("memory_matrix must have shape [num_templates, feature_dim].")
        self.register_buffer("M", memory_matrix.float())
        self.beta = float(beta)
        self.metric = metric
        self.normalize = normalize
        self.feature_mode = feature_mode
        self.image_shape = tuple(image_shape)
        self.num_templates = memory_matrix.shape[0]

    def _feature_transform(self, x):
        x = x.float()
        if self.feature_mode == "raw":
            return x
        if self.feature_mode == "centered":
            return x - x.mean(dim=-1, keepdim=True)
        if self.feature_mode == "bipolar":
            return x * 2.0 - 1.0
        if self.feature_mode == "binary":
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
            threshold = mean + 0.15 * std
            return (x > threshold).float() * 2.0 - 1.0
        if self.feature_mode == "binary_centered":
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
            binary = (x > mean + 0.15 * std).float()
            return binary - binary.mean(dim=-1, keepdim=True)
        if self.feature_mode == "profile":
            return self._profile_feature_transform(x)
        if self.feature_mode in {"shape", "hybrid", "hybrid_shape"}:
            return self._shape_feature_transform(x)
        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def _profile_feature_transform(self, x):
        """Stroke-profile features for real plate fonts.

        Pixel templates are sensitive to anti-aliasing, screws and tiny missing
        strokes. These features emphasize coarse stroke layout: projections,
        low-resolution occupancy, edge density and regional ink distribution.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        img_h, img_w = self.image_shape
        expected_dim = img_h * img_w
        if x.shape[-1] != expected_dim:
            return x

        img = x.view(-1, 1, img_h, img_w).float()
        mean = img.flatten(1).mean(dim=-1, keepdim=True).view(-1, 1, 1, 1)
        std = img.flatten(1).std(dim=-1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1, 1)
        binary = (img > mean + 0.10 * std).float()
        centered = (img - mean) / std

        coarse = F.avg_pool2d(binary, kernel_size=(8, 4), stride=(8, 4)).flatten(1)
        fine = F.avg_pool2d(binary, kernel_size=(4, 4), stride=(4, 4)).flatten(1)
        row_profile = binary.mean(dim=3).flatten(1)
        col_profile = binary.mean(dim=2).flatten(1)

        left = binary[:, :, :, : img_w // 2].mean(dim=3).flatten(1)
        right = binary[:, :, :, img_w // 2 :].mean(dim=3).flatten(1)
        top = binary[:, :, : img_h // 2, :].mean(dim=2).flatten(1)
        bottom = binary[:, :, img_h // 2 :, :].mean(dim=2).flatten(1)

        sobel_x = binary.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = binary.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        edges = F.conv2d(binary, sobel_x, padding=1).abs() + F.conv2d(binary, sobel_y, padding=1).abs()
        edge_coarse = F.avg_pool2d(edges, kernel_size=(8, 4), stride=(8, 4)).flatten(1)
        gray_coarse = F.avg_pool2d(centered, kernel_size=(8, 4), stride=(8, 4)).flatten(1)
        ink = binary.flatten(1).mean(dim=-1, keepdim=True)

        return torch.cat(
            [
                1.30 * coarse,
                0.75 * fine,
                1.10 * row_profile,
                1.10 * col_profile,
                0.55 * left,
                0.55 * right,
                0.45 * top,
                0.45 * bottom,
                0.45 * edge_coarse,
                0.25 * gray_coarse,
                ink,
            ],
            dim=-1,
        )

    def _shape_feature_transform(self, x):
        """Build pollution-tolerant features from a 64x32 character image.

        The original pixel vector is brittle under dirt, missing strokes and
        small affine changes. This representation keeps coarse appearance while
        adding row/column stroke projections and Sobel edge summaries.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        img_h, img_w = self.image_shape
        expected_dim = img_h * img_w
        if x.shape[-1] != expected_dim:
            return x

        img = x.view(-1, 1, img_h, img_w).float()
        mean = img.flatten(1).mean(dim=-1, keepdim=True).view(-1, 1, 1, 1)
        std = img.flatten(1).std(dim=-1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1, 1)
        binary = (img > mean + 0.15 * std).float()
        centered = (img - mean) / std

        pooled_binary = F.avg_pool2d(binary, kernel_size=(4, 4), stride=(4, 4)).flatten(1)
        pooled_centered = F.avg_pool2d(centered, kernel_size=(4, 4), stride=(4, 4)).flatten(1)
        row_projection = binary.mean(dim=3).flatten(1)
        col_projection = binary.mean(dim=2).flatten(1)

        sobel_x = binary.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
        sobel_y = binary.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
        grad_x = F.conv2d(binary, sobel_x, padding=1).abs()
        grad_y = F.conv2d(binary, sobel_y, padding=1).abs()
        edge_features = torch.cat(
            [
                F.avg_pool2d(grad_x, kernel_size=(8, 4), stride=(8, 4)).flatten(1),
                F.avg_pool2d(grad_y, kernel_size=(8, 4), stride=(8, 4)).flatten(1),
            ],
            dim=-1,
        )

        ink = binary.flatten(1).mean(dim=-1, keepdim=True)
        return torch.cat(
            [
                pooled_binary,
                0.5 * pooled_centered,
                row_projection,
                col_projection,
                0.35 * edge_features,
                ink,
            ],
            dim=-1,
        )

    def _memory_for_similarity(self):
        memory = self._feature_transform(self.M)
        if self.normalize and self.metric == "dot":
            return F.normalize(memory, p=2, dim=-1)
        return memory

    def _query_for_similarity(self, q):
        q = self._feature_transform(q)
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

    def forward(self, q, template_mask=None, return_attention=False, return_similarity=False):
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

        if return_attention and return_similarity:
            return retrieved, predicted_indices, attention_weights, sim_scores
        if return_attention:
            return retrieved, predicted_indices, attention_weights
        if return_similarity:
            return retrieved, predicted_indices, sim_scores
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

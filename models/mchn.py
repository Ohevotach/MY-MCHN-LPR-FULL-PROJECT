
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernHopfieldNetwork(nn.Module):
    """
    现代连续 Hopfield 网络 (MCHN) 核心实现。
    """
    def __init__(self, memory_matrix, beta=100.0, metric='manhattan'):
        super(ModernHopfieldNetwork, self).__init__()
        self.register_buffer('M', memory_matrix)
        self.beta = beta
        self.metric = metric
        self.num_templates = memory_matrix.shape[0]

    def _compute_similarity(self, q):
        if self.metric == 'manhattan':
            distances = torch.cdist(q, self.M, p=1.0)
            sim_scores = -distances
        elif self.metric == 'euclidean':
            distances = torch.cdist(q, self.M, p=2.0)
            sim_scores = -distances
        elif self.metric == 'dot':
            sim_scores = torch.matmul(q, self.M.t())
        else:
            raise ValueError(f"不支持的距离度量: {self.metric}")
        return sim_scores

    def _separation(self, sim_scores):
        attention_weights = F.softmax(self.beta * sim_scores, dim=-1)
        return attention_weights

    def forward(self, q, template_mask=None):
        """
        Args:
            q: 查询向量
            template_mask (torch.BoolTensor, optional): 先验注意力掩码，True为有效，False为屏蔽。
        """
        # 1. 计算所有模板的相似度
        sim_scores = self._compute_similarity(q)
        
        # 🌟 核心修复：施加位置掩码，把不允许搜索的模板相似度强行拉到 -1e9
        if template_mask is not None:
            sim_scores = sim_scores.masked_fill(~template_mask, -1e9)
            
        # 2. 通过 Softmax 分离注意力
        attention_weights = self._separation(sim_scores)
        
        # 3. 投影重构
        z = torch.matmul(attention_weights, self.M)
        
        # 4. 获取预测结果
        predicted_indices = torch.argmax(attention_weights, dim=-1)
        
        return z, predicted_indices

if __name__ == "__main__":
    print("🚀 开始测试现代 Hopfield 网络 (MCHN) 核心模块...")
    num_classes, feature_dim, batch_size = 34, 2048, 128
    dummy_memory = torch.rand((num_classes, feature_dim))
    mchn = ModernHopfieldNetwork(memory_matrix=dummy_memory, beta=100.0, metric='manhattan')
    dummy_q = torch.rand((batch_size, feature_dim))
    with torch.no_grad():
        reconstructed_z, preds = mchn(dummy_q)
    print("✨ MCHN 前向传播测试完美通过！")
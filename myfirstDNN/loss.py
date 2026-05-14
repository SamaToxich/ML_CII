import torch
import torch.nn as nn


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.
    L = -log( exp(s * cos(θ_y + m)) /
             (exp(s * cos(θ_y + m)) + Σ_{j≠y} exp(s * cos(θ_j))) )
    """
    def __init__(self, num_classes, embedding_size=512, s=16.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.weight)

    def set_scale(self, new_s):
        """Ручное увеличение scale"""
        self.s = new_s

    def forward(self, embeddings, labels):
        # Нормализуем веса
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        # Косинус между эмбеддингом и всеми классами
        cosine = torch.matmul(embeddings, norm_weight.t())

        # Косинус правильного класса
        cosine_of_target = cosine[torch.arange(embeddings.size(0)), labels]

        # Угол (в float32 для стабильности acos)
        theta = torch.acos(torch.clamp(cosine_of_target.float(), -1.0 + 1e-7, 1.0 - 1e-7))

        # Прибавляем margin
        cosine_with_margin = torch.cos(theta + self.m)

        # Вставляем обратно
        cosine[torch.arange(embeddings.size(0)), labels] = cosine_with_margin.to(cosine.dtype)

        # Scale + CrossEntropy
        output = cosine * self.s
        loss = nn.functional.cross_entropy(output, labels)

        return loss
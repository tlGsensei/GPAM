import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from models.attetion import MultiHeadAttention, ScaledDotProductAttention
import random
import numpy as np


class MRM(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, hidden_size=1536, dot=False):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot
        # 初始化 Multi-Head Attention 层
        self.n_head = 8
        self.hidden_size = hidden_size
        self.d_model = self.hidden_size  # 使用隐藏层维度作为模型的输入和输出维度
        self.d_k = self.d_v = self.hidden_size // self.n_head  # 设置每个头的维度
        self.multi_head_attention = MultiHeadAttention(self.n_head, self.d_model, self.d_k, self.d_v)
        self.temperature = 40
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.temperature)
        self.num_views = 4
        self.prompt_length = 10
        # 初始化soft prompts为可学习参数
        self.soft_prompts = nn.Parameter(torch.randn(self.num_views, self.prompt_length, hidden_size))
        self.lamda = 0.1
        self.alfa = 1
        self.beta = 5
        self.v = 0.1
        self.w = 0.2
        self.m = 1
        self.num_extra_samples = 20

    # 欧式距离
    def __dist__(self, x, y, dim):
        # 确保x和y位于相同的设备
        device = x.device
        y = y.to(device)
        if self.dot:
            return (x * y).sum(dim)  # (B, N, N)
        else:
            return torch.sqrt((torch.pow(x - y, 2)).sum(-1))

    # 马氏距离
    def __mahalanobis_dist__(self, x, y, covariance_matrix):
        diff = x - y
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, inv_covariance_matrix), diff.T))
        return mahalanobis_dist

    # 用support set求loss
    def __mrm_loss__(self, pos_distances, neg_distances, radius, margin, N, K, batch_size):
        pos_loss = 1.0 / self.alfa * torch.log(1 + torch.sum(torch.exp(self.alfa * (pos_distances.view(-1) - radius.repeat(int((batch_size * K + self.prompt_length) * N / N))))))
        neg_loss = 1.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (neg_distances.view(-1) - radius.repeat(int((batch_size * K + self.prompt_length) * (N - 1) + self.num_extra_samples)) + torch.mean(margin)))))
        # loss = 1.0 / N * torch.sum(self.lamda * radius ** 2 / margin ** 2 + pos_loss + neg_loss)
        loss = 1.0 / N / batch_size * torch.sum(pos_loss + neg_loss)
        return loss

    # 用query set求loss
    def __mrm_loss2__(self, pos_distances, neg_distances, radius, margin, N, K, batch_size):
        pos_loss = 1.0 / self.alfa * torch.log(1 + torch.sum(torch.exp(self.alfa * (pos_distances.view(-1) - radius.repeat(int(batch_size * K * N / N))))))
        neg_loss = 1.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (neg_distances.view(-1) - radius.repeat(int((batch_size + self.prompt_length) * K * (N - 1))) + torch.mean(margin)))))
        # loss = 1.0 / N * torch.sum(self.lamda * radius * radius + pos_loss + neg_loss)
        loss = 1.0 / N / batch_size * torch.sum(pos_loss + neg_loss)
        return loss

    # 计算4个view各自的权重
    def __softmax_linear__(self, mean, variance):
        view = []
        for i in range(4):
            view.append(torch.cat([mean[i], variance[i]], dim=1))
        all_view = torch.stack(view, dim=0)
        all_view = all_view.view(4, -1, self.hidden_size)
        _, scores, _ = self.scaled_dot_product_attention(all_view, all_view, all_view)
        attention_weights = scores
        # 对每个维度的注意力分布进行加权平均，得到每个维度的总体注意力分数
        total_attention_weights = torch.mean(attention_weights, dim=2)
        total_attention_weights = torch.mean(total_attention_weights, dim=1)
        # 对总体注意力分数再进行 softmax，得到最终的权重值
        weights = torch.softmax(total_attention_weights, dim=0)
        return weights

    def __sample_negatives__(self, support_anchor, radius, dim):
        num_extra_samples = 20
        max_attempts = 10000  # 设定一个最大尝试次数
        attempts = 0
        negatives = []
        while len(negatives) < num_extra_samples and attempts < max_attempts:
            attempts += 1
            sample = torch.randn(dim)  # 生成单个样本
            if all(self.__dist__(sample, center, 0) > r + self.m for center, r in zip(support_anchor, radius)):
                negatives.append(sample)
        # 如果尝试次数用尽后仍未获得足够的样本，则从已有样本中重复选取
        while len(negatives) < num_extra_samples:
            sample = negatives[random.randint(0, len(negatives) - 1)]
            negatives.append(sample)

        return torch.stack(negatives)

    def forward(self, support, query, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query_emb = self.sentence_encoder(query)  # (B * total_Q, D)
        hidden_size = support_emb.size(-1)
        support_drop = self.drop(support_emb)
        query_drop = self.drop(query_emb)
        support = support_drop.view(-1, N, K, hidden_size)  # (B, N, K, D)
        query = query_drop.view(-1, int(total_Q), hidden_size)  # (B, total_Q, D)
        # MultiHeadAttention进行更新
        batch_size, len_support, _, _ = support.size()
        _, len_query, _ = query.size()
        # 重塑支持集和查询集的尺寸，以便传入 Multi-Head Attention
        support = support.view(-1, len_support, hidden_size)
        query = query.view(-1, len_query, hidden_size)
        support = self.multi_head_attention(support, support, support)
        query = self.multi_head_attention(query, query, query)
        # 恢复支持集和查询集的尺寸
        support = support.view(batch_size, N, -1, hidden_size)
        query = query.view(batch_size, -1, hidden_size)
        # 在K不为零的情况下对向量进行处理
        support = support.permute(0, 2, 1, 3)  # 将 support 重排为 (batch_size, K, N, D)
        support = support.contiguous()  # 使得内存布局连续，以便正确地应用 view
        support = support.view(-1, N, 1, hidden_size)

        support_anchor = torch.mean(support.squeeze(2), 0)

        # 计算support set所有对，用于计算radius和margin，以及给出pred
        support_resize = support.expand(-1, -1, support_anchor.size(0), -1)
        support_anchor_resize = support_anchor.unsqueeze(0).unsqueeze(0).expand(support_resize.size())
        all_pair = self.__dist__(support_resize, support_anchor_resize, 3)  # (B, N, N)
        pos_results = []
        neg_results = []
        for pos_class_idx in range(N):
            # 计算正例对(anchor, +)
            pos_pair = all_pair[:, pos_class_idx, pos_class_idx]  # (B, 1)
            # 计算负例对(anchor, -)
            neg_class_indices = [i for i in range(N) if i != pos_class_idx]
            neg_pair = all_pair[:, pos_class_idx, neg_class_indices]  # (B, N - 1)
            pos_results.append(pos_pair)
            neg_results.append(neg_pair)
        pos_pairs = torch.stack(pos_results, dim=1)  # (B, N)
        neg_pairs = torch.stack(neg_results, dim=1)  # (B, N, N - 1)
        pos_distances = pos_pairs
        neg_distances = neg_pairs
        pos_distances_resize = pos_distances.permute(1, 0)
        neg_distances_resize = neg_distances.permute(1, 0, 2).reshape(N, -1)
        # 20个元素的话是找第3小的值
        # radius = torch.kthvalue(pos_distances_resize, int((1 - self.v) * pos_distances_resize.size(-1)) + 1, dim=1).values
        radius = torch.kthvalue(neg_distances_resize, int(self.v * neg_distances_resize.size(-1)) + 1, dim=1).values
        margin = torch.kthvalue(neg_distances_resize, int(self.w * neg_distances_resize.size(-1)) + 1, dim=1).values - radius

        # 计算query到每个圆心的距离
        query_resize = query.unsqueeze(2).expand(-1, -1, support_anchor.size(0), -1)
        support_anchor_resize = support_anchor.unsqueeze(0).unsqueeze(0).expand(query_resize.size())
        query_distances = self.__dist__(query_resize, support_anchor_resize, 3)  # (B, total_Q, N)
        # 将query到每个圆心的距离与对应半径进行比较，判断在哪个圆的范围内
        pred_list = []
        for i in range(batch_size):
            for j in range(int(total_Q)):
                flag_k = 1
                minn = 0
                t = []
                s = []
                for k in range(N):
                    t.append(query_distances[i, j, k])
                    s.append(radius[k] - query_distances[i, j, k])
                    if query_distances[i, j, k] <= radius[k] + margin[k]:
                        flag_k = 1
                if flag_k == 0:
                    pred_list.append(torch.tensor(N))
                else:
                    softmax_probabilities = F.softmax(torch.tensor(t).neg(), dim=0)
                    minn = torch.argmax(softmax_probabilities).item()
                    pred_list.append(torch.tensor(minn))
        pred = torch.stack(pred_list)
        self.m = torch.mean(margin)

        # mrm_loss = self.__mrm_loss__(pos_distances, neg_distances, radius, margin, N, K, batch_size)
        mrm_loss = 0
        # mrm_loss = self.__mrm_loss2__(pos_distances_q, neg_distances_q, radius, margin, N, K, batch_size)

        loss = mrm_loss

        return loss, pred


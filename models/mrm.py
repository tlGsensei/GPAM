import sys

sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from models.attetion import MultiHeadAttention
import numpy as np

class MRM(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, hidden_size=1536, dot=False):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot
        # 初始化 Multi-Head Attention 层
        self.n_head = 8
        hidden_size = 1536
        self.d_model = hidden_size  # 使用隐藏层维度作为模型的输入和输出维度
        self.d_k = self.d_v = hidden_size // self.n_head  # 设置每个头的维度
        self.multi_head_attention = MultiHeadAttention(self.n_head, self.d_model, self.d_k, self.d_v)
        self.lamda = 1
        self.alfa = 1
        self.beta = 3
        # self.lamda = 0.001
        # self.alfa = 0.01
        # self.beta = 0.03
        # 设置分位数函数参数v = 0.1
        self.v = 0.1
        # self.v = nn.Parameter(torch.tensor(0.1))  # 假设初始值为0.1
        # 先设置margin值为m = 1
        self.m = 1
        # 将 self.m 设置为可学习的参数
        # self.m = nn.Parameter(torch.tensor(0.8))  # 假设初始值为1.0

        # self.threshold = 0.05
        self.threshold = -3000

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)  # (B, N, N)
        else:
            return torch.sqrt((torch.pow(x - y, 2)).sum(-1))
            # return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    # MRM loss
    def __mrm_loss__(self, pos_distances, neg_distances, cov_matrices, anchors, N, K, batch_size):
        total_loss = 0
        neg_distances = neg_distances.permute(1, 0, 2).reshape(-1, N)
        for c in range(N):
            # 获取第 c 类的锚点 anchor_c 和协方差矩阵
            anchor_c = anchors[c].to(dtype=torch.float32)
            cov_matrix_c = cov_matrices[c].to(dtype=torch.float32)
            sigma_c = torch.mean(torch.sqrt(torch.diag(cov_matrix_c)))
            reg_term = 1e-1 * torch.eye(cov_matrix_c.size(0), device=cov_matrix_c.device, dtype=cov_matrix_c.dtype)
            cov_matrix_c += reg_term
            # 对于每个正类样本
            pos_loss = 0
            for x_plus in pos_distances[c]:
                x_plus = x_plus.to(dtype=torch.float32)
                # d = torch.sqrt((x_plus - anchor_c).T @ torch.inverse(cov_matrix_c) @ (x_plus - anchor_c))
                # 使用 Cholesky 分解计算逆矩阵
                # L = torch.cholesky(cov_matrix_c)
                # y = torch.cholesky_solve((x_plus - anchor_c).unsqueeze(1), L)
                # d = torch.sqrt(((x_plus - anchor_c).unsqueeze(1).T @ y).squeeze())
                # 维度太高，尝试使用简单的欧氏距离
                d = torch.norm(x_plus)
                pos_loss += self.alfa * (d / sigma_c - 1)
            # 对于每个负类样本
            neg_loss = 0
            for x_minus in neg_distances[c]:
                x_minus = x_minus.to(dtype=torch.float32)
                # d = torch.sqrt((x_minus - anchor_c).T @ torch.inverse(cov_matrix_c) @ (x_minus - anchor_c))
                # 使用 Cholesky 分解计算逆矩阵
                # L = torch.cholesky(cov_matrix_c)
                # y = torch.cholesky_solve((x_minus - anchor_c).unsqueeze(1), L)
                # d = torch.sqrt(((x_plus - anchor_c).unsqueeze(1).T @ y).squeeze())
                # 维度太高，尝试使用简单的欧氏距离
                d = torch.norm(x_minus)
                neg_loss += -self.beta * (d / sigma_c - 1)
            # 计算类别 c 的总损失
            class_loss = self.lamda * sigma_c ** 2 + (1 / self.alfa) * pos_loss + (1 / self.beta) * neg_loss
            total_loss += class_loss
        # 计算平均损失
        mrm_loss = total_loss / N / batch_size
        return mrm_loss

    def __mrm_loss2__(self, pos_distances, neg_distances, cov_matrices, radius, N, K, batch_size):
        sigma = 0
        for c in range(N):
            cov_matrix_c = cov_matrices[c].to(dtype=torch.float32)
            sigma_c = torch.mean(torch.sqrt(torch.diag(cov_matrix_c)))
            sigma += sigma_c
        pos_loss = 1.0 / self.alfa * torch.log(1 + torch.sum(torch.exp(self.alfa * (pos_distances.view(-1) - radius.repeat(batch_size).repeat(K)))))
        neg_loss = 1.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (neg_distances.view(-1) - (radius.repeat(batch_size).repeat(K).repeat(N - 1) + self.m)))))
        # loss = 1.0 / N * torch.sum(self.lamda * sigma * sigma + pos_loss + neg_loss)
        loss = 1.0 / N * torch.sum(pos_loss + neg_loss)
        return loss

    def __proto_loss__(self, support_anchor, query):
        logits = self.__dist__(support_anchor, query, 3) # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        return logits

    # 求正例负例权重
    def __get_weight__(self, pos_distances, neg_distances, radius):
        pos_w = torch.exp(self.alfa * (pos_distances.view(-1) - radius)) / (1 + torch.sum(torch.exp(self.alfa * (pos_distances.view(-1) - radius))))
        neg_w = -torch.exp(self.beta * (neg_distances.view(-1) - (radius + self.m)) / (1 + torch.sum(torch.exp(self.beta * (neg_distances.view(-1) - (radius + self.m))))))
        return pos_w, neg_w

    def __multivariate_normal_pdf__(self, x, mean, cov):
        k = mean.size(0)
        reg_term = 1e-1 * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        cov += reg_term
        # 所有输入张量都转换为 float32
        cov = cov.to(dtype=torch.float32)
        x = x.to(dtype=torch.float32)
        mean = mean.to(dtype=torch.float32)

        # 在对数空间计算概率密度函数，这样可以避免处理极小的数值
        log_term2 = -0.5 * (x - mean).T @ torch.inverse(cov) @ (x - mean)
        log_pdf = -0.5 * torch.logdet(cov) - (k / 2) * torch.log(torch.tensor(2 * np.pi, dtype=torch.float32)) + log_term2
        return log_pdf

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

        # 计算所有对
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
        neg_distances_resize = neg_distances.permute(1, 0, 2).reshape(N, -1)
        # 20个元素的话是找第3小的值
        radius = torch.kthvalue(neg_distances_resize, int(0.05 * neg_distances_resize.size(-1)) + 1, dim=1).values
        # radius = torch.kthvalue(neg_distances_resize, int(self.v * neg_distances_resize.size(-1)) + 1, dim=1).values - self.m
        self.m = torch.mean(torch.kthvalue(neg_distances_resize, int(0.15 * neg_distances_resize.size(-1)) + 1, dim=1).values - radius)
        # 计算query到每个圆心的距离
        query_resize = query.unsqueeze(2).expand(-1, -1, support_anchor.size(0), -1)
        support_anchor_resize = support_anchor.unsqueeze(0).unsqueeze(0).expand(query_resize.size())
        query_distances = self.__dist__(query_resize, support_anchor_resize, 3)  # (B, total_Q, N)

        category_samples = [torch.Tensor()] * N
        # 计算support set上的均值和协方差矩阵
        for i in range(N):
            category_samples[i] = support[:, i, 0, :]
        # 计算每个类别的均值和协方差矩阵
        category_means = []
        category_cov_matrices = []
        for samples in category_samples:
            mean = torch.mean(samples, dim=0)
            category_means.append(mean)
            centered_data = samples - mean
            cov_matrix = torch.matmul(centered_data.T, centered_data) / (samples.size(0) - 1)
            category_cov_matrices.append(cov_matrix)

        # # 计算adaptive threshold
        # # 计算query set上的均值和协方差矩阵
        # query_samples = [torch.Tensor()] * N
        # for i in range(N):
        #     query_samples[i] = query[:, i, :]
        # # 计算每个query类别的均值和协方差矩阵
        # query_means = []
        # query_cov_matrices = []
        # for samples in query_samples:
        #     mean = torch.mean(samples, dim=0)
        #     query_means.append(mean)
        #     centered_data = samples - mean
        #     cov_matrix = torch.matmul(centered_data.T, centered_data) / (samples.size(0) - 1)
        #     query_cov_matrices.append(cov_matrix)
        # # 计算每个类别的概率密度
        # prob_densities = {i: [] for i in range(len(query_means))}
        # for i in range(batch_size):
        #     for j in range(int(total_Q)):
        #     # for j, (mean, cov) in enumerate(zip(query_means, query_cov_matrices)):
        #         sample = query[i, j, :]
        #         log_prob = self.__multivariate_normal_pdf__(sample, query_means[j], query_cov_matrices[j])
        #         prob_densities[j].append(log_prob.item())
        # # 计算每个类别的阈值
        # thresholds = torch.zeros((batch_size, int(total_Q)))
        # for i in range(batch_size):
        #     for j in range(int(total_Q)):
        #         sorted_probs = sorted(prob_densities[i])
        #         threshold_index = int(0.6 * len(sorted_probs))
        #         thresholds[i, j] = sorted_probs[threshold_index]

        # 存储每个样例点最可能的类别
        most_likely_categories = torch.full((batch_size, int(total_Q)), N, dtype=torch.int64)
        for i in range(batch_size):
            for j in range(int(total_Q)):
                sample = query[i, j, :]
                log_probabilities = []
                for k in range(N):
                    log_prob = self.__multivariate_normal_pdf__(sample, category_means[k], category_cov_matrices[k])
                    log_probabilities.append(log_prob)
                max_log_prob, idx = max((val, idx) for idx, val in enumerate(log_probabilities))
                if max_log_prob > self.threshold:
                    most_likely_categories[i, j] = idx


        # 计算logits和pred，使用MRM损失函数
        # mrm_loss = self.__mrm_loss__(pos_distances, neg_distances, category_cov_matrices, support_anchor, N, K, batch_size)
        mrm_loss = self.__mrm_loss2__(pos_distances, neg_distances, category_cov_matrices, radius, N, K, batch_size)

        loss = mrm_loss
        pred = most_likely_categories.view(-1)

        return loss, pred


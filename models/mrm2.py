import sys
from sklearn.decomposition import PCA
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from models.attetion import MultiHeadAttention, ScaledDotProductAttention
import random
import numpy as np
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class MRM(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, hidden_size, dot, N, K, Q, na_rate, pns_rate):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = True
        # 初始化 Multi-Head Attention 层
        self.prompt_length = 104
        # self.prompt_length = 0
        self.n_head = 8
        self.hidden_size = hidden_size
        self.d_model = (self.hidden_size + self.prompt_length)  # 使用隐藏层维度作为模型的输入和输出维度
        self.d_k = self.d_v = (self.hidden_size + self.prompt_length) // self.n_head  # 设置每个头的维度
        self.multi_head_attention = MultiHeadAttention(self.n_head, self.d_model, self.d_k, self.d_v)
        self.temperature = 40
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.temperature)
        self.num_views = 4
        self.batch_size = 4
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        # 初始化soft prompts为可学习参数
        self.soft_prompts = nn.Parameter(torch.randn(self.num_views, self.batch_size * self.K, self.N, 1, self.prompt_length))
        self.soft_prompts2 = nn.Parameter(torch.randn(self.num_views, self.batch_size, int(self.N + self.na_rate * self.Q), self.prompt_length))
        self.lamda = 0.001
        self.alfa = 1
        self.beta = 3
        # self.v = 0.2
        # self.w = 0.3
        self.v = nn.Parameter(torch.tensor(0.1))
        self.w = nn.Parameter(torch.tensor(0.2))
        self.num_extra_samples = 0
        self.extra_rate = pns_rate

    # 欧式距离 or 马氏距离
    def __dist__(self, x, y, dim, n_components=2):
        # 确保x和y位于相同的设备
        device = x.device
        y = y.to(device)
        if self.dot:
        #     return (x * y).sum(dim)  # (B, N, N)
            return torch.sqrt((torch.pow(x - y, 2)).sum(-1))
        else:
            b, n, _, h = x.shape
            # 将 x 和 y 进行降维
            x_flat = x.reshape(b, n, -1)
            y_flat = y.reshape(b, n, -1)
            # 确保降维数不超过样本和特征数
            pca_n_components = min(n_components, min(x_flat.shape[-1], x_flat.shape[1]))
            # 使用PyTorch的线性层进行降维
            linear = torch.nn.Linear(x_flat.shape[-1], n_components, bias=False).to(device)
            x_pca_tensor = linear(x_flat)
            y_pca_tensor = linear(y_flat)
            # 对每个 batch 分别进行 PCA 降维和马氏距离计算
            mahalanobis_dist = torch.zeros((b, n, n), device=device)
            for batch_idx in range(b):
                diff_pca_tensor = x_pca_tensor[batch_idx] - y_pca_tensor[batch_idx]
                # 计算协方差矩阵
                covariance_matrix = torch.cov(diff_pca_tensor.T)
                # 添加正则化项以提高数值稳定性
                regularization_term = 1e-5 * torch.eye(covariance_matrix.shape[0], device=device)
                covariance_matrix += regularization_term
                # 计算逆协方差矩阵
                inv_covariance_matrix = torch.linalg.inv(covariance_matrix)
                # 计算马氏距离
                for i in range(n):
                    for j in range(n):
                        diff_vec = diff_pca_tensor[i] - diff_pca_tensor[j]
                        mahalanobis_dist[batch_idx, i, j] = torch.sqrt(torch.dot(torch.matmul(diff_vec, inv_covariance_matrix), diff_vec))
                mahalanobis_dist = mahalanobis_dist.to(device)
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

    # 用support和query set一起求loss
    def __mrm_loss3__(self, pos_distances, neg_distances, radius, margin, N, K, batch_size):
        pos_loss = 1.0 / self.alfa * torch.log(1 + torch.sum(torch.exp(self.alfa * (pos_distances.view(-1) - radius.repeat(int(pos_distances.view(-1).shape[0] / radius.shape[0]))))))
        neg_loss = 1.0 / self.beta * torch.log(1 + torch.sum(torch.exp(-self.beta * (neg_distances.view(-1) - radius.repeat(int(neg_distances.view(-1).shape[0] / radius.shape[0]))))))
        loss = 1.0 / N * torch.sum(self.lamda * radius * radius + pos_loss + neg_loss)
        return loss

    # 计算4个view各自的权重
    def __softmax_linear__(self, mean, variance):
        view = []
        for i in range(4):
            view.append(torch.cat([mean[i], variance[i]], dim=1))
        all_view = torch.stack(view, dim=0)
        all_view = all_view.view(4, -1, self.hidden_size + self.prompt_length)
        _, scores, _ = self.scaled_dot_product_attention(all_view, all_view, all_view)
        attention_weights = scores
        # 对每个维度的注意力分布进行加权平均，得到每个维度的总体注意力分数
        total_attention_weights = torch.mean(attention_weights, dim=2)
        total_attention_weights = torch.mean(total_attention_weights, dim=1)
        # 对总体注意力分数再进行 softmax，得到最终的权重值
        weights = torch.softmax(total_attention_weights, dim=0)
        return weights

    def __sample_negatives__(self, support_anchor, radius, margin, num_extra_samples, dim):
        max_attempts = 10000  # 设定一个最大尝试次数
        attempts = 0
        negatives = []
        while len(negatives) < num_extra_samples and attempts < max_attempts:
            attempts += 1
            sample = torch.randn(dim)  # 生成单个样本
            if all(self.__dist__(sample, center, 0) > r + m for center, r, m in zip(support_anchor, radius, margin)):
                negatives.append(sample)
        # 如果尝试次数用尽后仍未获得足够的样本，则从已有样本中重复选取
        while len(negatives) < num_extra_samples:
            sample = negatives[random.randint(0, len(negatives) - 1)]
            negatives.append(sample)

        return torch.stack(negatives)

    def __sample_negatives2__(self, support_anchor, radius, margin, num_extra_samples, dim):
        max_attempts = 100  # Set a maximum number of attempts
        min_distance = 9999
        negatives = []
        samples_list = []
        distances_list = []
        while len(samples_list) < max_attempts:
            sample = torch.randn(dim)  # Generate a single sample
            distances = [self.__dist__(sample, center, 0) - (r + m) for center, r, m in zip(support_anchor, radius, margin)]
            for i in distances:
                if i > 0 and i < min_distance:
                    min_distance = i
            samples_list.append(sample)
            distances_list.append(min_distance)

        # 从距离列表中选择前 num_extra_samples 个最小值对应的样本
        _, indices = torch.topk(torch.tensor(distances_list), k=num_extra_samples, largest=False)
        negatives = [samples_list[i] for i in indices]

        return torch.stack(negatives)

    def forward(self, support, query, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.sentence_encoder(support)  # (B * N * K, D)
        query_emb = self.sentence_encoder(query)  # (B * total_Q, D)
        support_list = []
        query_list = []

        batch_size = self.batch_size

        for i in range(4):
            support_drop = self.drop(support_emb[i])
            query_drop = self.drop(query_emb[i])
            support_reshaped = support_drop.view(batch_size, N, K, self.hidden_size)  # (B, N, K, D)
            query_reshaped = query_drop.view(batch_size, int(total_Q), self.hidden_size)  # (B, total_Q, D)
            support_list.append(support_reshaped)
            query_list.append(query_reshaped)

        # 增加soft prompt
        # self.soft_prompts[i]的形状是(prompt_length, hidden_size)
        for i in range(4):
            support_list[i] = support_list[i].contiguous().view(-1, N * K, self.hidden_size)
            query_list[i] = query_list[i].contiguous().view(-1, int(total_Q), self.hidden_size)
            # Add soft prompt to support and query
            if self.prompt_length > 0:
                current_prompt = self.soft_prompts[i].view(batch_size, N, K, self.prompt_length)  # Reshape prompt to (batch_size, N, K, prompt_length)
                current_prompt = current_prompt.view(batch_size, -1, self.prompt_length)  # Reshape to (batch_size, N, prompt_length)
                support_with_prompt = torch.cat([current_prompt, support_list[i]], dim=2)  # (batch_size, prompt_length + N * K, hidden_size)
                current_prompt2 = self.soft_prompts2[i].view(batch_size, -1, self.prompt_length)  # Reshape prompt2 to (batch_size, int(N + na_rate * K), prompt_length)
                query_with_prompt = torch.cat([current_prompt2, query_list[i]], dim=2)  # (batch_size, int(N + na_rate * K) + total_Q, hidden_size)

                # Update support_list and query_list
                support_list[i] = support_with_prompt
                query_list[i] = query_with_prompt

        # MultiHeadAttention进行更新
        for i in range(4):
            support_list[i] = self.multi_head_attention(support_list[i], support_list[i], support_list[i])
            query_list[i] = self.multi_head_attention(query_list[i], query_list[i], query_list[i])
            # Reshape support_list back to original format
            support_list[i] = support_list[i].view(batch_size, -1, self.hidden_size + self.prompt_length)
            query_list[i] = query_list[i].view(batch_size, -1, self.hidden_size + self.prompt_length)
            # Further reshaping if necessary
            support_list[i] = support_list[i].view(batch_size, -1, 1, self.hidden_size + self.prompt_length)
            query_list[i] = query_list[i].view(batch_size, -1, self.hidden_size + self.prompt_length)

        # 求均值和方差
        support_mean = []
        support_variance = []
        query_mean = []
        query_variance = []
        for i in range(4):
            support_mean.append(support_list[i].mean(dim=1))
            support_variance.append(support_list[i].var(dim=1))
            query_mean.append(query_list[i].mean(dim=1))
            query_variance.append(query_list[i].var(dim=1))

        # 计算4个view各自的权重
        support_weights = self.__softmax_linear__(support_mean, support_variance)
        query_weights = self.__softmax_linear__(query_mean, query_variance)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        support_weights = torch.tensor([0.25, 0.25, 0.25, 0.25]).to(device)
        query_weights = torch.tensor([0.25, 0.25, 0.25, 0.25]).to(device)

        support_tensor = torch.stack(support_list, dim=0)
        support = torch.sum(support_tensor * support_weights.view(-1, 1, 1, 1, 1), dim=0)

        query_tensor = torch.stack(query_list, dim=0)
        query = torch.sum(query_tensor * query_weights.view(-1, 1, 1, 1), dim=0)
        support_anchor = torch.mean(support.view(batch_size, K, N, -1).mean(dim=1), 0)
        # support_anchor = torch.mean(support.squeeze(2), 0)

        # 计算support set所有对，用于计算radius和margin，以及给出pred
        support_resize = support.expand(-1, -1, support_anchor.size(0), -1)
        support_anchor_resize = support_anchor.unsqueeze(0).unsqueeze(0).expand(support_resize.size())
        all_pair = self.__dist__(support_resize, support_anchor_resize, 3)  # (B, N, N)
        pos_results = []
        neg_results = []
        for pos_class_idx in range(N):
            # 计算正例对(anchor, +)，注意低K时(K = 1)样例本身就是anchor，此时计算正例对无意义
            pos_pair = all_pair[:, pos_class_idx, pos_class_idx]  # (B, 1)
            # 避免NAN错误
            pos_pair = torch.where(pos_pair < 1e-10, torch.tensor(1e-10).cuda(), pos_pair)
            # 计算负例对(anchor, -)
            neg_class_indices = [i for i in range(N) if i != pos_class_idx]
            neg_pair = all_pair[:, pos_class_idx, neg_class_indices]  # (B, N - 1)
            # 避免NAN错误
            neg_pair = torch.where(neg_pair < 1e-10, torch.tensor(1e-10).cuda(), neg_pair)
            pos_results.append(pos_pair)
            neg_results.append(neg_pair)
        pos_pairs = torch.stack(pos_results, dim=1)  # (B, N)
        neg_pairs = torch.stack(neg_results, dim=1)  # (B, N, N - 1)
        pos_distances = pos_pairs
        neg_distances = neg_pairs
        pos_distances_resize = pos_distances.permute(1, 0)
        neg_distances_resize = neg_distances.permute(1, 0, 2).reshape(N, -1)
        all_distances = torch.cat((pos_distances_resize, neg_distances_resize), dim=1)

        # radius = torch.kthvalue(all_distances, int(self.v * all_distances.size(-1)) + 1, dim=1).values
        radius = torch.kthvalue(neg_distances_resize, int(self.v * neg_distances_resize.size(-1)) + 1, dim=1).values
        margin = torch.kthvalue(neg_distances_resize, int(self.w * neg_distances_resize.size(-1)) + 1, dim=1).values - radius

        self.num_extra_samples = int(self.extra_rate * neg_distances_resize.size(-1))
        # if self.extra_rate > 0:
        if self.num_extra_samples > 0:
            # 额外生成的伪负例并加入到原负例中
            # extra_negatives = self.__sample_negatives__(support_anchor, radius, margin, self.num_extra_samples, self.hidden_size)
            extra_negatives = self.__sample_negatives2__(support_anchor, radius, margin, self.num_extra_samples, self.hidden_size + self.prompt_length)
            extra_neg_pair = self.__dist__(extra_negatives.unsqueeze(1).expand(-1, support_anchor.size(0), -1), support_anchor.unsqueeze(0).expand(extra_negatives.size(0), -1, -1), 3)
            neg_results = []
            for pos_class_idx in range(N):
                # 计算使用伪负例扩充后的负例对(anchor, -)
                neg_class_indices = [i for i in range(N) if i != pos_class_idx]
                neg_pair = all_pair[:, pos_class_idx, neg_class_indices]  # (B, N - 1)
                neg_results.append(neg_pair)
            neg_pairs = torch.stack(neg_results, dim=1)
            neg_pairs = neg_pairs.permute(0, 2, 1).reshape(-1, N)
            device = neg_pairs.device
            extra_neg_pair = extra_neg_pair.to(device)
            combined_neg_pairs = torch.cat([neg_pairs, extra_neg_pair], dim=0)  # [B * (N - 1) + extra, N]
            neg_distances = combined_neg_pairs
            neg_distances_resize = neg_distances.permute(1, 0)
            all_distances = torch.cat((pos_distances_resize, neg_distances_resize), dim=1)
            # 更新半径和margin的值
            # radius = torch.kthvalue(all_distances, int(self.v * all_distances.size(-1)) + 1, dim=1).values
            radius = torch.kthvalue(neg_distances_resize, int(self.v * neg_distances_resize.size(-1)) + 1, dim=1).values
            margin = torch.kthvalue(neg_distances_resize, int(self.w * neg_distances_resize.size(-1)) + 1, dim=1).values - radius

        # 不加入额外负例的情况
        else:
            neg_pairs = neg_pairs.permute(0, 2, 1).reshape(-1, N)
            neg_distances = neg_pairs

        query_resize = query.unsqueeze(2).expand(-1, -1, support_anchor.size(0), -1)
        support_anchor_resize = support_anchor.unsqueeze(0).unsqueeze(0).expand(query_resize.size())
        # 计算query到每个圆心的距离
        query_distances = self.__dist__(query_resize, support_anchor_resize, 3)  # (B, total_Q, N)
        # 将query到每个圆心的距离与对应半径进行比较，判断在哪个圆的范围内
        pred_list = []
        for i in range(batch_size):
            for j in range(int(total_Q)):
                flag_k = 0
                minn = 0
                t = []
                s = []
                for r in range(N):
                    t.append(query_distances[i, j, r])
                    s.append(query_distances[i, j, r] - radius[r])
                    # s.append(query_distances[i, j, r])
                    if query_distances[i, j, r] <= radius[r] + margin[r]:
                    # if query_distances[i, j, r] <= radius[r]:
                        flag_k = 1
                if flag_k == 0:
                    pred_list.append(torch.tensor(N))
                else:
                    softmax_probabilities = F.softmax(torch.tensor(s).neg(), dim=0)
                    minn = torch.argmax(softmax_probabilities).item()
                    pred_list.append(torch.tensor(minn))
        pred = torch.stack(pred_list)

        # 计算query set所有对，用于计算loss
        query_resize = query.unsqueeze(2).expand(-1, -1, support_anchor.size(0), -1)
        support_anchor_resize = support_anchor.unsqueeze(0).unsqueeze(0).expand(query_resize.size())
        all_pair_q = self.__dist__(query_resize, support_anchor_resize, 3)  # (B, N, N)
        pos_results_q = []
        neg_results_q = []
        for pos_class_idx in range(N):
            # 计算正例对(anchor, +)
            pos_pair = all_pair_q[:, pos_class_idx, pos_class_idx]  # (B, 1)
            # 计算负例对(anchor, -)
            neg_class_indices = [i for i in range(N) if i != pos_class_idx]
            neg_pair = all_pair_q[:, pos_class_idx, neg_class_indices]  # (B, N - 1)
            pos_results_q.append(pos_pair)
            neg_results_q.append(neg_pair)
        pos_pairs_q = torch.stack(pos_results_q, dim=1)  # (B, N)
        neg_pairs_q = torch.stack(neg_results, dim=1)  # (B, N, N - 1)
        pos_distances_q = pos_pairs_q
        neg_distances_q = neg_pairs_q
        neg_distances_q_resize = neg_distances_q.permute(0, 2, 1).reshape(-1, N)

        # mrm_loss = self.__mrm_loss__(pos_distances, neg_distances, radius, margin, N, K, batch_size)
        # mrm_loss = self.__mrm_loss2__(pos_distances_q, neg_distances_q, radius, margin, N, K, batch_size)
        mrm_loss = self.__mrm_loss3__(torch.cat([pos_distances, pos_distances_q], dim=0), torch.cat([neg_distances, neg_distances_q_resize], dim=0), radius, margin, N, K, batch_size)

        loss = mrm_loss

        return loss, pred


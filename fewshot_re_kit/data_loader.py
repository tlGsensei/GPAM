import torch
import torch.utils.data as data
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
import random
import json
import pickle
from generate_context_debias import GenerateAdvRank as GetAavRank
from generate_context_debias import GenerateAdvExamples as GetAdvExample
from utils import clean_text, EntityMarker
from transformers import BertTokenizer, BertForMaskedLM, BertConfig


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(
                self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label

    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label


def get_loader(name, encoder, N, K, Q, batch_size,
               num_workers=0, collate_fn=collate_fn, na_rate=0, root='./data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


class FewRelDatasetcodalab(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        Q_na = int(self.na_rate * self.Q)
        length = len(self.json_data)
        print("episode: ", index)
        word, pos1, pos2, mask = self.__getraw__(self.json_data[index]['meta_test'])
        self.__additem__(query_set, word, pos1, pos2, mask)
        for j in range(self.N):
            if self.K == 1:
                # if True:
                # temp1 = self.json_data[index]['meta_test']
                temp2 = self.json_data[index]['meta_train'][j][0]
                word, pos1, pos2, mask = self.__getraw__(self.json_data[index]['meta_train'][j][0])
                self.__additem__(support_set, word, pos1, pos2, mask)
            else:
                for k in range(self.K):
                    word, pos1, pos2, mask = self.__getraw__(self.json_data[index]['meta_train'][j][k])
                    self.__additem__(support_set, word, pos1, pos2, mask)
        # NA
        # for j in range(Q_na):
        #     cur_class = np.random.choice(na_classes, 1, False)[0]
        #     index = np.random.choice(
        #         list(range(len(self.json_data[cur_class]))),
        #         1, False)[0]
        #     word, pos1, pos2, mask = self.__getraw__(
        #         self.json_data[cur_class][index])
        #     word = torch.tensor(word).long()
        #     pos1 = torch.tensor(pos1).long()
        #     pos2 = torch.tensor(pos2).long()
        #     mask = torch.tensor(mask).long()
        #     self.__additem__(query_set, word, pos1, pos2, mask)
        # query_label += [self.N] * Q_na

        return support_set, query_set

    def __len__(self):
        return 1000000000


def collate_fn_codalab(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets, query_sets = zip(*data)
    # for i in range(len(support_sets[0])):
    #     for k in support_sets:
    #         batch_support[k] += support_sets[i][k]
    # for i in range(len(query_sets[0])):
    #     for k in query_sets:
    #         batch_query[k] += query_sets[i][k]

    # 处理 support_sets
    for support_set in support_sets:
        for k in support_set:
            batch_support[k] += support_set[k]

    # 处理 query_sets
    for query_set in query_sets:
        for k in query_set:
            batch_query[k] += query_set[k]

    for k in batch_support:
        batch_support[k] = torch.stack([torch.tensor(item) for item in batch_support[k]], 0)
    for k in batch_query:
        batch_query[k] = torch.stack([torch.tensor(item) for item in batch_query[k]], 0)
    return batch_support, batch_query


def get_loader_codalab(name, encoder, N, K, Q, batch_size,
                       num_workers=0, collate_fn=collate_fn_codalab, na_rate=0, root='./test'):
    dataset = FewRelDatasetcodalab(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=0,
                                  collate_fn=collate_fn_codalab)
    return iter(data_loader)


class FewRelDatasetPair(data.Dataset):
    """
    FewRel Pair Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'],
                                     item['h'][2][0],
                                     item['t'][2][0])
        return word

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support = []
        query = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                word = self.__getraw__(
                    self.json_data[class_name][j])
                if count < self.K:
                    support.append(word)
                else:
                    query.append(word)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]
            word = self.__getraw__(
                self.json_data[cur_class][index])
            query.append(word)
        query_label += [self.N] * Q_na

        for word_query in query:
            for word_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set, query_label

    def __len__(self):
        return 1000000000


def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label


def get_loader_pair(name, encoder, N, K, Q, batch_size,
                    num_workers=0, collate_fn=collate_fn_pair, na_rate=0, root='./data', encoder_name='bert'):
    dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


class InputExample(object):
    def __init__(self, unique_id, text, head_span, tail_span, head_type, tail_type, label, relation_dp):
        self.unique_id = unique_id
        self.text = text  # list
        self.ori_text = text
        self.head_span = head_span
        self.tail_span = tail_span
        self.head_type = head_type
        self.tail_type = tail_type
        self.label = label
        self.relation_dp = relation_dp


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, ori_text_all, conex_text_all, entimask_text_all, headmask_text_all, tailmask_text_all,
                 tokens, input_ids, input_mask, relation_ids, relation_mask, head_span, tail_span, label):
        self.unique_id = unique_id
        self.ori_text_all = ori_text_all
        self.conex_text_all = conex_text_all
        self.entimask_text_all = entimask_text_all
        self.headmask_text_all = headmask_text_all
        self.tailmask_text_all = tailmask_text_all
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.relation_ids = relation_ids
        self.relation_mask = relation_mask
        self.head_span = head_span
        self.tail_span = tail_span
        self.label = label


class AddPromptToken(object):
    def __init__(self, unique_id, label, head_span, tail_span, ori_text, ori_text_MarkerToken,
                 conex_text_MarkerToken, entimask_text_MarkerToken, headmask_text_MarkerToken,
                 tailmask_text_MarkerToken, pseudo, pseudo_adv, pseudo_maskE, text, text_prompt, relation_prompt):
        self.unique_id = unique_id
        self.label = label
        self.head_span = head_span
        self.tail_span = tail_span
        self.ori_text = ori_text
        self.ori_text_MarkerToken = ori_text_MarkerToken
        self.conex_text_MarkerToken = conex_text_MarkerToken
        self.entimask_text_MarkerToken = entimask_text_MarkerToken
        self.headmask_text_MarkerToken = headmask_text_MarkerToken
        self.tailmask_text_MarkerToken = tailmask_text_MarkerToken
        self.pseudo = pseudo
        self.pseudo_adv = pseudo_adv
        self.pseudo_maskE = pseudo_maskE
        self.text = text
        self.text_prompt = text_prompt
        self.relation_prompt = relation_prompt


class RelationInfo(object):
    def __init__(self, relation_id, relation_description):
        self.relation_id = relation_id
        self.text = relation_description  # list


def load_relation_info_dict(tokenizer):
    """
        Args:
            filepath(str) : relation description file path
        Return:
            relation_info_dict(dict): mapping from relation surface form to class RelationInfo which include relation id and description
    """
    file_path = './data/fewrel/relation_description_new.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        relation_info_dict = {}
        line = f.readline()
        while line:
            relation_name, relation_description = line.split("    ")
            relation_description = tokenizer.tokenize(relation_description.strip())  # 不太懂tacred数据集的关系描述是后面的数字
            if len(relation_description) > 128 - 8:
                relation_description = relation_description[: (128 - 8)]
            # relation_description = ['[CLS]'] + relation_description + ['[SEP]']
            relation_info_dict[relation_name] = RelationInfo(len(relation_info_dict), relation_description)
            line = f.readline()
    return relation_info_dict


class FewRelDatasetSemi(data.Dataset):
    """
    FewRel Semi Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        # 加载wordrank
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, output_hidden_states=True)
        relation_info_dict = load_relation_info_dict(tokenizer)
        if not os.path.exists('./data/train_wordrank.npy'):
            train_ori_examples = FewRelDatasetSemi.preprocess('./data/train_wiki.json', relation_info_dict)
            train_adv_rank = GetAavRank(train_ori_examples, tokenizer)
        else:
            train_adv_rank_1 = np.load('./data/train_wordrank.npy', allow_pickle=True)
            train_adv_rank = train_adv_rank_1.tolist()

        know_class_adv_example = GetAdvExample(train_adv_rank, tokenizer)
        self.know_class_adv_example = know_class_adv_example

    def preprocess(file_path, relation_info_dict):
        """
        Args:
            file_path(str): file path to dataset
            relation_info_dict: mapping from relation surface to RelationInfo
        Return:
            data: list of InputExample
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            ori_data = json.load(f)
            unique_id = 0
            for relation_name, ins_list in ori_data.items():
                relation_id = relation_info_dict[relation_name].relation_id
                relation_dp = relation_info_dict[relation_name].text
                for instance in ins_list:
                    text = clean_text(instance['tokens'])
                    head_span = [instance['h'][2][0][0], instance['h'][2][0][-1]]
                    tail_span = [instance['t'][2][0][0], instance['t'][2][0][-1]]
                    head_type = []
                    tail_type = []
                    data.append(InputExample(unique_id, text, head_span, tail_span, head_type, tail_type, relation_id,
                                             relation_dp))
                    unique_id += 1
        return data

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask, context_debiased, head_debiased, tail_debiased):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        d['context_debiased'].append(context_debiased)
        d['head_debiased'].append(head_debiased)
        d['tail_debiased'].append(tail_debiased)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'context_debiased': [], 'head_debiased': [],
                       'tail_debiased': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'context_debiased': [], 'head_debiased': [],
                     'tail_debiased': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
                # 生成上下文无关view
                # context_debiased = word
                context_debiased = self.know_class_adv_example
                head_debiased = word
                tail_debiased = word
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                context_debiased = torch.tensor(context_debiased).long()
                head_debiased = torch.tensor(head_debiased).long()
                tail_debiased = torch.tensor(tail_debiased).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask, context_debiased, head_debiased,
                                     tail_debiased)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask, context_debiased, head_debiased, tail_debiased)
                count += 1

            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            context_debiased = torch.tensor(context_debiased).long()
            head_debiased = torch.tensor(head_debiased).long()
            tail_debiased = torch.tensor(tail_debiased).long()
            self.__additem__(query_set, word, pos1, pos2, mask, context_debiased, head_debiased, tail_debiased)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label

    def __len__(self):
        return 1000000000


def collate_fn_semi(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'context_debiased': [], 'head_debiased': [],
                     'tail_debiased': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'context_debiased': [], 'head_debiased': [],
                   'tail_debiased': []}
    batch_label = []

    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)

    return batch_support, batch_query, batch_label


def get_loader_semi(name, encoder, N, K, Q, batch_size,
                    num_workers=0, collate_fn=collate_fn_semi, na_rate=0, root='./data', encoder_name='bert'):
    dataset = FewRelDatasetSemi(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    # return iter(data_loader)
    return data_loader


class FewRelDatasetWithPseudoLabel(data.Dataset):
    def __init__(self, args, examples, tokenizer, encoder, N, K, Q, na_rate, batch_size):
        """
        Args:
            examples: list of InputExample
        """
        self.max_len = args.max_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.examples = examples

        self.examples_prompt = self.add_prompt(examples=self.examples, tokenizer=self.tokenizer, args=args)
        actual_max_len = self.get_max_seq_length(self.examples_prompt, self.tokenizer)
        self.features = self.convert_examples_to_features(examples_prompt=self.examples_prompt,
                                                          seq_length=max(2 + actual_max_len, self.max_len),
                                                          tokenizer=self.tokenizer)
        unique_relation_ids = set()
        # 遍历features列表中的每个元素，将其中的label添加到集合中
        for item in self.examples_prompt:
            # 假设每个item是一个包含label字段的对象，其中label是一个整数
            label = item.label  # 假设label是一个整数
            unique_relation_ids.add(label)  # 使用add方法将单个整数添加到集合中
        self.classes2 = unique_relation_ids
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.batches = self._create_batches()

    def __additem__(self, d, word, pos1, pos2, context_debiased, entimask_debiased, head_debiased, tail_debiased,
                    raw_label):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['context_debiased'].append(context_debiased)
        d['entimask_debiased'].append(entimask_debiased)
        d['head_debiased'].append(head_debiased)
        d['tail_debiased'].append(tail_debiased)
        d['raw_label'].append(raw_label)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'context_debiased': [], 'entimask_debiased': [],
                       'head_debiased': [], 'tail_debiased': [], 'raw_label': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'context_debiased': [], 'entimask_debiased': [],
                     'head_debiased': [], 'tail_debiased': [], 'raw_label': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        batch_index = index // self.batch_size
        target_classes = self.batches[batch_index]
        na_classes = list(filter(lambda x: x not in target_classes, self.classes2))
        for i, class_name in enumerate(target_classes):
            count = 0
            for j in range(self.K + self.Q):
                # 随机生成一个index
                index2 = random.randint(0, len(self.features) - 1)
                while self.features[index2].label != target_classes[i]:
                    index2 = random.randint(0, len(self.features) - 1)  # 重新生成索引，直到满足条件为止
                raw_label = self.features[index2].label
                head_span = self.features[index2].head_span
                tail_span = self.features[index2].tail_span
                ori_text_all = self.features[index2].ori_text_all
                conex_text_all = self.features[index2].conex_text_all
                entimask_text_all = self.features[index2].entimask_text_all
                headmask_text_all = self.features[index2].headmask_text_all
                tailmask_text_all = self.features[index2].tailmask_text_all
                word = ori_text_all
                pos1 = head_span[0]
                pos2 = tail_span[0]
                context_debiased = conex_text_all
                entimask_debiased = entimask_text_all
                head_debiased = headmask_text_all
                tail_debiased = tailmask_text_all
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, context_debiased, entimask_debiased, head_debiased,
                                     tail_debiased, raw_label)
                else:
                    self.__additem__(query_set, word, pos1, pos2, context_debiased, entimask_debiased, head_debiased,
                                     tail_debiased, raw_label)
                count += 1
            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = random.sample(na_classes, self.N)
            index2 = random.randint(0, len(self.features) - 1)
            while self.features[index2].label not in cur_class:
                index2 = random.randint(0, len(self.features) - 1)  # 重新生成索引，直到满足条件为止
            raw_label = self.features[index2].label
            head_span = self.features[index2].head_span
            tail_span = self.features[index2].tail_span
            ori_text_all = self.features[index2].ori_text_all
            conex_text_all = self.features[index2].conex_text_all
            entimask_text_all = self.features[index2].entimask_text_all
            headmask_text_all = self.features[index2].headmask_text_all
            tailmask_text_all = self.features[index2].tailmask_text_all
            word = ori_text_all
            pos1 = head_span[0]
            pos2 = tail_span[0]
            context_debiased = conex_text_all
            entimask_debiased = entimask_text_all
            head_debiased = headmask_text_all
            tail_debiased = tailmask_text_all
            self.__additem__(query_set, word, pos1, pos2, context_debiased, entimask_debiased, head_debiased,
                             tail_debiased, raw_label)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label

    def __len__(self):
        return len(self.examples)

    def collate_fn(data):
        batch_support = {'word': [], 'pos1': [], 'pos2': [], 'context_debiased': [], 'entimask_debiased': [],
                         'head_debiased': [], 'tail_debiased': [], 'raw_label': []}
        batch_query = {'word': [], 'pos1': [], 'pos2': [], 'context_debiased': [], 'entimask_debiased': [],
                       'head_debiased': [], 'tail_debiased': [], 'raw_label': []}
        batch_label = []
        support_sets, query_sets, query_labels = zip(*data)
        for i in range(len(support_sets)):
            batch_support['word'] += torch.tensor(support_sets[i]['word'], dtype=torch.long)
            batch_support['pos1'] += torch.tensor(support_sets[i]['pos1'], dtype=torch.long)
            batch_support['pos2'] += torch.tensor(support_sets[i]['pos2'], dtype=torch.long)
            batch_support['context_debiased'] += torch.tensor(support_sets[i]['context_debiased'], dtype=torch.long)
            batch_support['entimask_debiased'] += torch.tensor(support_sets[i]['entimask_debiased'], dtype=torch.long)
            batch_support['head_debiased'] += torch.tensor(support_sets[i]['head_debiased'], dtype=torch.long)
            batch_support['tail_debiased'] += torch.tensor(support_sets[i]['tail_debiased'], dtype=torch.long)
            batch_support['raw_label'] += torch.tensor(support_sets[i]['raw_label'], dtype=torch.long)
        for i in range(len(query_sets)):
            batch_query['word'] += torch.tensor(query_sets[i]['word'], dtype=torch.long)
            batch_query['pos1'] += torch.tensor(query_sets[i]['pos1'], dtype=torch.long)
            batch_query['pos2'] += torch.tensor(query_sets[i]['pos2'], dtype=torch.long)
            batch_query['context_debiased'] += torch.tensor(query_sets[i]['context_debiased'], dtype=torch.long)
            batch_query['entimask_debiased'] += torch.tensor(query_sets[i]['entimask_debiased'], dtype=torch.long)
            batch_query['head_debiased'] += torch.tensor(query_sets[i]['head_debiased'], dtype=torch.long)
            batch_query['tail_debiased'] += torch.tensor(query_sets[i]['tail_debiased'], dtype=torch.long)
            batch_query['raw_label'] += torch.tensor(query_sets[i]['raw_label'], dtype=torch.long)
            batch_label += query_labels[i]

        batch_support['word'] = torch.stack(batch_support['word'], 0)
        batch_support['pos1'] = torch.stack(batch_support['pos1'], 0)
        batch_support['pos2'] = torch.stack(batch_support['pos2'], 0)
        batch_support['context_debiased'] = torch.stack(batch_support['context_debiased'], 0)
        batch_support['entimask_debiased'] = torch.stack(batch_support['entimask_debiased'], 0)
        batch_support['head_debiased'] = torch.stack(batch_support['head_debiased'], 0)
        batch_support['tail_debiased'] = torch.stack(batch_support['tail_debiased'], 0)
        batch_support['raw_label'] = torch.stack(batch_support['raw_label'], 0)

        batch_query['word'] = torch.stack(batch_query['word'], 0)
        batch_query['pos1'] = torch.stack(batch_query['pos1'], 0)
        batch_query['pos2'] = torch.stack(batch_query['pos2'], 0)
        batch_query['context_debiased'] = torch.stack(batch_query['context_debiased'], 0)
        batch_query['entimask_debiased'] = torch.stack(batch_query['entimask_debiased'], 0)
        batch_query['head_debiased'] = torch.stack(batch_query['head_debiased'], 0)
        batch_query['tail_debiased'] = torch.stack(batch_query['tail_debiased'], 0)
        batch_query['raw_label'] = torch.stack(batch_query['raw_label'], 0)

        batch_label = torch.tensor(batch_label)

        return batch_support, batch_query, batch_label

    def preprocess(file_path, relation_info_dict):
        """
        Args:
            file_path(str): file path to dataset
            relation_info_dict: mapping from relation surface to RelationInfo
        Return:
            data: list of InputExample
        """
        with open("./data/fewrel/entity_type/result/label_t.pickle", 'rb') as pickle_file1:
            datat_t1 = pickle.load(pickle_file1)
        with open("./data/fewrel/entity_type/result/label_h.pickle", 'rb') as pickle_file2:
            datat_h1 = pickle.load(pickle_file2)
        with open("./data/fewrel/entity_type/result/unlabel_dict_t.pickle", 'rb') as pickle_file1:
            datat_t2 = pickle.load(pickle_file1)
        with open("./data/fewrel/entity_type/result/unlabel_dict_h.pickle", 'rb') as pickle_file2:
            datat_h2 = pickle.load(pickle_file2)
        datat_t = {**datat_t1, **datat_t2}
        datat_h = {**datat_h1, **datat_h2}

        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            ori_data = json.load(f)
            unique_id = 0
            for relation_name, ins_list in ori_data.items():
                relation_id = relation_info_dict[relation_name].relation_id
                relation_dp = relation_info_dict[relation_name].text
                data_entitytype_h = datat_h[relation_name]
                data_entitytype_t = datat_t[relation_name]
                for num, instance in enumerate(ins_list):
                    text = clean_text(instance['tokens'])
                    head_span = [instance['h'][2][0][0], instance['h'][2][0][-1]]
                    tail_span = [instance['t'][2][0][0], instance['t'][2][0][-1]]
                    head_type = data_entitytype_h[num]
                    tail_type = data_entitytype_t[num]
                    data.append(InputExample(unique_id, text, head_span, tail_span, head_type, tail_type, relation_id,
                                             relation_dp))
                    unique_id += 1
        return data

    def data_split(examples1, examples2, ratio=0.7):
        from sklearn.utils import shuffle
        examples = examples1 + examples2
        examples = shuffle(examples)
        train_num = int(len(examples) * ratio)
        train_examples = examples[:train_num]
        test_examples = examples[train_num:]
        return train_examples, test_examples

    '''
    def get_max_seq_length(self, examples, tokenizer):
        max_seq_len = -1
        for example in examples:
            bert_tokens = tokenizer.tokenize(' '.join(example.text))
            cur_len = len(bert_tokens)
            if cur_len > max_seq_len:
                max_seq_len = cur_len
        return max_seq_len
    '''

    def get_max_seq_length(self, examples_prompt, tokenizer):
        max_seq_len = -1
        remove_cnt = 0
        new_examples_list = []
        for example in examples_prompt:
            cur_len_1 = len(example.ori_text_MarkerToken[0]) + len(example.ori_text_MarkerToken[1])
            cur_len_2 = len(example.entimask_text_MarkerToken[0]) + len(example.entimask_text_MarkerToken[1])
            cur_len = max(cur_len_1, cur_len_2)
            if cur_len <= self.max_len - 3:
                new_examples_list.append(example)
            else:
                remove_cnt += 1
                continue
            if cur_len > max_seq_len:
                max_seq_len = cur_len
        print("removed sentence number:{}".format(remove_cnt))
        self.examples = new_examples_list
        return max_seq_len + 3

    def add_prompt(self, examples, tokenizer, args):
        examples_prompt_token, SUBJECT, OBJECT = [], [], []
        entityMarker = EntityMarker()
        examples = examples.example_with_adv
        nn = 0
        for example in examples:
            OriTextMarker = EntityMaskMarker = ConExchMarker = True
            h_flag = t_flag = ht_flag = False
            if OriTextMarker == True:
                ori_text_Marker = entityMarker.tokenize(example, h_flag, t_flag, ht_flag, tokenizer)
            if ConExchMarker == True:
                conex_text_Marker = entityMarker.adv_tokenize(example, h_flag, t_flag, ht_flag, tokenizer)
            if EntityMaskMarker == True:
                if example.head_type is not None and example.tail_type is not None and example.head_type != 0 and example.tail_type != 0:
                    random_num = random.random()
                    h_flag = random_num < args.h_alpha
                    t_flag = random_num < args.t_alpha and random_num > args.h_alpha
                    ht_flag = random_num > args.t_alpha
                    entimask_text_Marker = entityMarker.tokenize(example, h_flag, t_flag, ht_flag, tokenizer)
                    headmask_text_Marker = entityMarker.tokenize(example, h_flag, False, False, tokenizer)
                    tailmask_text_Marker = entityMarker.tokenize(example, False, t_flag, False, tokenizer)
                else:
                    nn = nn + 1
                    entimask_text_Marker = entityMarker.tokenize(example, h_flag, t_flag, ht_flag, tokenizer)
                    headmask_text_Marker = entityMarker.tokenize(example, h_flag, False, False, tokenizer)
                    tailmask_text_Marker = entityMarker.tokenize(example, False, t_flag, False, tokenizer)

            SUBJECT = " ".join(example.ori_text[example.head_span[0]: example.head_span[1] + 1])
            OBJECT = " ".join(example.ori_text[example.tail_span[0]: example.tail_span[1] + 1])
            text_prompt = f"head {SUBJECT} {tokenizer.mask_token} tail {OBJECT}."
            relation_prompt = f"subject {tokenizer.mask_token} object. "
            text_1 = ' '.join(example.ori_text)
            text_2 = text_1.replace(SUBJECT, "head" + SUBJECT)
            text_3 = text_2.replace(OBJECT, "tail" + OBJECT)

            examples_prompt_token.append(
                AddPromptToken(
                    unique_id=example.unique_id,
                    label=example.label,  # bert_token
                    head_span=example.head_span,
                    tail_span=example.tail_span,
                    ori_text=example.ori_text,
                    ori_text_MarkerToken=ori_text_Marker,
                    conex_text_MarkerToken=conex_text_Marker,
                    entimask_text_MarkerToken=entimask_text_Marker,
                    headmask_text_MarkerToken=headmask_text_Marker,
                    tailmask_text_MarkerToken=tailmask_text_Marker,
                    # headmask_text_MarkerToken=entimask_text_Marker,
                    # tailmask_text_MarkerToken=entimask_text_Marker,
                    pseudo=example.pseudo,
                    pseudo_adv=example.pseudo_adv,
                    pseudo_maskE=example.pseudo_maskE,
                    text=text_3,
                    text_prompt=text_prompt,
                    relation_prompt=relation_prompt
                )
            )
        return examples_prompt_token

    def convert_examples_to_features(self, examples_prompt, seq_length, tokenizer):
        features = []
        for example in examples_prompt:
            ori_input_ids = tokenizer.convert_tokens_to_ids(
                example.ori_text_MarkerToken[0]) + tokenizer.convert_tokens_to_ids(example.ori_text_MarkerToken[1])
            ori_token_type = [0] * len(example.ori_text_MarkerToken[0]) + [1] * len(example.ori_text_MarkerToken[1])
            ori_mask_ids = [1] * len(example.ori_text_MarkerToken[0] + example.ori_text_MarkerToken[1])

            conex_input_ids = tokenizer.convert_tokens_to_ids(
                example.conex_text_MarkerToken[0]) + tokenizer.convert_tokens_to_ids(example.conex_text_MarkerToken[1])
            conex_token_type = [0] * len(example.conex_text_MarkerToken[0]) + [1] * len(
                example.conex_text_MarkerToken[1])
            conex_mask_ids = [1] * len(example.conex_text_MarkerToken[0] + example.conex_text_MarkerToken[1])

            entimask_input_ids = tokenizer.convert_tokens_to_ids(
                example.entimask_text_MarkerToken[0]) + tokenizer.convert_tokens_to_ids(
                example.entimask_text_MarkerToken[1])
            entimask_token_type = [0] * len(example.entimask_text_MarkerToken[0]) + [1] * len(
                example.entimask_text_MarkerToken[1])
            entimask_mask_ids = [1] * len(example.entimask_text_MarkerToken[0] + example.entimask_text_MarkerToken[1])

            headmask_input_ids = tokenizer.convert_tokens_to_ids(
                example.headmask_text_MarkerToken[0]) + tokenizer.convert_tokens_to_ids(
                example.headmask_text_MarkerToken[1])
            headmask_token_type = [0] * len(example.headmask_text_MarkerToken[0]) + [1] * len(
                example.headmask_text_MarkerToken[1])
            headmask_mask_ids = [1] * len(example.headmask_text_MarkerToken[0] + example.headmask_text_MarkerToken[1])

            tailmask_input_ids = tokenizer.convert_tokens_to_ids(
                example.tailmask_text_MarkerToken[0]) + tokenizer.convert_tokens_to_ids(
                example.tailmask_text_MarkerToken[1])
            tailmask_token_type = [0] * len(example.tailmask_text_MarkerToken[0]) + [1] * len(
                example.tailmask_text_MarkerToken[1])
            tailmask_mask_ids = [1] * len(example.tailmask_text_MarkerToken[0] + example.tailmask_text_MarkerToken[1])

            # Zero-pad up to the sequence length
            ori_input_ids = ori_input_ids + [0] * (seq_length - len(ori_input_ids))
            ori_token_type = ori_token_type + [0] * (seq_length - len(ori_token_type))
            ori_mask_ids = ori_mask_ids + [0] * (seq_length - len(ori_mask_ids))
            ori_text_all = [_ for _ in (ori_input_ids, ori_token_type, ori_mask_ids)]

            conex_input_ids = conex_input_ids + [0] * (seq_length - len(conex_input_ids))
            conex_token_type = conex_token_type + [0] * (seq_length - len(conex_token_type))
            conex_mask_ids = conex_mask_ids + [0] * (seq_length - len(conex_mask_ids))
            conex_text_all = [_ for _ in (conex_input_ids, conex_token_type, conex_mask_ids)]

            entimask_input_ids = entimask_input_ids + [0] * (seq_length - len(entimask_input_ids))
            entimask_token_type = entimask_token_type + [0] * (seq_length - len(entimask_token_type))
            entimask_mask_ids = entimask_mask_ids + [0] * (seq_length - len(entimask_mask_ids))
            entimask_text_all = [_ for _ in (entimask_input_ids, entimask_token_type, entimask_mask_ids)]

            headmask_input_ids = headmask_input_ids + [0] * (seq_length - len(headmask_input_ids))
            headmask_token_type = headmask_token_type + [0] * (seq_length - len(headmask_token_type))
            headmask_mask_ids = headmask_mask_ids + [0] * (seq_length - len(headmask_mask_ids))
            headmask_text_all = [_ for _ in (headmask_input_ids, headmask_token_type, headmask_mask_ids)]

            tailmask_input_ids = tailmask_input_ids + [0] * (seq_length - len(tailmask_input_ids))
            tailmask_token_type = tailmask_token_type + [0] * (seq_length - len(tailmask_token_type))
            tailmask_mask_ids = tailmask_mask_ids + [0] * (seq_length - len(tailmask_mask_ids))
            tailmask_text_all = [_ for _ in (tailmask_input_ids, tailmask_token_type, tailmask_mask_ids)]

            assert len(ori_input_ids) == seq_length
            assert len(ori_token_type) == seq_length
            assert len(ori_mask_ids) == seq_length
            assert len(conex_input_ids) == seq_length
            assert len(conex_token_type) == seq_length
            assert len(conex_mask_ids) == seq_length
            assert len(entimask_input_ids) == seq_length
            assert len(entimask_token_type) == seq_length
            assert len(entimask_mask_ids) == seq_length
            bert_tokens_1 = tokenizer.tokenize(example.text)
            bert_tokens_2 = tokenizer.tokenize(example.text_prompt)
            bert_tokens = bert_tokens_1 + bert_tokens_2
            # Account for [CLS] and [SEP] with "- 2"
            if len(bert_tokens) > seq_length - 3:
                bert_tokens_1 = bert_tokens_1[0: (seq_length - len(bert_tokens_2) - 3)]
            tokens = []
            # input_type_ids = [
            tokens.append("[CLS]")
            # input_type_ids.append(0) # input type id is not needed in relation extraction task
            for token in bert_tokens_1:
                tokens.append(token)
                # input_type_ids.append(0)
            tokens.append("[SEP]")
            for token in bert_tokens_2:
                tokens.append(token)
            tokens.append("[SEP]")
            # input_type_ids.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            # relation_type_ids.append(0)
            # relation_token = example.relation_dp
            relation_token_prompt = tokenizer.tokenize(example.relation_prompt)
            relation_tokens = []
            relation_tokens.append("[CLS]")
            # for token in relation_token:
            #     relation_tokens.append(token)
            relation_tokens.append("[SEP]")
            for token in relation_token_prompt:
                relation_tokens.append(token)
            relation_tokens.append("[SEP]")
            relation_ids = tokenizer.convert_tokens_to_ids(relation_tokens)
            relation_mask = [1] * len(relation_ids)
            # Zero-pad up to the sequence length
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                # input_type_ids.append(0)
            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            # assert len(input_type_ids) == seq_length
            while len(relation_ids) < seq_length:
                relation_ids.append(0)
                relation_mask.append(0)
            assert len(relation_ids) == seq_length
            assert len(relation_mask) == seq_length

            features.append(
                InputFeatures(
                    unique_id=example.unique_id,
                    ori_text_all=ori_text_all,
                    conex_text_all=conex_text_all,
                    entimask_text_all=entimask_text_all,
                    headmask_text_all=headmask_text_all,
                    tailmask_text_all=tailmask_text_all,
                    # headmask_text_all=entimask_text_all,
                    # tailmask_text_all=entimask_text_all,
                    tokens=tokens,  # bert_token
                    input_ids=input_ids,
                    input_mask=input_mask,
                    relation_ids=relation_ids,
                    relation_mask=relation_mask,
                    head_span=example.head_span,
                    tail_span=example.tail_span,
                    label=example.label
                )
            )
        return features

    def _create_batches(self):
        # 创建每个batch的类别集合
        batches = []
        num_batches = len(self.examples) // self.batch_size
        for _ in range(num_batches):
            selected_classes = random.sample(self.classes2, self.N)
            batches.append(selected_classes)
        return batches

    def fill_bert_entity_spans(self, examples, features):
        for example, feature in zip(examples, features):
            bert_tokens = feature.tokens[1:]
            actual_tokens = example.text
            head_span = example.head_span
            tail_span = example.tail_span
            actual_token_index, bert_token_index, actual_to_bert = 0, 0, []
            while actual_token_index < len(actual_tokens):
                start, end = bert_token_index, bert_token_index
                actual_token = actual_tokens[actual_token_index]
                token_index = 0
                while token_index < len(actual_token):
                    bert_token = bert_tokens[bert_token_index]
                    if bert_token.startswith('##'):
                        bert_token = bert_token[2:]
                    assert (bert_token.lower() == actual_token[token_index:token_index + len(bert_token)].lower())
                    end = bert_token_index
                    token_index += len(bert_token)
                    bert_token_index += 1
                actual_to_bert.append([start, end])
                actual_token_index += 1

            feature.head_span = (1 + actual_to_bert[head_span[0]][0], 1 + actual_to_bert[head_span[1]][1])
            feature.tail_span = (1 + actual_to_bert[tail_span[0]][0], 1 + actual_to_bert[tail_span[1]][1])


class FewRelUnsupervisedDataset(data.Dataset):
    """
    FewRel Unsupervised Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        total = self.N * self.K
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

        indices = np.random.choice(list(range(len(self.json_data))), total, False)
        for j in indices:
            word, pos1, pos2, mask = self.__getraw__(
                self.json_data[j])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(support_set, word, pos1, pos2, mask)

        return support_set

    def __len__(self):
        return 1000000000


def collate_fn_unsupervised(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    support_sets = data
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    return batch_support


def get_loader_unsupervised(name, encoder, N, K, Q, batch_size,
                            num_workers=0, collate_fn=collate_fn_unsupervised, na_rate=0, root='./data'):
    dataset = FewRelUnsupervisedDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)

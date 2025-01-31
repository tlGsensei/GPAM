import os
from torch.utils.data import Dataset
from utils import get_ori_feat, get_important_scores
from transformers import BertForMaskedLM
from tqdm import tqdm
import numpy as np
import pickle

# 从本地文件夹装载模型的方法
# model_path = "./dataroot/models/bert-base-uncased"
# mlm_model = BertModel.from_pretrained(model_path)
# mlm_model.to('cuda')

# huggingface装载模型
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_model.to('cuda')

class examplesRank(object):
    def __init__(self, head_span, label, tail_span, head_type, tail_type, ori_text, unique_id, word_rank):
        self.head_span = head_span
        self.label = label
        self.tail_span = tail_span
        self.ori_text = ori_text
        self.unique_id = unique_id
        self.word_rank = word_rank
        self.head_type = head_type
        self.tail_type = tail_type

class examplesAdv(object):
    def __init__(self, head_span, label, tail_span,head_type, tail_type, ori_text,unique_id,word_rank, text_adv):
        self.head_span = head_span
        self.label = label
        self.tail_span = tail_span
        self.ori_text =ori_text
        self.unique_id = unique_id
        self.word_rank = word_rank
        self.text_adv = text_adv
        self.head_type = head_type
        self.tail_type = tail_type
        self.pseudo = -1
        self.pseudo_maskE = -1
        self.pseudo_adv = -1

class GenerateAdvRank(Dataset):
    def __init__(self, examples, tokenizer):
        """
        This class aims to generate adv examples for input text
        Args:
        examples: list of InputExample
        """
        self.tokenizer = tokenizer
        self.examples = examples
        print(len(self.examples))

        self.examples_with_rank = self.get_score_rank(examples = self.examples, tokenizer = self.tokenizer)

    def __len__(self):
        return len(self.examples)


    def get_score_rank(self, examples, tokenizer):
        examples_with_rank = []
        for example in tqdm(examples):
            SUBJECT = " ".join(example.text[example.head_span[0]: example.head_span[1] + 1])
            OBJECT = " ".join(example.text[example.tail_span[0]: example.tail_span[1] + 1])
            text_ori = ' '.join(example.text)
            text_ori_prompt = f"{SUBJECT} {tokenizer.mask_token} {OBJECT}."
            bert_tokens_1 = tokenizer.tokenize(text_ori)
            bert_tokens_2 = tokenizer.tokenize(text_ori_prompt)
            ori_rel_feat, no_need_mask_position, actual_to_bert,text_len = get_ori_feat(example.text, example.head_span, example.tail_span, bert_tokens_1, bert_tokens_2, mlm_model, tokenizer) #原始句子得到的关系表征-通过prompt计算
            rank_index = get_important_scores(example.head_span, example.tail_span, example.text, text_ori_prompt, ori_rel_feat, tokenizer, text_len, mlm_model)
            examples_with_rank.append(
                    examplesRank(
                    unique_id=example.unique_id,
                    label=example.label,  # bert_token
                    head_span=example.head_span,
                    tail_span=example.tail_span,
                    head_type=example.head_type,
                    tail_type=example.tail_type,
                    ori_text=example.ori_text,
                    word_rank = rank_index,
                )
            )
        if not os.path.exists('./data/train_wordrank.npy'):
            a = np.array(examples_with_rank)
            np.save('./data/train_wordrank.npy', a)
        if not os.path.exists('./data/val_wordrank.npy'):
            a = np.array(examples_with_rank)
            np.save('./data/val_wordrank.npy', a)

        return examples_with_rank

class GenerateAdvExamples(Dataset):
    def __init__(self, examples, tokenizer):
        """
        This class aims to generate adv examples for input text
        Args:
        examples: list of InputExample
        """
        self.tokenizer = tokenizer
        self.examples = examples
        print(len(self.examples))
        self.example_with_adv = self.generate_adv_example(examples = self.examples, tokenizer = self.tokenizer)

    def __len__(self):
        return len(self.examples)

    def generate_adv_example(self, examples, tokenizer):
        examples_with_adv = []
        f_read = open('./data/fewrel/syn_word_vob_new.pkl', 'rb')
        word_dic = pickle.load(f_read)
        f_read.close()
        for example in examples:
            new_adv_words_rank = dict()
            count_num = 0
            for w_index in example.word_rank:
                if example.ori_text[w_index] in word_dic and count_num <= 5 :
                    new_adv_words_rank[w_index] = word_dic[example.ori_text[w_index]]
                    count_num = count_num + 1
                else:
                    new_adv_words_rank[w_index] = [example.ori_text[w_index]]
            example_adv1 = []
            example_adv2 = []
            example_adv = [0,0]
            for ind, w in enumerate(example.ori_text):
                if ind in example.word_rank and len(new_adv_words_rank[ind]) > 1:
                    example_adv1.append(new_adv_words_rank[ind][0])
                    example_adv2.append(new_adv_words_rank[ind][1])
                elif ind in example.word_rank and len(new_adv_words_rank[ind]) == 1:
                    example_adv1.append(new_adv_words_rank[ind][0])
                    example_adv2.append(new_adv_words_rank[ind][0])
                else:
                    example_adv1.append(w)
                    example_adv2.append(w)
            example_adv[0] = example_adv1
            example_adv[1] = example_adv2
            examples_with_adv.append(
                examplesAdv(
                    unique_id=example.unique_id,
                    label=example.label,  # bert_token
                    head_span=example.head_span,
                    tail_span=example.tail_span,
                    head_type = example.head_type,
                    tail_type=example.tail_type,
                    ori_text=example.ori_text,
                    word_rank=example.word_rank,
                    text_adv = example_adv
                )
            )
        return examples_with_adv

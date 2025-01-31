#from munkres import Munkres, print_matrix
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn import metrics
from transformers import PreTrainedTokenizer
from typing import Tuple
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords



def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def cal_info(pseudo_label_list, true_label):
    """
    pseudo_label_list: 二维列表
    true_label: 一维numpy
    """
    p_acc_list = []
    num_pseudo_label = len(pseudo_label_list)
    NMI = np.zeros((num_pseudo_label, num_pseudo_label))
    for pseudo_label in pseudo_label_list:
        p_acc_list.append(eval_acc(true_label, maplabels(true_label, np.array(pseudo_label))))
    for i in range(num_pseudo_label):
        for j in range(num_pseudo_label):
            if i == j:
                continue
            try:
                NMI[i][j] = metrics.normalized_mutual_info_score(np.array(pseudo_label_list[i]),np.array(pseudo_label_list[j]))
            except:
                print(i,j)
                exit()
    mean_NMI = np.mean(NMI, axis = 1).tolist()
    data = [(acc, nmi) for acc, nmi in zip(p_acc_list, mean_NMI)]
    data.sort()
    p_acc_list = [acc for acc, _ in data]
    mean_NMI = [nmi for _, nmi in data]
    print("####")
    print(p_acc_list)
    print(mean_NMI)
    print("####")

def compute_position(tru, pse, kl1,b):
    t_pos_mean_kl1 = torch.sum(tru * kl1)/(torch.sum(tru)-b)
    t_neg_mean_kl1 = torch.sum((1-tru) * kl1) / (torch.sum(1-tru))
    p_pos_mean_kl1 = torch.sum(pse * kl1)/(torch.sum(pse)-b)
    p_neg_mean_kl1 = torch.sum((1-pse) * kl1) / (torch.sum(1-pse))
    deta_pos = torch.abs(t_pos_mean_kl1 - p_pos_mean_kl1) / p_pos_mean_kl1
    deta_neg = torch.abs(t_neg_mean_kl1 - p_neg_mean_kl1) / p_neg_mean_kl1
    return deta_pos, deta_neg

def label_calibration(vob_kl1, deta_pos, deta_neg, pseudo_label,t_label,b):

    p_pos_mean = torch.sum(vob_kl1 * pseudo_label)/(torch.sum(pseudo_label)-b)
    p_neg_mean = torch.sum(vob_kl1 * (1-pseudo_label))/(torch.sum(1-pseudo_label))
    p_pos_cal = p_pos_mean * torch.abs(1 - deta_pos)
    p_neg_cal = p_neg_mean * (1 + deta_neg)
    p_pos_cal_mask = torch.lt(vob_kl1, p_pos_cal).float()
    p_neg_cal_mask = torch.gt(vob_kl1, p_neg_cal).float()
    #calcuate error
    #previous error
    prev_error = t_label - pseudo_label
    prev_dif_error = torch.sum((prev_error.float()==1).float()) / torch.sum(1-pseudo_label)
    prev_sam_error = torch.sum((prev_error.float()==-1).float()) / torch.sum(pseudo_label)
    #cal error--use center
    #cal error---use therthod
    dif_error_1 = p_pos_cal_mask * (t_label - pseudo_label)
    dif_error_2 = p_neg_cal_mask * (t_label - pseudo_label)
    dif_error = (torch.sum((dif_error_1==1).float()) + torch.sum((dif_error_2==1).float())) / torch.sum(p_pos_cal_mask * (1-pseudo_label) + p_neg_cal_mask * (1- pseudo_label))
    same_error = (torch.sum((dif_error_1==-1).float()) + torch.sum((dif_error_2==-1).float())) / torch.sum(p_pos_cal_mask * pseudo_label + p_neg_cal_mask * pseudo_label)
    return p_pos_cal_mask, p_neg_cal_mask,prev_dif_error,prev_sam_error, dif_error,same_error

def label_calibration_1(vob_kl1, pseudo_label,t_label,b, center_label):
    #calcuate error
    #previous error
    prev_error = t_label - pseudo_label
    prev_dif_error = torch.sum((prev_error.float()==1).float()) / torch.sum(1-pseudo_label)
    prev_sam_error = torch.sum((prev_error.float()==-1).float()) / torch.sum(pseudo_label)
    #cal error--use center
    dif_error_1 = t_label - (center_label * pseudo_label)
    dif_error = torch.sum((dif_error_1.float() == 1).float()) / torch.sum(1 - pseudo_label*center_label)
    same_error = torch.sum((dif_error_1.float() ==-1).float()) / torch.sum(pseudo_label * center_label)
    return prev_dif_error,prev_sam_error, dif_error,same_error

def compute_masktensor(center_kl1, batch_size, t_label, pseudo_label):
    center_probability = F.softmax(-1 * center_kl1, dim = -1)
    center_probability[center_probability > 0.96] = 1 # threshod = 0.9
    center_probability[center_probability <= 0.96] = 0
    high_proba = torch.nonzero(center_probability == 1, as_tuple = True) # 这里应该判断一下是否为空
    if len(high_proba[0]) == 0:
        intra_position_mask = torch.zeros(batch_size, batch_size)
        inter_position_mask = torch.zeros(batch_size, batch_size)
    else:
        high_proba_position = high_proba[0]
        high_proba_label = high_proba[1]
        modified_label = -1 * torch.ones(batch_size)
        for i in range(len(high_proba_position)):
            modified_label[high_proba_position[i]] = high_proba_label[i]
        modified_label_binary = (modified_label.unsqueeze(0) == modified_label.unsqueeze(1)).float()
        modified_label_expand1 = modified_label.unsqueeze(0).expand(batch_size,-1)
        modified_label_expand2 = modified_label.unsqueeze(1).expand(-1,batch_size)
        abandoned_position1 = (modified_label_expand1 == -1).float()
        abandoned_position2 = (modified_label_expand2 == -1).float()
        mask_abandoned = (1-abandoned_position1) * (1-abandoned_position2)
        intra_position_mask = modified_label_binary * mask_abandoned # intra = pull in instance, inter = pull out instance
        diag_intra = torch.diag(intra_position_mask)
        a_diag_intra = torch.diag_embed(diag_intra)
        intra_position_mask = intra_position_mask - a_diag_intra
        inter_position_mask = (1-modified_label_binary) * mask_abandoned
    return intra_position_mask, inter_position_mask

def compute_error(intra_position_mask, inter_position_mask, t_label,  pseudo_label):
    prev_error = t_label - pseudo_label
    prev_dif_error = torch.sum((prev_error.float() == 1).float()) / torch.sum(1 - pseudo_label)
    prev_sam_error = torch.sum((prev_error.float() == -1).float()) / torch.sum(pseudo_label)
    if torch.sum(intra_position_mask) == 0:
        same_error = torch.tensor(0)
    else:
        after_error_1 = t_label - intra_position_mask
        same_error = torch.sum((after_error_1.float() == -1).float()) / (torch.sum(intra_position_mask)+100)
    if torch.sum(inter_position_mask) == 0:
        dif_error = torch.tensor(0)
    else:
        after_error_2 = (1 - t_label) - inter_position_mask
        dif_error = torch.sum((after_error_2 == -1).float()) / torch.sum(inter_position_mask)
    return prev_dif_error,prev_sam_error, dif_error, same_error


def compute_kld2(p_logit, q_logit):
    p = F.softmax(p_logit, dim = 1) # (B, n_class) 
    q = F.softmax(q_logit, dim = 1) # (B, n_class)
    return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = 1)
    
def compute_kld(p_logit, q_logit):
    p = F.softmax(p_logit, dim = -1) # (B, B, n_class) 
    q = F.softmax(q_logit, dim = -1) # (B, B, n_class)
    return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim = -1) # (B, B)

def compute_kld3(p_logit, q_logit):
    M_logit = (p_logit +  q_logit)/2
    M = F.softmax(M_logit, dim = -1)
    p = F.softmax(p_logit, dim = -1) # (B, B, n_class)
    q = F.softmax(q_logit, dim = -1) # (B, B, n_class)
    JS_1 = torch.sum(p * (torch.log(p + 1e-16) - torch.log(M + 1e-16)), dim = -1)
    JS_2 = torch.sum(q * (torch.log(q + 1e-16) - torch.log(M + 1e-16)), dim = -1)
    return (JS_1+JS_2)/2 # (B, B)
    

def data_split(examples, ratio = [0.6, 0.2, 0.2]):
    from sklearn.utils import shuffle
    examples = shuffle(examples)
    train_num = int(len(examples) * ratio[0])
    dev_num = int(len(examples) * ratio[1])
    train_examples = examples[:train_num]
    dev_examples = examples[train_num:train_num + dev_num]
    test_examples = examples[train_num + dev_num:]
    return train_examples, dev_examples, test_examples

def data_split2(Examples, ratio = [6/7, 1/7]):
    from sklearn.utils import shuffle
    Examples.example_with_adv = shuffle(Examples.example_with_adv)
    dev_num = int(len(Examples.example_with_adv) * ratio[0])
    dev_examples = Examples.example_with_adv[:dev_num]
    test_examples = Examples.example_with_adv[dev_num:]
    return dev_examples, test_examples

def data_split_either_label(Examples, ratio = [1/2, 1/2]):
    from sklearn.utils import shuffle
    Examples = shuffle(Examples)
    dev_num = int(len(Examples) * ratio[0])
    dev_examples = Examples[:dev_num]
    test_examples = Examples[dev_num:]
    return dev_examples, test_examples


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def L2Reg(net):
    reg_loss = 0
    for name, params in net.named_parameters():
        if name[-4:] != 'bias':
            reg_loss += torch.sum(torch.pow(params, 2))
    return reg_loss

def flush():#清下gpu
    '''
    :rely on:import torch
    '''
    torch.cuda.empty_cache()
    '''
    或者在命令行中
    ps aux | grep python
    kill -9 [pid]
    或者
    nvidia-smi --gpu-reset -i [gpu_id]
    '''
    
def clean_text(text): # input is word list
    ret = []
    for word in text:
        normalized_word = re.sub(u"([^\u0020-\u007f])", "", word)
        if normalized_word == '' or normalized_word == ' ' or normalized_word == '    ':
            normalized_word = '[UNK]'
        ret.append(normalized_word)
    return ret
    

def get_pseudo_label(pseudo_label_list, idxes):    
    ret = []
    for idx in idxes:
        ret.append(pseudo_label_list[idx])
    ret = torch.tensor(ret).long()
    return ret


def eval_acc(L1, L2):
    sum = np.sum(L1[:]==L2[:])
    return sum/len(L2)

def eval_p_r_f1(ground_truth, label_pred):
    def _bcubed(i):
        C = label_pred[i]
        n = 0
        for j in range(len(label_pred)):
            if label_pred[j] != C:
                continue
            if ground_truth[i] == ground_truth[j]:
                n += 1
        p = n / num_cluster[C]
        r = n / num_class[ground_truth[i]]
        return p, r

    ground_truth -= 1
    label_pred -= 1
    num_class = [0] * 16
    num_cluster = [0] * 16
    for c in ground_truth:
        num_class[c] += 1
    for c in label_pred:
        num_cluster[c] += 1

    precision = recall = fscore = 0.

    for i in range(len(label_pred)):
        p, r = _bcubed(i)
        precision += p
        recall += r
    precision = precision / len(label_pred)
    recall = recall / len(label_pred)
    fscore = 2 * precision * recall / (precision + recall) # f1 score

    return precision, recall, fscore

def tsne_and_save_data(data, label, epoch):
    '''
    Args:
        data: ndarray (num_examples, dims)
        label: ndarray (num_examples)
    '''
    from sklearn.manifold import TSNE
    import numpy as np
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data) # (num_examples, n_components)
    #fig, ax = plt.subplots()
    x, y, c = result[:, 0], result[:, 1], label
    f = "./{}.npz".format(epoch)
    np.savez(f, x=x, y=y, true_label=label)



def tsne_and_save_pic(data, label, new_class, acc):
    '''
    Args:
        data: ndarray (num_examples, dims)
        label: ndarray (num_examples)
    '''
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    def mscatter(x,y,ax=None, m=None, **kw):
        import matplotlib.markers as mmarkers
        if not ax: ax=plt.gca()
        sc = ax.scatter(x,y,**kw)
        if (m is not None) and (len(m)==len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                            marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data) # (num_examples, n_components)
    fig, ax = plt.subplots()
    x, y, c = result[:, 0], result[:, 1], label
    '''
    m = {1:'o',2:'s',3:'D',4:'+'}
    cm = list(map(lambda x:m[x],c))#将相应的标签改为对应的marker
    '''
    #scatter = mscatter(x, y, c=c, m=cm, ax=ax,cmap=plt.cm.RdYlBu)
    scatter = mscatter(x, y, c=c, ax=ax,cmap=plt.cm.Paired, s = 9.0)
    if new_class:
        ax.set_title("t-sne for new class, result:{}".format(acc))
        plt.savefig( 't-sne-new-class.svg' ) 
    else:
        ax.set_title("t-sne for known class, result:{}".format(acc))
        plt.savefig( 't-sne-known-class.svg' ) 


def k_means(data, clusters): # range from 0 to clusters - 1
    #return KMeans(n_clusters=clusters,random_state=0,algorithm='full').fit(data).predict(data)
    return KMeans(n_clusters=clusters,algorithm='full').fit(data).predict(data)



def maplabels(L1, L2):
    Label1 = np.unique(L1)
    Label2 = np.unique(L2)
    nClass1 = len(Label1)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2*ind_cla1)

    row_ind, col_ind = linear_sum_assignment(-G.T)
    newL2 = np.zeros(L2.shape, dtype=int)
    for i in range(nClass2):
        for j in range(len(L2)):
            if L2[j] == row_ind[i]:
                newL2[j] = col_ind[i]
    return newL2


def endless_get_next_batch(loaders, iters, batch_size):
    try:
        data = next(iters)
    except StopIteration:
        iters = iter(loaders)
        data = next(iters)
    # In PyTorch 0.3, Batch Norm no longer works for size 1 batch,
    # so we will skip leftover batch of size < batch_size
    if len(data[0]) < batch_size:
        return endless_get_next_batch(loaders, iters, batch_size)
    return data, iters

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def _worker_init_fn_():
    import random
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // (2**32-1)
    random.seed(torch_seed)
    np.random.seed(np_seed)


def mask_tokens(inputs, tokenizer, not_mask_pos=None):
    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15)
    special_tokens_mask = tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    if not_mask_pos is None:
        masked_indices = torch.bernoulli(probability_matrix).bool()
    else:
        masked_indices = torch.bernoulli(probability_matrix).bool() & (~(not_mask_pos.bool())) # ** can't mask entity marker **
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def fill_bert_entity_spans(text_ori, SUBJECT, OBJECT, sentence_tokens):
    bert_tokens = sentence_tokens[1:]
    actual_tokens = text_ori
    head_span = SUBJECT
    tail_span = OBJECT
    actual_token_index, bert_token_index, actual_to_bert = 0, 0, []
    while actual_token_index < len(actual_tokens):
        start, end = bert_token_index, bert_token_index
        actual_token = actual_tokens[actual_token_index]
        token_index = 0
        while token_index < len(actual_token):
            bert_token = bert_tokens[bert_token_index]
            if bert_token.startswith('##'):
                bert_token = bert_token[2:]
            if bert_token.lower() != actual_token[token_index:token_index+len(bert_token)].lower():
                x = x+1
            assert(bert_token.lower()==actual_token[token_index:token_index+len(bert_token)].lower())
            end = bert_token_index
            token_index += len(bert_token)
            bert_token_index += 1
        actual_to_bert.append([start, end])
        actual_token_index += 1
    head_span = (1 + actual_to_bert[head_span[0]][0], 1 + actual_to_bert[head_span[1]][1])
    tail_span = (1 + actual_to_bert[tail_span[0]][0], 1 + actual_to_bert[tail_span[1]][1])
    not_mask_position = [0] *len(sentence_tokens)
    not_mask_position[head_span[0]:head_span[1]+1] = [1]*len(range(head_span[1]+1-head_span[0]))
    not_mask_position[tail_span[0]:tail_span[1]+1] = [1]*len(range(tail_span[1]+1-tail_span[0]))
    return actual_to_bert, not_mask_position


# generate adv examples
def get_ori_feat(text_ori, SUBJECT, OBJECT, bert_tokens_1, bert_tokens_2, mlm_model, tokenizer):
    sentence_tokens = []
    sentence_prompt_tokens = []
    sen_input_token_type = []
    sen_prompt_token_type = []
    sentence_tokens.append("[CLS]")
    sen_input_token_type.append(0)
    for token in bert_tokens_1:
        sentence_tokens.append(token)
        sen_input_token_type.append(0)
    sentence_tokens.append("[SEP]")
    sen_input_token_type.append(0)
    for token in bert_tokens_2:
        sentence_prompt_tokens.append(token)
        sen_prompt_token_type.append(1)
    sentence_prompt_tokens.append("[SEP]")
    sen_prompt_token_type.append(1)

    sentence_input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
    sentence_prompt_ids = tokenizer.convert_tokens_to_ids(sentence_prompt_tokens)
    actual_to_bert, not_mask_position = fill_bert_entity_spans(text_ori, SUBJECT, OBJECT, sentence_tokens)
    # cat
    s_input_with_prompt_ids = sentence_input_ids + sentence_prompt_ids
    sentence_token_type_ids = sen_input_token_type + sen_prompt_token_type
    input_mask_all = [1] * (len(sentence_input_ids) + len(sentence_prompt_ids))
    Individual_len = len(input_mask_all) +5
    # Zero-pad up to the sequence length
    while len(input_mask_all) < Individual_len :
        input_mask_all.append(0)
        s_input_with_prompt_ids.append(0)
        sentence_token_type_ids.append(0)
    assert len(input_mask_all) == Individual_len
    assert len(s_input_with_prompt_ids) == Individual_len
    assert len(sentence_token_type_ids) == Individual_len
    input_mask_all = torch.Tensor(input_mask_all)
    s_input_with_prompt_ids = torch.Tensor(s_input_with_prompt_ids)
    sentence_token_type_ids = torch.Tensor(sentence_token_type_ids)
    input_mask_all = torch.unsqueeze(input_mask_all, 0).type(torch.LongTensor).to('cuda')
    s_input_with_prompt_ids = torch.unsqueeze(s_input_with_prompt_ids, 0).type(torch.LongTensor).to('cuda')
    sentence_token_type_ids = torch.unsqueeze(sentence_token_type_ids, 0).type(torch.LongTensor).to('cuda')

    results = mlm_model(s_input_with_prompt_ids, token_type_ids=sentence_token_type_ids, attention_mask = input_mask_all,output_hidden_states=True)
    encoder_layers = results.hidden_states[11]  # get feature in layer 11 (batch_size, seq_len, bert_embedding)
    _, mask_idx = (s_input_with_prompt_ids == 103).nonzero(as_tuple=True)  # 找到mask位置
    mask_ori_feat = encoder_layers[:,mask_idx[0], :]
    return mask_ori_feat, not_mask_position,actual_to_bert, Individual_len

def _get_masked(ori_text, remove_all_index):
    masked_words = []
    for i in remove_all_index:
        masked_words.append(ori_text[0:i] + ['[UNK]'] + ori_text[i + 1:])
    return masked_words

def get_important_scores(SUBJECT, OBJECT, ori_text, text_ori_prompt, ori_rel_feat, tokenizer,text_len,mlm_model):
    position_index1 = list(range(len(ori_text)))
    del position_index1[OBJECT[0]: OBJECT[1]+1] #移除obj
    position_index2 = list(range(len(ori_text)))
    del position_index2[SUBJECT[0]: SUBJECT[1] + 1] #移除subj
    remove_all_index = list(set(position_index1).intersection(set(position_index2)))
    masked_words = _get_masked(ori_text, remove_all_index)
    texts = [' '.join(words) for words in masked_words]  # list of text of masked words
    all_input_ids = []
    all_masks = []
    all_segs = []
    for text in texts:
        inputs = tokenizer.encode_plus(text, text_ori_prompt, add_special_tokens=True, max_length = text_len, padding= 'max_length')
        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs['attention_mask']
        all_input_ids.append(input_ids)
        all_masks.append(attention_mask)
        all_segs.append(token_type_ids)
    seqs = torch.tensor(all_input_ids, dtype=torch.long).to('cuda')
    masks = torch.tensor(all_masks, dtype=torch.long).to('cuda')
    segs = torch.tensor(all_segs, dtype=torch.long).to('cuda')
    # Run prediction for full data
    results = mlm_model(seqs, token_type_ids=segs, attention_mask = masks, output_hidden_states=True)
    encoder_layers = results.hidden_states[11]  # get feature in layer 11 (batch_size, seq_len, bert_embedding)
    _, mask_idx = (seqs == 103).nonzero(as_tuple=True)
    with_UNK_feat = encoder_layers[torch.arange(len(texts)), mask_idx, :]
    Diff_score = torch.cosine_similarity(ori_rel_feat, with_UNK_feat)
    score_of_index = sorted(enumerate(Diff_score), key=lambda x: x[1], reverse=False)
    list_of_index = [score_of_index[i][0] for i in range(len(score_of_index))]
    rank_index=[]
    for ii in list_of_index:
        rank_index.append(remove_all_index[ii])
    return rank_index

def get_syn_word_list(word):
    syn_word_list = []
    for _, sys in enumerate(wn.synsets(word)):
        for term in sys.lemma_names():
            if word.lower() not in term.lower() and word.lower() not in term.lower() and len(term.split('_'))==1 and len(term.split('-'))==1:
                syn_word_list.append(term)
    return list(set(syn_word_list))

def get_antonym_word_list(word):
    antonym_word_list = []
    for syn in wn.synsets(word):
        for term in syn._lemmas:
            if term.antonyms():
                antonym_word_list.append(term.antonyms()[0].name())
        for sim_syn in syn.similar_tos():
            for term in sim_syn._lemmas:
                if term.antonyms():
                    antonym_word_list.append(term.antonyms()[0].name())
    return list(set(antonym_word_list))


class EntityMarker():
    """Converts raw text to BERT-input ids and finds entity position.

    Attributes:
        tokenizer: Bert-base tokenizer.
        h_pattern: A regular expression pattern -- * h *. Using to replace head entity mention.
        t_pattern: A regular expression pattern -- ^ t ^. Using to replace tail entity mention.
        err: Records the number of sentences where we can't find head/tail entity normally.
        args: Args from command line.
    """

    def __init__(self, args=None):
        # self.tokenizer = tokenizer  # "/data/pan/zh/REContext/bert-base-uncased"
        self.h_pattern = re.compile("\* h \*")
        self.t_pattern = re.compile("\^ t \^")
        self.err = 0
        self.args = args

    def tokenize(self, example, h_blank, t_blank, ht_blank, tokenizer):
        """Tokenizer for `CM`(typically), `CT`, `OC` settings.

        This function converts raw text to BERT-input ids and uses entity-marker to highlight entity
        position and randomly raplaces entity metion with special `BLANK` symbol. Entity mention can
        be entity type(If h_type and t_type are't none). And this function returns ids that can be
        the inputs to BERT directly and entity postion.

        Args:
            raw_text: A python list of tokens.
            h_pos_li: A python list of head entity postion. For example, h_pos_li maybe [2, 6] which indicates
                that heah entity mention = raw_text[2:6].
            t_pos_li: The same as h_pos_li.
            h_type: Head entity type. This argument is used when we use `CT` settings as described in paper.
            t_type: Tail entity type.
            h_blank: Whether convert head entity mention to `BLANK`.
            t_blank: Whether convert tail entity mention to `BLANK`.

        Returns:
            tokenized_input: Input ids that can be the input to BERT directly.
            h_pos: Head entity position(head entity marker start positon).
            t_pos: Tail entity position(tail entity marker start positon).

        Example:
            raw_text: ["Bill", "Gates", "founded", "Microsoft", "."]
            h_pos_li: [0, 2]
            t_pos_li: [3, 4]
            h_type: None
            t_type: None
            h_blank: True
            t_blank: False

            Firstly, replace entity mention with special pattern:
            "* h * founded ^ t ^ ."

            Then, replace pattern:
            "[CLS] [unused0] [unused4] [unused1] founded [unused2] microsoft [unused3] . [SEP]"

            Finally, find the postions of entities and convert tokenized sentence to ids:
            [101, 1, 5, 2, 2631, 3, 7513, 4, 1012, 102]
            h_pos: 1
            t_pos: 5
        """
        tokens = []
        h_mention = []
        t_mention = []
        raw_text = example.ori_text
        h_pos_li = example.head_span
        t_pos_li = example.tail_span
        h_type1 = example.head_type
        t_type1 = example.tail_type
        for i, token in enumerate(raw_text):
            token = token.lower()
            if i == h_pos_li[0]:
                tokens += ['*', 'h', '*']
                h_mention.append(token)
                continue
            if i >= h_pos_li[0] and i <= h_pos_li[-1]:
                h_mention.append(token)
                continue
            if i == t_pos_li[0]:
                tokens += ['^', 't', '^']
                t_mention.append(token)
                continue
            if i >= t_pos_li[0] and i <= t_pos_li[-1]:
                t_mention.append(token)
                continue
            tokens.append(token)
        text = " ".join(tokens)
        h_mention = " ".join(h_mention)
        t_mention = " ".join(t_mention)

        # tokenize
        tokenized_text = tokenizer.tokenize(text)
        tokenized_head = tokenizer.tokenize(h_mention)
        tokenized_tail = tokenizer.tokenize(t_mention)

        if h_type1 is not None and t_type1 is not None and (h_type1 != 0 and h_type1 != []) and (t_type1 != 0 and t_type1 != []):
            tokenized_head_type = tokenizer.tokenize(h_type1)
            tokenized_tail_type = tokenizer.tokenize(t_type1)
            h_type = " ".join(tokenized_head_type)
            t_type = " ".join(tokenized_tail_type)
        else:
            h_type = h_type1
            t_type = t_type1

        p_text = " ".join(tokenized_text)
        p_head = " ".join(tokenized_head)
        p_tail = " ".join(tokenized_tail)

        if h_blank:
            p_text = self.h_pattern.sub(h_type, p_text)
            p_text = self.t_pattern.sub(p_tail, p_text)
            h_entity = h_type
            t_entity = p_tail
            t_prompt = (h_type + " [MASK] " + p_tail + " [SEP]").split()
        elif t_blank:
            p_text = self.h_pattern.sub(p_head, p_text)
            p_text = self.t_pattern.sub(t_type, p_text)
            h_entity = p_head
            t_entity = t_type
            t_prompt = (p_head + " [MASK] " + t_type + " [SEP]").split()
        elif ht_blank:
            p_text = self.h_pattern.sub(h_type, p_text)
            p_text = self.t_pattern.sub(t_type, p_text)
            h_entity = h_type
            t_entity = t_type
            t_prompt = (h_type + " [MASK] " + t_type + " [SEP]").split()
        else:
            p_text = self.h_pattern.sub(p_head, p_text)
            p_text = self.t_pattern.sub(p_tail, p_text)
            h_entity = p_head
            t_entity = p_tail
            t_prompt = (p_head + " [MASK] " + p_tail + " [SEP]").split()
        f_text = ("[CLS] " + p_text + " [SEP]").split()
        #tokenized_input = self.tokenizer.convert_tokens_to_ids(f_text)
        return f_text, t_prompt, h_entity, t_entity


    def adv_tokenize(self, example, h_blank, t_blank, ht_blank, tokenizer):
        """Tokenizer for `CM`(typically), `CT`, `OC` settings.

        This function converts raw text to BERT-input ids and uses entity-marker to highlight entity
        position and randomly raplaces entity metion with special `BLANK` symbol. Entity mention can
        be entity type(If h_type and t_type are't none). And this function returns ids that can be
        the inputs to BERT directly and entity postion.

        Args:
            raw_text: A python list of tokens.
            h_pos_li: A python list of head entity postion. For example, h_pos_li maybe [2, 6] which indicates
                that heah entity mention = raw_text[2:6].
            t_pos_li: The same as h_pos_li.
            h_type: Head entity type. This argument is used when we use `CT` settings as described in paper.
            t_type: Tail entity type.
            h_blank: Whether convert head entity mention to `BLANK`.
            t_blank: Whether convert tail entity mention to `BLANK`.

        Returns:
            tokenized_input: Input ids that can be the input to BERT directly.
            h_pos: Head entity position(head entity marker start positon).
            t_pos: Tail entity position(tail entity marker start positon).

        Example:
            raw_text: ["Bill", "Gates", "founded", "Microsoft", "."]
            h_pos_li: [0, 2]
            t_pos_li: [3, 4]
            h_type: None
            t_type: None
            h_blank: True
            t_blank: False

            Firstly, replace entity mention with special pattern:
            "* h * founded ^ t ^ ."

            Then, replace pattern:
            "[CLS] [unused0] [unused4] [unused1] founded [unused2] microsoft [unused3] . [SEP]"

            Finally, find the postions of entities and convert tokenized sentence to ids:
            [101, 1, 5, 2, 2631, 3, 7513, 4, 1012, 102]
            h_pos: 1
            t_pos: 5
        """
        tokens = []
        h_mention = []
        t_mention = []
        raw_text = example.text_adv[0]
        h_pos_li = example.head_span
        t_pos_li = example.tail_span
        h_type1 = example.head_type
        t_type1 = example.tail_type
        for i, token in enumerate(raw_text):
            token = token.lower()
            if i == h_pos_li[0]:
                tokens += ['*', 'h', '*']
                h_mention.append(token)
                continue
            if i >= h_pos_li[0] and i <= h_pos_li[-1]:
                h_mention.append(token)
                continue
            if i == t_pos_li[0]:
                tokens += ['^', 't', '^']
                t_mention.append(token)
                continue
            if i >= t_pos_li[0] and i <= t_pos_li[-1]:
                t_mention.append(token)
                continue
            tokens.append(token)
        text = " ".join(tokens)
        h_mention = " ".join(h_mention)
        t_mention = " ".join(t_mention)
        # tokenize
        tokenized_text = tokenizer.tokenize(text)
        tokenized_head = tokenizer.tokenize(h_mention)
        tokenized_tail = tokenizer.tokenize(t_mention)

        if h_type1 is not None and t_type1 is not None and (h_type1 != 0 and h_type1 != []) and (t_type1 != 0 and t_type1 != []):
        # if h_type1 is not None and t_type1 is not None and h_type1 !=0 and t_type1 != 0:
            tokenized_head_type = tokenizer.tokenize(h_type1)
            tokenized_tail_type = tokenizer.tokenize(t_type1)
            h_type = " ".join(tokenized_head_type)
            t_type = " ".join(tokenized_tail_type)
        else:
            h_type = h_type1
            t_type = t_type1

        p_text = " ".join(tokenized_text)
        p_head = " ".join(tokenized_head)
        p_tail = " ".join(tokenized_tail)

        if h_blank:
            p_text = self.h_pattern.sub(h_type, p_text)
            p_text = self.t_pattern.sub(p_tail, p_text)
            h_entity = h_type
            t_entity = p_tail
            t_prompt = (h_type + " [MASK] " + p_tail + " [SEP]").split()
        elif t_blank:
            p_text = self.h_pattern.sub(p_head, p_text)
            p_text = self.t_pattern.sub(t_type, p_text)
            h_entity = p_head
            t_entity = t_type
            t_prompt = (p_head + " [MASK] " + t_type + " [SEP]").split()
        elif ht_blank:
            p_text = self.h_pattern.sub(h_type, p_text)
            p_text = self.t_pattern.sub(t_type, p_text)
            h_entity = h_type
            t_entity = t_type
            t_prompt = ( h_type + " [MASK] " + t_type + " [SEP]").split()
        else:
            p_text = self.h_pattern.sub(p_head, p_text)
            p_text = self.t_pattern.sub(p_tail, p_text)
            h_entity = p_head
            t_entity = p_tail
            t_prompt = (p_head + " [MASK] " + p_tail + " [SEP]").split()
        f_text = ("[CLS] " + p_text + " [SEP]").split()
        #tokenized_input = self.tokenizer.convert_tokens_to_ids(f_text)
        return f_text, t_prompt, h_entity, t_entity


def match(y_true, y_pred):
    #y_true = y_true.dtype(np.int64)
    #y_pred = y_pred.dtype(np.int64)
    y_true = y_true.numpy()
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y = np.zeros(y_true.shape[0])
    for i in range(y_pred.size):
        for j in row_ind:
            if y_true[i] == col_ind[j]:
                new_y[i] = row_ind[j]
    new_y = torch.from_numpy(new_y).long()
    new_y = new_y.view(new_y.size()[0])
    return new_y
import multiprocessing
import pickle
from typing import Counter
import numpy as np
import sklearn
import json
import torch
from data import ASPECT_BEGIN,ASPECT_IN,OPINION_BEGIN,OPINION_IN
IGNORE_INDEX = -1
PADDING_INDEX = 0
ASPECT=[ASPECT_BEGIN,ASPECT_IN]
OPINION=[OPINION_BEGIN,OPINION_IN]

# 对抗学习
class FGM(object):
    """Reference: https://arxiv.org/pdf/1605.07725.pdf"""
    def __init__(self,
                 model,
                 emb_name='word_embeddings.',
                 epsilon=1.0):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self):
        """Add adversity."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """restore embedding"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if self.emb_name in name:
                    param.grad = self.grad_backup[name]
                else:
                    param.grad += self.grad_backup[name]


def get_aspects(tags, length, ignore_index=-1):
    spans = []
    start = -1
    for i in range(length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 1:
            if start == -1:
                start = i
        elif tags[i][i] != 1:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length-1])
    return spans


def get_opinions(tags, length, ignore_index=-1):
    spans = []
    start = -1
    for i in range(length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 2:
            if start == -1:
                start = i
        elif tags[i][i] != 2:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length-1])
    return spans


class Metric():
    def __init__(self, args, predictions, ner_predictions, goldens, ner_goldens, bert_lengths, sen_lengths, tokens_ranges, ignore_index=-1, max_sequence_len=100):
        self.args = args
        self.predictions = predictions
        self.ner_predictions = ner_predictions
        self.goldens = goldens
        self.ner_goldens = ner_goldens
        self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = -1
        self.max_sequence_len = max_sequence_len
        self.data_num = len(self.predictions)


    def get_spans(self, tags, length, token_range, types):
        spans = []
        start = IGNORE_INDEX
        begin, mid = types
        for i in range(length):
            l, r = token_range[i]
            if l >= self.max_sequence_len or r >= self.max_sequence_len:
                start = IGNORE_INDEX
                break
            if tags[l] == IGNORE_INDEX:
                continue
            elif tags[l] == begin:
                if start != IGNORE_INDEX:
                    spans.append([start, i - 1])
                start = i
            elif tags[l] not in types:
                if start != IGNORE_INDEX:
                    spans.append([start, i - 1])
                    start = IGNORE_INDEX
        if start != IGNORE_INDEX:
            spans.append([start, length - 1])
        return spans


    def find_pair(self, tags, aspect_spans, opinion_spans, token_ranges):
        pairs = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 6
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                if tag_num[5] == 0: continue
                sentiment = -1
                pairs.append([al, ar, pl, pr, sentiment])
        return pairs

    def find_triplet(self, tags, aspect_spans, opinion_spans, token_ranges):
        triplets = []
        used_ops = set()
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 4
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                        
                
                if sum(tag_num[:]) == 0: continue
                sentiment = -1
                if tag_num[0] >= tag_num[1] and tag_num[0] >= tag_num[2] and tag_num[0] >= tag_num[3]:
                    continue
                if tag_num[1] >= tag_num[2] and tag_num[1] >= tag_num[3]:
                    sentiment = 1
                elif tag_num[2] >= tag_num[1] and tag_num[2] >= tag_num[3]:
                    sentiment = 2
                elif tag_num[3] >= tag_num[1] and tag_num[3] >= tag_num[2]:
                    sentiment = 3
                if sentiment == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                    input()
                triplets.append([al, ar, pl, pr, sentiment])
                used_ops.add((pl,pr))
        ops = set([(x[2],x[3]) for x in triplets if x[0]!=0])
        triplets = [x for x in triplets if not (x[0]==0 and (x[2],x[3]) in ops)]
        
        return triplets

    def score_aspect(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = self.get_spans(self.ner_goldens[i], self.sen_lengths[i], self.tokens_ranges[i], ASPECT)
            for spans in golden_aspect_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_aspect_spans = self.get_spans(self.ner_predictions[i], self.sen_lengths[i], self.tokens_ranges[i], ASPECT)
            for spans in predicted_aspect_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_opinion(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_opinion_spans = self.get_spans(self.ner_goldens[i], self.sen_lengths[i], self.tokens_ranges[i], OPINION)
            for spans in golden_opinion_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_opinion_spans = self.get_spans(self.ner_predictions[i], self.sen_lengths[i], self.tokens_ranges[i], OPINION)
            for spans in predicted_opinion_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))
        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_uniontags(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        
        for i in range(self.data_num):
            golden_aspect_spans = self.get_spans(self.ner_goldens[i], self.sen_lengths[i], self.tokens_ranges[i], ASPECT)
            golden_opinion_spans = self.get_spans(self.ner_goldens[i], self.sen_lengths[i], self.tokens_ranges[i], OPINION)
            # if i == 39:
            #     json.dump(self.goldens[i],open('./gold_tags.json','w'))
                
            #     print(golden_aspect_spans)
            #     print(golden_opinion_spans)
            if self.args.task == 'pair':
                golden_tuples = self.find_pair(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                golden_tuples = self.find_triplet(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            for pair in golden_tuples:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))

            predicted_aspect_spans = self.get_spans(self.ner_predictions[i], self.sen_lengths[i], self.tokens_ranges[i], ASPECT)
            # if (0,0) not in predicted_aspect_spans:
                # predicted_aspect_spans.append((0,0))
            predicted_opinion_spans = self.get_spans(self.ner_predictions[i], self.sen_lengths[i], self.tokens_ranges[i], OPINION)
            # import pdb;pdb.set_trace()
            
            # if i == 3:
            #     json.dump(self.predictions[i],open('./predict_tags.json','w'))
            #     json.dump(self.tokens_ranges[i],open('./predict_ranges.json','w'))
                
            #     print(predicted_aspect_spans)
            #     print(predicted_opinion_spans)
            if self.args.task == 'pair':
                predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                predicted_tuples = self.find_triplet(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
                # if i == 39:
                #     print(predicted_tuples)
            for pair in predicted_tuples:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))

        if self.args.debug:
            json.dump(sorted(predicted_set),open('./predict.json','w'))
            json.dump(sorted(golden_set),open('./golden.json','w'))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
#coding utf-8

import json, os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import trange

from data import load_data_instances, DataIterator
# from model import MultiInferBert
from biaffine import BiaffineModel as MultiInferBert
from transformers import AdamW,get_linear_schedule_with_warmup
import utils
from focal_loss import FocalLoss
# from torch.optim.swa_utils import AveragedModel, SWALR

# 设置随机种子，便于复现结果以及避免随机使得模型之间更加可比
import random
SEED = 2020

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def train(args):

    # load dataset
    #[json.loads(x) for x in open(args.prefix + args.dataset + '/lap_rest_train.jsonl')]#
    train_sentence_packs = [json.loads(x) for x in open(args.prefix + args.dataset + '/lap_rest_dev.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/acdc_train.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/absa_train.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/charger_train.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/lap_rest_train.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/headphone_train.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/lamp_train.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/pen_train.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/starlink_train.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/battery_train.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/connectivity_train.jsonl')] \


    random.shuffle(train_sentence_packs)
    dev_sentence_packs = [json.loads(x) for x in open(args.prefix + args.dataset + '/absa_dev.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/lamp_dev.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/charger_dev.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/starlink_dev.jsonl')] \

    test_sentence_packs = [json.loads(x) for x in open(args.prefix + args.dataset + '/pen_dev.jsonl')]
    instances_train = load_data_instances(train_sentence_packs, args)
    instances_dev = load_data_instances(dev_sentence_packs, args)
    instances_test = load_data_instances(test_sentence_packs, args)
    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)
    testset = DataIterator(instances_test, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model = MultiInferBert(args).to(args.device)

    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters(), 'lr': 5e-5},
    #     {'params': model.cls_linear.parameters()}
    # ], lr=5e-5)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr,correct_bias=True)

    gradient_accumulation_steps = 1
    warmup_ratio = 0.06
    t_total = trainset.batch_count // gradient_accumulation_steps * args.epochs
    warmup_steps =  int(t_total * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=t_total,
                )


    best_joint_f1 = 0
    best_joint_epoch = 0
    early_stop = args.early_stop
    if args.focal_loss > 0:
        loss_fn = FocalLoss(gamma=args.focal_loss,ignore_index=-1)
    # model.load_state_dict(torch.load('./savemodel/tmp.bin'))
    # swa_model = AveragedModel(model).to(args.device)
    # swa_start = 5
    # swa_scheduler = SWALR(optimizer, swa_lr=2e-6)
    gradient_accumulation_steps = args.gradient_accumulation_steps
    use_fgm = args.use_fgm
    use_fp16 = args.use_fp16
    if use_fgm:
        fgm = utils.FGM(model,epsilon=args.fgm_epsilon)
    
    if use_fp16:
        from torch.cuda import amp
        scaler = amp.GradScaler()

    for i in range(args.epochs):
        if early_stop == 0:
            print('Early stopped!!')
            break
        print('Epoch:{}'.format(i))
        trainset.shuffle()
        for step in trange(trainset.batch_count):
            tokens, lengths, masks, sens_lens, token_ranges, tags = trainset.get_batch(step)

            def get_loss(model, tokens, masks, tags):
                preds = model(tokens, masks)
                preds_flatten = preds.reshape([-1, preds.shape[3]])
                tags_flatten = tags.reshape([-1])

                if args.focal_loss > 0:
                    loss = loss_fn(preds_flatten, tags_flatten)
                else:
                    loss = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1,label_smoothing=args.label_smoothing)
                return loss

            if args.use_fp16:
                with amp.autocast():
                    loss = get_loss(model, tokens, masks, tags)
            else:
                loss = get_loss(model, tokens, masks, tags)

            if gradient_accumulation_steps > 1:
                loss /= gradient_accumulation_steps

            # Backward pass
            if use_fp16:
                scaler.scale(loss).backward(retain_graph=True if use_fgm else False)
            else:
                loss.backward(retain_graph=True if use_fgm else False)

            # 对抗训练
            if use_fgm:
                fgm.backup_grad()
                fgm.attack()
                model.zero_grad()
                adv_loss = get_loss(model, tokens, masks, tags)
                if gradient_accumulation_steps > 1:
                    adv_loss /= gradient_accumulation_steps
                if use_fp16:
                    scaler.scale(adv_loss).backward()
                else:
                    adv_loss.backward()
                fgm.restore_grad()
                fgm.restore()

            if (step+1)%gradient_accumulation_steps == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters and take a step using the computed gradient
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                model.zero_grad()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # # if i >= swa_start:
            # #     swa_model.update_parameters(model)  # To update parameters of the averaged model.
            # #     swa_scheduler.step()                # Switch to SWALR.
            # # else:
            # scheduler.step()

            if step == trainset.batch_count // 2 - 1:
                print('dev')
                joint_precision, joint_recall, joint_f1 = eval(model, devset, args)
                print('test')
                eval(model,testset,args)

                if joint_f1 > best_joint_f1:
                    model_path = args.model_dir + 'bert' + '_' + args.dataset + '_' + args.task + '.bin'
                    torch.save(model.state_dict(), model_path)
                    best_joint_f1 = joint_f1
                    best_joint_epoch = i
                    early_stop = args.early_stop
                else:
                    early_stop -= 1
                    if early_stop == 0:
                        break
        
        if early_stop == 0:
            print('Early stopped!!')
            break
        print('dev')
        joint_precision, joint_recall, joint_f1 = eval(model, devset, args)
        print('test')
        eval(model, testset, args)
        if joint_f1 > best_joint_f1:
            model_path = args.model_dir + 'bert' + '_' + args.dataset + '_' + args.task + '.bin'
            torch.save(model.state_dict(), model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
            early_stop = args.early_stop
        else:
            early_stop -= 1
            if early_stop == 0:
                break

    # Update bn statistics for the swa_model at the end
    # torch.optim.swa_utils.update_bn(loader, swa_model)
    # Use swa_model to make predictions on test data 
    # joint_precision, joint_recall, joint_f1 = eval(swa_model, devset, args)
    # if joint_f1 > best_joint_f1:
    #     model_path = args.model_dir + 'bert' + '_' + args.dataset + '_' + args.task + '.bin'
    #     torch.save(swa_model.state_dict(), model_path)


    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1))


def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        for i in range(dataset.batch_count):
            tokens, lengths, masks, sens_lens, token_ranges, tags = dataset.get_batch(i)
            preds = model(tokens, masks)
            preds = torch.argmax(preds, dim=3)
            all_preds.append(preds)
            all_labels.append(tags)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = utils.Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        precision, recall, f1 = metric.score_uniontags()
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   opinion_results[2]))
        print(args.task + '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1


def test(args):
    print("Evaluation on testset:")
    model_path = args.model_dir + 'bert' + '_' + args.dataset + '_' + args.task + '.bin'
    model = MultiInferBert(args).to(args.device)
    # model.load_state_dict(torch.load('./savemodel/triplet_v12.bin'))
    model.load_state_dict(torch.load(model_path))
    # model = torch.load(model_path).to(args.device)
    model.eval()

    results = {}


    print('all')
    sentence_packs = [json.loads(x) for x in open(args.prefix + args.dataset + '/absa_dev.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/acdc_dev.jsonl')] \
        + [json.loads(x) for x in open(args.prefix + args.dataset + '/charger_dev.jsonl')]
    instances = load_data_instances(sentence_packs, args)
    testset = DataIterator(instances, args)
    results['all'] = eval(model, testset, args)

    for name in ['absa','charger','acdc','headphone','lamp','pen','starlink','connectivity','battery']:
        print(name)
        sentence_packs = [json.loads(x) for x in open(args.prefix + args.dataset + f'/{name}_dev.jsonl')] 
        instances = load_data_instances(sentence_packs, args)
        testset = DataIterator(instances, args)
        results[name] = eval(model, testset, args)

    for k in results:
        print(k,results[k])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="pair", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--dataset', type=str, default="res14", choices=["shulex", "res14", "lap14", "res15", "res16"],
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=100,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_tokenizer_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--biaffine_size', type=int, default=300,
                        help='dimension of biaffine')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')                  
    parser.add_argument('--early_stop', type=int, default=10,
                        help='early stop')
    parser.add_argument('--nhops', type=int, default=1,
                        help='inference times')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=6,
                        help='label number')
    parser.add_argument('--do_lower_case', type=bool, default=False,
                        help='do lower case')          
    parser.add_argument('--focal_loss', type=float, default=0.0,
                        help='whether to use focal loss')  
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='whether to use label smoothing') 
    parser.add_argument('--use_fgm', type=bool, default=False,
                        help='FGM') 
    parser.add_argument('--fgm_epsilon', type=float, default=1.0,
                        help='fgm_epsilon') 
    parser.add_argument('--use_fp16', type=bool, default=False,
                        help='FP16')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='gradient_accumulation_steps')                    
    args = parser.parse_args()

    if args.task == 'triplet':
        args.class_num = 8

    if args.mode == 'train':
        train(args)
        test(args)
    else:
        test(args)

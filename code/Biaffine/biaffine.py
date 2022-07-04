import torch
import torch.nn

from transformers import  AutoModel

class Biaffine(torch.nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = torch.nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(input1.device)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(input2.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        # b * len1 * (out_feature + dim2)
        affine = self.linear(input1)
        # b * (len1 + out_features) * dim2
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        # b * dim2 * len2
        input2 = torch.transpose(input2, 1, 2)
        # b * (len1 + out_features) * len2
        biaffine = torch.bmm(affine, input2)
        # b * len2 * (len1 + out_features)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        # b , len1, len2, out_featuers
        return torch.transpose(biaffine,1,2)

class BiaffineModel(torch.nn.Module):
    def __init__(self, args):
        super(BiaffineModel, self).__init__()

        self.args = args
        self.bert = AutoModel.from_pretrained(args.bert_model_path)

        self.start = torch.nn.Linear(args.bert_feature_dim, args.biaffine_size)
        self.end = torch.nn.Linear(args.bert_feature_dim, args.biaffine_size)

        self.biaffine = Biaffine(args.biaffine_size, args.biaffine_size, args.class_num)

        self.dropout_output = torch.nn.Dropout(0.1)

    def biaffine_forward(self, start, end, mask):
        '''generate mask'''
        max_length = start.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])
        biaffine = self.biaffine(start,end)
        return biaffine * mask

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        bert_feature = outputs.last_hidden_state
        bert_feature = self.dropout_output(bert_feature)

        start = self.start(bert_feature)
        end = self.end(bert_feature)

        return self.biaffine_forward(start,end,attention_mask)
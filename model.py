import torch.nn as nn


class MaliciousCommentDetector(nn.Module):
    def __init__(self, bert, args):
        super(MaliciousCommentDetector, self).__init__()
        self.bert = bert
        self.dr_rate = args.dr_rate

        self.classifier = nn.Linear(args.hidden_size, args.num_labels)
        if self.dr_rate:
            self.dropout = nn.Dropout(p=self.dr_rate)

    def forward(self, token_ids, token_type_ids, attention_mask):
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=token_type_ids,
                              attention_mask=attention_mask, return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)

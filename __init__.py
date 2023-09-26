        tp = (y_true_one_hot * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true_one_hot) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true_one_hot) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true_one_hot * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        f1 = f1.detach()

        CE = nn.CrossEntropyLoss()(y_pred, y_true)
        loss = CE - f1.mean()
        
        return loss


class XLMCnnLstm(nn.Module):
    def __init__(self, num_classes):
        super(XLMCnnLstm, self).__init__()
        self.bert = AutoModel.from_pretrained("xlm-roberta-base")
        self.conv1d = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, bidirectional=False)
        self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(128, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state
        pooled_output = pooled_output.permute(0, 2, 1)  # Reshaping for Conv1d
        conv_output = self.conv1d(pooled_output)
        conv_output = torch.relu(conv_output)
        pooled_output = self.maxpool(conv_output)
        pooled_output = pooled_output.permute(0, 2, 1)  # Reshaping for LSTM
        lstm_output, _ = self.lstm(pooled_output)
        lstm_output = lstm_output[:, -1, :]  # Taking the last hidden state
        dropout_output = self.dropout(lstm_output)
        logits = self.dense(dropout_output)
        return logits

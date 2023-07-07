import torch
import torch.nn as nn
import torch.nn.functional as F

class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        y_true_one_hot = F.one_hot(y_true.to(torch.int64), 2).to(torch.float32)
        
        tp = (y_true_one_hot * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true_one_hot) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true_one_hot) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true_one_hot * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        f1=f1.detach()
        CE =torch.nn.CrossEntropyLoss(weight=( 1 - f1))(y_pred, y_true_one_hot)
        return  CE.mean()
    

class RecallCELoss(nn.Module):
    def __init__(self, num_classes):
        super(RecallCELoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        recall_scores = []
        nc = torch.zeros(self.num_classes).to(inputs.device)
        pc = torch.zeros(self.num_classes).to(inputs.device)
        
        for c in range(self.num_classes):
            tp = torch.sum((inputs.argmax(dim=1) == c) & (targets == c))
            fn = torch.sum((inputs.argmax(dim=1) != c) & (targets == c))
            
            recall = tp / (tp + fn + 1e-8)
            recall_scores.append(recall)
            nc[c] = tp + fn
            pc[c] = 1
        
        recall_ce_loss = -torch.mean((1 - torch.tensor(recall_scores)) * nc * torch.log(pc))
        return recall_ce_loss
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

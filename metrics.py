import torch

from torchmetrics import Metric

class FMeasure(Metric):
    def __init__(self, num_classes, threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.num_classes = num_classes
        self.threshold = threshold

    def update(self, preds, targets):
        preds = (preds >= self.threshold).to(torch.int32)
        targets = targets.to(torch.int32)

        for c in range(self.num_classes):
            true_positives = torch.sum((preds == 1) & (targets == 1) & (targets == c))
            false_positives = torch.sum((preds == 1) & (targets == 0) & (targets == c))
            false_negatives = torch.sum((preds == 0) & (targets == 1) & (targets == c))

            self.true_positives[c] += true_positives
            self.false_positives[c] += false_positives
            self.false_negatives[c] += false_negatives

    def compute(self):
        epsilon = torch.tensor(1e-7)
        precision = self.true_positives / (self.true_positives + self.false_positives + epsilon)
        recall = self.true_positives / (self.true_positives + self.false_negatives + epsilon)
        f_measure = 2 * precision * recall / (precision + recall + epsilon)
        macro_average = torch.mean(f_measure)

        return macro_average


# Example usage
preds = torch.tensor([[0.9, 0.1, 0.3], [0.2, 0.8, 0.6], [0.7, 0.4, 0.5]])
targets = torch.tensor([0, 1, 2])
num_classes = 3

f_measure = FMeasure(num_classes)
f_measure.update(preds, targets)
macro_average = f_measure.compute()

print("F-measure:", macro_average.item())

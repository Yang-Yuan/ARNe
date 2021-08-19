import torch.nn as nn

class WReN_Loss(nn.Module):
	def __init__(self, beta):
		super(WReN_Loss, self).__init__()

		self.loss_targets = nn.CrossEntropyLoss()
		self.loss_meta_targets = nn.BCEWithLogitsLoss()

		self.beta = beta

	def forward(self, logits_target, labels_target, logits_meta, labels_meta):
		loss_targets = self.loss_targets(logits_target, labels_target)
		loss_meta = self.loss_meta_targets(logits_meta, labels_meta)
		return loss_targets, loss_meta, (loss_targets + self.beta * loss_meta)
import torch
from torch import nn
import src.models as models
from src.utils import DatasetSplit
from torch.utils.data import DataLoader


class Server(object):

	def __init__(self, arg, test_dataset):
		if arg.datasets == 'mnist':
			self.global_model = models.CNNMnist(arg)
		elif arg.datasets == 'cifar10':
			self.global_model = models.CNNCifar10(arg)
		elif arg.datasets == 'cwru':
			self.global_model = models.WDCNN_CWRU(arg.channel_num, arg.class_num)

		if arg.continued:
			self.global_model.load_state_dict(torch.load('model.pkl'))
		self.test_dataset = test_dataset
		self.global_epoch = arg.global_epoch
		self.batch_size = arg.batch_size
		self.user_num = arg.user_num
		self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=arg.batch_size)


	def aggregate(self, client_num, weight_arry):
		for name in weight_arry.keys():
			weight_arry[name] = torch.div(weight_arry[name], client_num)
		self.global_model.load_state_dict(weight_arry)

	def eval_model(self):
		self.global_model.eval()
		criterion = nn.CrossEntropyLoss()
		total_loss = 0.0
		correct = 0
		data_size = 0
		for b_id, batch in enumerate(self.test_loader):
			x, y = batch
			data_size += self.batch_size
			output = self.global_model(x)
			total_loss += criterion(output, y).item()

			_, predicted = torch.max(output.data, 1)
			correct += (predicted == y).sum().item()

		acc = 100 * correct / data_size
		loss = total_loss / data_size
		return acc, loss


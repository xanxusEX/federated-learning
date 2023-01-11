import torch
from torch import nn
import torch.nn.functional as F
import src.models as models
from torch.utils.data import DataLoader
from src.utils import DatasetSplit
import numpy as np


class Client(object):

	def __init__(self, arg, c_id, train_dataset, user_group):
		if arg.datasets == 'mnist':
			self.local_model = models.CNNMnist(arg)
		elif arg.datasets == 'cifar10':
			self.local_model = models.CNNCifar10(arg)
		elif arg.datasets == 'cwru':
			self.local_model = models.WDCNN_CWRU(arg.channel_num, arg.class_num)

		self.c_id = c_id
		self.train_dataset = train_dataset
		self.local_epoch = arg.local_epoch
		self.batch_size = arg.batch_size
		self.global_epoch = arg.global_epoch
		self.user_group = list(user_group[c_id])
		self.cur_dataset = [self.user_group[i] for i in range(int(0.5 * len(self.user_group)))]

		self.datasplit = DatasetSplit(self.train_dataset, set(self.cur_dataset))
		#增量算子参数
		self.theta = 0
		self.beta = 0
		self.last_data_size = len(self.cur_dataset)
		self.selected_times = 0


	def local_train(self, arg, cur_model, cur_epoch, server):
		assert isinstance(cur_model, nn.Module)
		self.local_model = cur_model
		self.local_model.train()
		epoch_loss = []

		optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001)

		criterion = nn.CrossEntropyLoss()

		#数据增量机制
		#计算当前轮数的深度
		depth = int((1 + cur_epoch / self.global_epoch) * 0.5 * len(self.user_group))
		#计算数据量增量
		increase = depth - self.last_data_size
		#获得增量后数据集
		self.cur_dataset = [self.user_group[i] for i in range(depth)]
		self.datasplit = DatasetSplit(self.train_dataset, set(self.cur_dataset))

		#TO TRAIN WITH WHOLE LOCAL-SET
		# self.datasplit = DatasetSplit(self.train_dataset, self.user_group)

		train_loader = DataLoader(self.datasplit, batch_size=self.batch_size, shuffle=True)

		#增量算子参数计算（改进后）
		self.selected_times += 1
		if increase != 0 and self.theta != 0:
			gamma = increase / self.last_data_size
			# print(increase, self.last_data_size, gamma)
			self.last_data_size = len(self.cur_dataset)
			self.theta = self.theta - gamma * self.theta
		else:
			#参数theta根据自身活跃度变化
			self.theta = self.selected_times
			self.last_data_size = len(self.cur_dataset)
		# print(self.theta)
		##参数beta代表模型深度（即聚合权重），模型权重与增量数量成正比
		self.beta = 1 - 2 * np.arctan(self.theta) / np.pi

		#本地训练
		for e in range(self.local_epoch):
			batch_loss = []
			train_accs = []
			for b_id, batch in enumerate(train_loader):
				x, y = batch
				optimizer.zero_grad()
				output = self.local_model(x)
				loss = criterion(output, y)
				loss.backward()
				optimizer.step()

				batch_loss.append(loss.item())

				_, predicted = torch.max(output.data, 1)
				total = y.size(0)  # labels 的长度
				correct = (predicted == y).sum().item()  # 预测正确的数目
				train_accs.append(100 * correct / total)

			epoch_loss.append(sum(batch_loss) / len(batch_loss))
			print("Local Epoch {},   ACC: {}\n".format(e + 1, sum(train_accs) / len(train_accs)))
		print("Client {} Loss : {}\n".format(self.c_id, sum(epoch_loss) / len(epoch_loss)))






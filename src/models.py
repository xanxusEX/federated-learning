from torch import nn
import torch.nn.functional as F
import torch
import pickle


class CNNMnist(nn.Module):

	def __init__(self, arg):
		super(CNNMnist, self).__init__()
		self.cov1 = nn.Conv2d(arg.channel_num, 10, kernel_size=5)
		self.cov2 = nn.Conv2d(10, 20, kernel_size=5)
		self.drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, arg.class_num)

	def forward(self, x):
		x = self.cov1(x)
		x = F.relu(F.max_pool2d(x, 2))
		x = self.drop(self.cov2(x))
		x = F.relu(F.max_pool2d(x, 2))
		x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)


class CNNCifar10(nn.Module):

	def __init__(self, arg):
		super(CNNCifar10, self).__init__()
		self.cov1 = nn.Conv2d(3, 6, 5)
		self.pooling = nn.MaxPool2d(2, 2)
		self.cov2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, arg.class_num)

	def forward(self, x):
		x = self.pooling(F.relu(self.cov1(x)))
		x = self.pooling(F.relu(self.cov2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.log_softmax(x, dim=1)

class WDCNN_CWRU(nn.Module):

	def __init__(self, input_channel=1, output_channel=13):
		super(WDCNN_CWRU, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv1d(input_channel, 16, kernel_size=64, stride=16, padding=24),
			nn.BatchNorm1d(16),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2)
		)

		self.layer2 = nn.Sequential(
			nn.Conv1d(16, 32, kernel_size=3, padding=1),
			nn.BatchNorm1d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2))

		self.layer3 = nn.Sequential(
			nn.Conv1d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2)
		)  # 32, 12,12     (24-2) /2 +1

		self.layer4 = nn.Sequential(
			nn.Conv1d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2)
		)  # 32, 12,12     (24-2) /2 +1

		self.layer5 = nn.Sequential(
			nn.Conv1d(64, 64, kernel_size=3),
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2)
			# nn.AdaptiveMaxPool1d(4)
		)  # 32, 12,12     (24-2) /2 +1

		self.fc = nn.Sequential(
			nn.Linear(192, 100),
			nn.ReLU(inplace=True),
			nn.Linear(100, output_channel)
		)

	def forward(self, x):
		x = x.view(-1, 1, 2048)
		# print(x.shape)
		x = self.layer1(x)  # [16 64]
		# print(x.shape)
		x = self.layer2(x)  # [32 124]
		# print(x.shape)
		x = self.layer3(x)  # [64 61]
		# print(x.shape)
		x = self.layer4(x)  # [64 29]
		# print(x.shape)
		x = self.layer5(x)  # [64 13]
		# print(x.shape)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

if __name__ == "__main__":
	cnn_cwru = WDCNN_CWRU()
	dump_x = pickle.dumps(cnn_cwru.state_dict())
	print(len(dump_x))
	import preprocess
	path = '../data/cwru_experiment'
	train_X, train_Y, test_X, test_Y = preprocess.prepro(d_path=path,
	                                                            length=2048,
	                                                            number=1000,
	                                                            normal=True,
	                                                            rate=[0.7, 0.2, 0.1],
	                                                            enc=True,
	                                                            enc_step=28)
	from torch.utils.data import Dataset, DataLoader, TensorDataset
	import numpy as np

	train_X = train_X[:, :, np.newaxis]
	print(train_X.shape[1:])
	train_X, train_Y = torch.FloatTensor(train_X), torch.LongTensor(train_Y)
	print(train_Y.shape)
	train_set = TensorDataset(train_X, train_Y)
	print(len(train_set))
	# train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True)
	#
	# criterion = nn.CrossEntropyLoss()
	#
	# for b_id, batch in enumerate(train_loader):
	# 	x, y = batch
	# 	predict = cnn_cwru(x)
	# 	print(predict[0], y[0])
	# 	loss = criterion(predict, y)
	# 	print(loss)
	# 	break

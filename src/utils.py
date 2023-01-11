import torchvision as tv
import torch
from src.sampling import cifar_iid, cifar_noniid, mnist_noniid, mnist_iid
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import phe
from src.preprocess import prepro


def get_datasets(arg):
	train_dataset, test_dataset, user_group = [], [], []
	if arg.datasets == "cifar10":
		apply_transform = tv.transforms.Compose(
			[tv.transforms.ToTensor(), tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
		)
		train_dataset = tv.datasets.CIFAR10(root='../data/cifar10', train=True, download=True, transform=apply_transform)
		test_dataset = tv.datasets.CIFAR10(root='../data/cifar10', train=False, transform=apply_transform)
		if arg.iid:
			user_group = cifar_iid(train_dataset, arg.user_num)
		else:
			user_group = cifar_noniid(train_dataset, arg.user_num)

	elif arg.datasets == "mnist":
		train_dataset = tv.datasets.MNIST(root='../data/mnist', train=True, download=True, transform=tv.transforms.ToTensor())
		test_dataset = tv.datasets.MNIST(root="../data/mnist", train=False, transform=tv.transforms.ToTensor())
		if arg.iid:
			user_group = mnist_iid(train_dataset, arg.user_num)
		else:
			user_group = mnist_noniid(train_dataset, arg.user_num)
	elif arg.datasets == 'cwru':
		train_X, train_Y, test_X, test_Y = prepro(d_path='D:\python files\FedAvg\data\cwru_experiment',
		                                                                    length=2048,
		                                                                    number=1000,
		                                                                    normal=True,
		                                                                    rate=[0.7, 0.2, 0.1],
		                                                                    enc=True,
		                                                                    enc_step=28)
		train_X, test_X = train_X[:, :, np.newaxis], test_X[:, :, np.newaxis]
		train_X, train_Y = torch.FloatTensor(train_X), torch.LongTensor(train_Y)
		test_X, test_Y = torch.FloatTensor(test_X), torch.LongTensor(test_Y)
		train_dataset = TensorDataset(train_X, train_Y)
		test_dataset = TensorDataset(test_X, test_Y)
		user_group = cifar_iid(train_dataset, arg.user_num)

	return train_dataset, test_dataset, user_group


class DatasetSplit(Dataset):

	def __init__(self, dataset, index):
		self.dataset = dataset
		self.idx = [int(i) for i in index]

	def __len__(self):
		return len(self.idx)

	def __getitem__(self, item):
		x, y = self.dataset[self.idx[item]]
		return torch.tensor(x), torch.tensor(y)


def add_noise(w, sigma, is_laplace=True):
	assert isinstance(w, dict)
	noise = {}
	for name, param, in w.items():
		if is_laplace:  #拉普拉斯噪声
			noise[name] = torch.tensor(np.random.laplace(0, sigma, param.shape))
		else:           #可选高斯噪声，但需要gpu
			noise[name] = torch.FloatTensor(param.shape).normal_(0, sigma)
		param.add_(torch.Tensor.long(noise[name]))
	return w


def generate_keypair(n_length=1024):
	return phe.paillier.generate_paillier_keypair(n_length=1024)


def encrypt_param(public_key, w):
	assert isinstance(w, dict)
	encrypt_w = {}
	w_shape = {}
	for name, param in w.items():
		w_shape[name] = param.shape
		param_list = param.flatten(0).cpu().numpy().tolist()
		encrypt_w[name] = [public_key.encrypt(parameter) for parameter in param_list]
	return encrypt_w, w_shape


def decrypt_param(private_key, encrypt_w, w_shape):
	assert isinstance(encrypt_w, dict)
	assert isinstance(w_shape, dict)
	w = {}
	for name, param in encrypt_w.items():
		w[name] = [private_key.decrypt(parameter) for parameter in param]
		w[name] = torch.reshape(torch.Tensor(w[name]), w_shape[name])
	return w

def get_SD(param1, param2):
	pass







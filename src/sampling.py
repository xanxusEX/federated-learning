import numpy as np
import torchvision as tv


def mnist_iid(dataset, num_users):
	"""
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
	num_items = int(len(dataset) / num_users)
	dict_users, all_idxs = {}, [i for i in range(len(dataset))]
	for i in range(num_users):
		dict_users[i] = set(np.random.choice(all_idxs, num_items,
		                                     replace=False))
		all_idxs = list(set(all_idxs) - dict_users[i])
	return dict_users


def mnist_noniid(dataset, num_users):
	"""
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
	# 60,000 training imgs -->  200 imgs/shard X 300 shards
	num_shards, num_imgs = 200, 300
	idx_shard = [i for i in range(num_shards)]
	dict_users = {i: np.array([]) for i in range(num_users)}
	idxs = np.arange(num_shards * num_imgs)
	labels = dataset.train_labels.numpy()

	# sort labels
	idxs_labels = np.vstack((idxs, labels))
	idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
	idxs = idxs_labels[0, :]

	# divide and assign 2 shards/client
	for i in range(num_users):
		rand_set = set(np.random.choice(idx_shard, 2, replace=False))
		idx_shard = list(set(idx_shard) - rand_set)
		for rand in rand_set:
			dict_users[i] = np.concatenate(
				(dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
	return dict_users


def cifar_iid(dataset, num_users):
	"""
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
	num_items = int(len(dataset) / num_users)
	dict_users, all_idxs = {}, [i for i in range(len(dataset))]
	for i in range(num_users):
		dict_users[i] = set(np.random.choice(all_idxs, num_items,
		                                     replace=False))
		all_idxs = list(set(all_idxs) - dict_users[i])
	return dict_users


def cifar_noniid(dataset, num_users):
	"""
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
	num_shards, num_imgs = 200, 250
	idx_shard = [i for i in range(num_shards)]
	dict_users = {i: np.array([]) for i in range(num_users)}
	idxs = np.arange(num_shards * num_imgs)
	# labels = dataset.train_labels.numpy()
	labels = np.array(dataset.train_labels)

	# sort labels
	idxs_labels = np.vstack((idxs, labels))
	idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
	idxs = idxs_labels[0, :]

	# divide and assign
	for i in range(num_users):
		rand_set = set(np.random.choice(idx_shard, 2, replace=False))
		idx_shard = list(set(idx_shard) - rand_set)
		for rand in rand_set:
			dict_users[i] = np.concatenate(
				(dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
	return dict_users


if __name__ == "__main__":
	train_dataset = tv.datasets.MNIST(root='../data/mnist', train=True, download=True,
	                                  transform=tv.transforms.ToTensor())
	test_dataset = tv.datasets.MNIST(root="../data/mnist", train=False, transform=tv.transforms.ToTensor())
	user_group_iid = mnist_iid(train_dataset, 10)
	user_group_non = mnist_noniid(train_dataset, 10)
	print(user_group_iid)

import argparse
import random
import torch
import copy
from matplotlib import pyplot as plt
import numpy as np

from src.utils import get_datasets, add_noise
from src.server import Server
from src.client import Client



def arg_parser():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--global_epoch', type=int, default=5)
    my_parser.add_argument('--user_num', type=int, default=20)
    my_parser.add_argument('--frac', type=float, default=0.25)
    my_parser.add_argument('--local_epoch', type=int, default=10)
    my_parser.add_argument('--lr', type=float, default=0.01)
    my_parser.add_argument('--momentum', type=float, default=0.5)
    my_parser.add_argument('--batch_size', type=int, default=10)
    my_parser.add_argument('--sigma', type=float, default=0.1)

    my_parser.add_argument('--continued', type=int, default=0)
    my_parser.add_argument('--datasets', type=str, default='mnist')
    my_parser.add_argument('--channel_num', type=int, default=1)
    my_parser.add_argument('--class_num', type=int, default=10)
    my_parser.add_argument('--client_num', type=int, default=5)
    my_parser.add_argument('--iid', type=int, default=1)  #1->True, 0->False
    return my_parser


def monitorize(x, y1, y2):

    plt.subplot(211)
    plt.plot(x, y1)
    plt.subplot(212)
    plt.plot(x, y2)
    plt.show()


if __name__ == '__main__':
    arg = arg_parser()
    arg = arg.parse_args()
    print(arg)
    train_dataset, test_dataset, user_group = get_datasets(arg)
    server = Server(arg, test_dataset)
    clients = []
    user_num = arg.user_num
    for i in range(user_num):
        clients.append(Client(arg, i, train_dataset, user_group))
    global_epoch = arg.global_epoch
    print(train_dataset, test_dataset)

    total_acc, total_loss, epochs = [], [], []
    for e in range(global_epoch):
        print("Global Epoch {}\n".format(e + 1))

        weight_arry = {}
        weight_shape = {}
        client_betas = []
        delta_cs = []
        for name, param in server.global_model.state_dict().items():
            weight_arry[name] = torch.zeros_like(param)
            weight_arry[name] = torch.Tensor.float(weight_arry[name])


        client_num = int(max(arg.user_num * arg.frac, 1))
        candidates = random.sample(clients, client_num)

        for c in candidates:
            c.local_train(arg, cur_model=copy.deepcopy(server.global_model), cur_epoch=e+1, server=server)
            client_betas.append(1 - c.beta)

        beta_sum = sum(client_betas)
        client_betas = [client_betas[i] * client_num / beta_sum for i in range(len(client_betas))]
        print(client_betas)
        for i in range(len(candidates)):
            noise_dict = add_noise(candidates[i].local_model.state_dict(), arg.sigma, False)
            for name, param in noise_dict.items():
        
                #FEDERATED INCREASING LEARNING
                weight_param = torch.mul(param, client_betas[i])
                weight_arry[name].add_(weight_param)

        server.aggregate(client_num, weight_arry)
        acc, loss = server.eval_model()
        total_acc.append(acc)
        total_loss.append(loss)
        epochs.append(e + 1)
        if acc >= 95:
            break
        print("CURRENT ACC: {}".format(acc))
    print("\n")
    print("||TOTAL RESULT||\n")
    print("ACC: {}, LOSS: {}".format(total_acc[-1], total_loss[-1]))
    print("|| COST ||\n")
    print("GLOBAL EPOCH: {}, LOCAL EPOCH: {}".format(epochs[-1], arg.local_epoch))

    # torch.save(server.global_model.state_dict(), 'model.pkl')
    monitorize(epochs, total_acc, total_loss)




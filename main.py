from __future__ import print_function
import os
import copy
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

from trainer.fixbi_trainer import train_fixbi
from src.dataset import get_dataset
import src.utils as utils

parser = argparse.ArgumentParser(description="FIXBI EXPERIMENTS")
parser.add_argument('-db_path', help='gpu number', type=str, default='/database')
parser.add_argument('-baseline_path', help='baseline path', type=str, default='AD_Baseline')
parser.add_argument('-save_path', help='save path', type=str, default='Logs/save_test')
parser.add_argument('-source', help='source', type=str, default='amazon')
parser.add_argument('-target', help='target', type=str, default='dslr')
parser.add_argument('-workers', default=4, type=int, help='dataloader workers')
parser.add_argument('-gpu', help='gpu number', type=str, default='0,1')
parser.add_argument('-epochs', default=100, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-th', default=2.0, type=float, help='Threshold')
parser.add_argument('-bim_start', default=100, type=int, help='Bidirectional Matching')
parser.add_argument('-sp_start', default=100, type=int, help='Self-Penalization')
parser.add_argument('-cr_start', default=100, type=int, help='Consistency Regularization')
parser.add_argument('-lam_sd', default=0.7, type=float, help='Source Dominant Mixup ratio')
parser.add_argument('-lam_td', default=0.3, type=float, help='Target Dominant Mixup ratio')


def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Use GPU(s): {} for training".format(args.gpu))
    print(args)

    num_classes, resnet_type = utils.get_data_info()
    src_trainset, src_testset = get_dataset(args.source, path=args.db_path)
    tgt_trainset, tgt_testset = get_dataset(args.target, path=args.db_path)

    src_train_loader = torchdata.DataLoader(src_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_train_loader = torchdata.DataLoader(tgt_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_test_loader = torchdata.DataLoader(tgt_testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)

    lr, l2_decay, momentum, nesterov = utils.get_train_info()
    net_sd, head_sd, classifier_sd = utils.get_net_info(num_classes)
    net_td, head_td, classifier_td = utils.get_net_info(num_classes)

    learnable_params_sd = list(net_sd.parameters()) + list(head_sd.parameters()) + list(classifier_sd.parameters())
    learnable_params_td = list(net_td.parameters()) + list(head_td.parameters()) + list(classifier_td.parameters())

    optimizer_sd = optim.SGD(learnable_params_sd, lr=lr, momentum=momentum, weight_decay=l2_decay, nesterov=nesterov)
    optimizer_td = optim.SGD(learnable_params_td, lr=lr, momentum=momentum, weight_decay=l2_decay, nesterov=nesterov)

    sp_param_sd = nn.Parameter(torch.tensor(5.0).cuda(), requires_grad=True)
    sp_param_td = nn.Parameter(torch.tensor(5.0).cuda(), requires_grad=True)

    optimizer_sd.add_param_group({"params": [sp_param_sd], "lr": lr})
    optimizer_td.add_param_group({"params": [sp_param_td], "lr": lr})

    ce = nn.CrossEntropyLoss().cuda()
    mse = nn.MSELoss().cuda()

    net_sd, head_sd, classifier_sd = utils.load_net(args, net_sd, head_sd, classifier_sd)
    net_td, head_td, classifier_td = copy.deepcopy(net_sd), copy.deepcopy(head_sd), copy.deepcopy(classifier_sd)
    
    loaders = [src_train_loader, tgt_train_loader]
    optimizers = [optimizer_sd, optimizer_td]
    models_sd = [net_sd, head_sd, classifier_sd]
    models_td = [net_td, head_td, classifier_td]
    sp_params = [sp_param_sd, sp_param_td]
    losses = [ce, mse]
    
    for epoch in range(args.epochs):
        train_fixbi(args, loaders, optimizers, models_sd, models_td, sp_params, losses, epoch)

        utils.evaluate(nn.Sequential(*models_sd), tgt_test_loader)
        utils.evaluate(nn.Sequential(*models_td), tgt_test_loader)
        utils.final_eval(nn.Sequential(*models_sd), nn.Sequential(*models_td), tgt_test_loader)

        utils.save_net(args, models_sd, 'sdm')
        utils.save_net(args, models_td, 'tdm')


if __name__ == "__main__":
    main()

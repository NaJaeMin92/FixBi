import time
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import network.models as models


def get_data_info():
    resnet_type = 50
    num_classes = 31
    return num_classes, resnet_type


def get_net_info(num_classes):
    net = torch.nn.parallel.DataParallel(models.ResNet50().encoder).cuda()
    classifier = torch.nn.parallel.DataParallel(nn.Linear(256, num_classes)).cuda()
    head = torch.nn.parallel.DataParallel(models.Head()).cuda()

    return net, head, classifier


def get_train_info():
    lr = 0.001
    l2_decay = 5e-4
    momentum = 0.9
    nesterov = False
    return lr, l2_decay, momentum, nesterov


def load_net(args, net, head, classifier):
    print("Load pre-trained baseline model !")
    save_folder = args.baseline_path
    net.module.load_state_dict(torch.load(save_folder + '/net.pt'), strict=False)
    head.module.load_state_dict(torch.load(save_folder + '/head.pt'), strict=False)
    classifier.module.load_state_dict(torch.load(save_folder + '/classifier.pt'), strict=False)
    return net, head, classifier


def save_net(args, models, type):
    save_folder = args.save_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    net, head, classifier = models[0], models[1], models[2]

    torch.save(net.module.state_dict(), save_folder + '/' + 'net_' + str(type) + '.pt')
    torch.save(head.module.state_dict(), save_folder + '/' + 'head_' + str(type) + '.pt')
    torch.save(classifier.module.state_dict(), save_folder + '/' + 'classifier_' + str(type) + '.pt')


def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()


def evaluate(models, loader):
    start = time.time()
    total = 0
    correct = 0
    set_model_mode('eval', [models])
    with torch.no_grad():
        for step, tgt_data in enumerate(loader):
            tgt_imgs, tgt_labels = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)
            tgt_preds = models(tgt_imgs)
            pred = tgt_preds.argmax(dim=1, keepdim=True)
            correct += pred.eq(tgt_labels.long().view_as(pred)).sum().item()
            total += tgt_labels.size(0)

    print('Accuracy: {:.2f}%'.format((correct / total) * 100))
    print("Eval time: {:.2f}".format(time.time() - start))
    set_model_mode('train', [models])


def get_sp_loss(input, target, temp):
    criterion = nn.NLLLoss(reduction='none').cuda()
    loss = torch.mul(criterion(torch.log(1 - F.softmax(input / temp, dim=1)), target.detach()), 1).mean()
    return loss


def get_target_preds(args, x):
    top_prob, top_label = torch.topk(F.softmax(x, dim=1), k=1)
    top_label = top_label.squeeze().t()
    top_prob = top_prob.squeeze().t()
    top_mean, top_std = top_prob.mean(), top_prob.std()
    threshold = top_mean - args.th * top_std
    return top_label, top_prob, threshold


def mixup_criterion_hard(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss().cuda()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_fixmix_loss(net, src_imgs, tgt_imgs, src_labels, tgt_pseudo, ratio):
    mixed_x = ratio * src_imgs + (1 - ratio) * tgt_imgs
    mixed_x = net(mixed_x)
    loss = mixup_criterion_hard(mixed_x, src_labels.detach(), tgt_pseudo.detach(), ratio)
    return loss


def final_eval(models_sd, models_td, tgt_test_loader):
    total = 0
    correct = 0
    set_model_mode('eval', [*models_sd])
    set_model_mode('eval', [*models_td])

    with torch.no_grad():
        for step, tgt_data in enumerate(tgt_test_loader):
            tgt_imgs, tgt_labels = tgt_data
            tgt_imgs, tgt_labels = tgt_imgs.cuda(), tgt_labels.cuda()
            pred_sd = F.softmax(models_sd(tgt_imgs), dim=1)
            pred_td = F.softmax(models_td(tgt_imgs), dim=1)
            softmax_sum = pred_sd + pred_td
            _, final_pred = torch.topk(softmax_sum, 1)
            correct += final_pred.eq(tgt_labels.long().view_as(final_pred)).sum().item()
            total += tgt_labels.size(0)

    print('Final Accuracy: {:.2f}%'.format((correct / total) * 100))
    set_model_mode('train', [*models_sd])
    set_model_mode('train', [*models_td])

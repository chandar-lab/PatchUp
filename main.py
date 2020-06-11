#!/usr/bin/env python
from __future__ import division
import os, sys

from modules.cutmix import CutMix
from modules.cutout import Cutout
from modules.patchup import PatchUpMode
from utility.adversarial_attack import run_test_adversarial
from torchvision import transforms
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch.backends.cudnn as cudnn
import matplotlib as mpl

mpl.use('Agg')

from utility.utils import *
import models

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
from collections import OrderedDict
from data_loader import *
from utility.utils import copy_script_to_folder
import shutil
from utility.plots import *
import numpy as np
import argparse

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or SVHN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'svhn', 'mnist'],
                    help='Choose between Cifar10/100 and SVHN.')
parser.add_argument('--data_dir', type=str, default='cifar10',
                    help='file where results are to be written')
parser.add_argument('--root_dir', type=str, default='experiments',
                    help='folder where results are to be stored')
parser.add_argument('--labels_per_class', type=int, default=5000, metavar='NL',
                    help='labels_per_class')
parser.add_argument('--valid_labels_per_class', type=int, default=0, metavar='NL',
                    help='validation labels_per_class')
parser.add_argument('--arch', metavar='ARCH', default='preactresnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: preactresnet18)')
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--train', type=str, default='vanilla',
                    choices=['vanilla', 'mixup', 'manifold_mixup', 'cutout', 'patchup', 'cutmix', 'dropblock'])
parser.add_argument('--alpha', type=float, default=0.0, help='alpha parameter for mixup')
parser.add_argument('--cutout', type=int, default=8, help='size of cut out')
parser.add_argument('--drop_block', type=int, default=7, help='block size of dropblock')
parser.add_argument('--drop_block_all', type=str, default='False', help='apply dropblock on all layers')
parser.add_argument('--k', type=int, default=2, help='k hidden blocks')
parser.add_argument('--keep_prob', type=float, default=.9, help='feature keep probability in the dropblock')
parser.add_argument('--gamma', type=float, default=.9, help='feature probability to be altered in the PatchUp')
parser.add_argument('--patchup_block', type=int, default=7, help='block size of PatchUp')
parser.add_argument('--patchup_prob', type=float, default=.7, help='PatchUp probability')
parser.add_argument('--patchup_type', type=str, default='soft', choices=['soft', 'hard'], help='PatchUp Mode')
parser.add_argument('--cutmix_prob', type=float, default=.5, help='cutmix probability')
parser.add_argument('--dropout', action='store_true', default=False,
                    help='whether to use dropout or not in final layer')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--data_aug', type=int, default=1)
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--step_factors', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by step factor on schedule, number of step factors should be equal to schedule')
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--add_name', type=str, default='')
parser.add_argument('--job_id', type=str, default='')
parser.add_argument('--checkpoint', type=str, default='False',
                    help='checkpoint models and save the best model at training time')
parser.add_argument('--affine_test', type=str, default='False', help='test affine deformations')
parser.add_argument('--affine_path', type=str, default='../../data/affine/', help='test affine data directory')
parser.add_argument('--fsgm_attack', type=str, default='False', help='test FGSM adversarial attack')
parser.add_argument('--loss_params', type=int, nargs='+', default=[1, 1], help='loss controller for patchup')

args = parser.parse_args()

args.patchup_type = PatchUpMode.SOFT if args.patchup_type == 'soft' else PatchUpMode.HARD
args.k = int(args.k)
args.drop_block_all = False if args.drop_block_all == 'False' else True
args.affine_test = False if args.affine_test == 'False' else True
args.fsgm_attack = False if args.fsgm_attack == 'False' else True
loss_alpha, loss_beta = args.loss_params[0], args.loss_params[1]


args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

args.checkpoint = False if args.checkpoint == 'False' else True

out_str = str(args)
print(out_str)


def experiment_name_non_mnist(dataset='cifar10',
                              arch='',
                              epochs=400,
                              dropout=True,
                              batch_size=64,
                              lr=0.01,
                              momentum=0.5,
                              decay=0.0005,
                              data_aug=1,
                              train='vanilla',
                              alpha=0.0,
                              job_id=None,
                              add_name='',
                              patchup_mode=PatchUpMode.SOFT, keep_prob=args.keep_prob, gamma=args.gamma,
                              k=args.k, dropblock_all=args.drop_block_all):
    """
    Args:
        Following are the information that we need to create a name for the experiment.
        dataset, arch, epochs, dropout, batch_size, lr, momentum, decay, data_aug, train,
        alpha, job_id, add_name, patchup_mode, keep_prob, k, dropblock_all.

    Returns:
        The naame of the experiment.

    """
    exp_name = dataset
    exp_name += '_arch_' + str(arch)
    exp_name += '_train_' + str(train)
    exp_name += '_m_alpha_' + str(alpha)
    if dropout:
        exp_name += '_do_' + 'true'
    else:
        exp_name += '_do_' + 'False'
    exp_name += '_eph_' + str(epochs)
    exp_name += '_bs_' + str(batch_size)
    exp_name += '_lr_' + str(lr)
    exp_name += '_mom_' + str(momentum)
    exp_name += '_decay_' + str(decay)
    exp_name += '_data_aug_' + str(data_aug)

    if add_name != '':
        exp_name += '_add_name_' + str(add_name)
    if train == 'patchup' or train == 'dropblock' or train == 'manifold_mixup':
        exp_name += '_k_' + str(k)
    if train == 'dropblock':
        exp_name += 'dropbloc_all_' + str(dropblock_all) + '_keep_prob_' + str(keep_prob)
    if train == 'patchup':
        exp_name += '_patchup_block_' + str(args.patchup_block) + '_patchup_prob_' + str(args.patchup_prob) + \
                    '_patchup_mode_' + str(patchup_mode.value) + '_gamma_' + str(gamma)

    if job_id != None:
        exp_name += '_job_id_' + str(job_id)
    print('experiement name: ' + exp_name)
    return exp_name


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, step_factors, schedule):
    lr = args.learning_rate
    assert len(step_factors) == len(schedule), "length of step_factors and schedule should be equal"
    for (gamma, step) in zip(step_factors, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


bce_loss = nn.BCELoss().to(device)
softmax = nn.Softmax(dim=1).to(device)
criterion = nn.CrossEntropyLoss().to(device)
mse_loss = nn.MSELoss().to(device)


def train(train_loader, model, optimizer, epoch, args, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    lam = None
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.long()
        input, target = input.to(device), target.to(device)
        data_time.update(time.time() - end)
        ###  clean training####
        if args.train == 'mixup':
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var, mixup=True, alpha=args.alpha)
            loss = bce_loss(softmax(output), reweighted_target)
        elif args.train == 'patchup':
            input_var, target_var = Variable(input), Variable(target)
            patchup_type = PatchUpMode.HARD if args.patchup_type == PatchUpMode.HARD else PatchUpMode.SOFT

            r = np.random.rand(1)
            if r < args.patchup_prob:
                target_a, target_b, target_reweighted, output, portion = model(input_var, target_var,
                                                                               lam=lam,
                                                                               patchup=True,
                                                                               epoch=epoch,
                                                                               patchup_type=patchup_type, k=args.k)

                target_a_var = torch.autograd.Variable(target_a)
                target_b_var = torch.autograd.Variable(target_b)

                loss = loss_alpha * bce_loss(softmax(output), target_a_var) * \
                       portion + bce_loss(softmax(output), target_b_var) * (1. - portion) + \
                       loss_beta * bce_loss(softmax(output), target_reweighted)
            else:
                input_var, target_var = Variable(input), Variable(target)
                output, reweighted_target = model(input_var, target_var)
                loss = bce_loss(softmax(output), reweighted_target)

        elif args.train == 'dropblock':
            # block_size = args.drop_block
            input_var, target_var = Variable(input), Variable(target)

            output, reweighted_target = model(input_var, target_var, dropblock=True, dropblock_all=args.drop_block_all,
                                              k=args.k)
            loss = bce_loss(softmax(output), reweighted_target)

        elif args.train == 'manifold_mixup':
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var, manifold_mixup=True,
                                              alpha=args.alpha,
                                              lam=lam, k=args.k)
            loss = bce_loss(softmax(output), reweighted_target)

        elif args.train == 'vanilla':
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var)
            loss = bce_loss(softmax(output), reweighted_target)

        elif args.train == 'cutout':
            input_transforms = transforms.Compose([Cutout(n_holes=1, length=args.cutout)])
            cut_input = input_transforms(input)
            input_var = torch.autograd.Variable(cut_input, requires_grad=True)

            target_var = torch.autograd.Variable(target)

            output, reweighted_target = model(input_var, target_var)
            loss = bce_loss(softmax(output), reweighted_target)

        elif args.train == 'cutmix':
            r = np.random.rand(1)
            if r < args.cutmix_prob:
                cutmix = CutMix(beta=1.)
                target_a, target_b, input, lam = cutmix.apply(inputs=input, target=target)
                target_a = to_one_hot(target_a, args.num_classes)
                target_b = to_one_hot(target_b, args.num_classes)
                input_var = torch.autograd.Variable(input, requires_grad=True)
                target_a_var = torch.autograd.Variable(target_a)
                target_b_var = torch.autograd.Variable(target_b)
                output = model(input_var)
                loss = bce_loss(softmax(output), target_a_var) * lam + bce_loss(softmax(output), target_b_var) * (
                        1. - lam)
            else:
                # compute output
                input_var, target_var = Variable(input), Variable(target)
                output, reweighted_target = model(input_var, target_var)
                loss = bce_loss(softmax(output), reweighted_target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg),
        log)
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.to(device, non_blocking=True)
            input = input.to(device)
        with torch.no_grad():
            input_var = Variable(input)
            target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print_log(
        '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f} '.format(
            top1=top1, top5=top5, error1=100 - top1.avg, losses=losses), log)
    error1 = 100 - top1.avg
    return top1.avg, top5.avg, error1, losses.avg


best_acc = 0


def main():
    ### set up the experiment directories########
    exp_name = experiment_name_non_mnist(dataset=args.dataset,
                                         arch=args.arch,
                                         epochs=args.epochs,
                                         dropout=args.dropout,
                                         batch_size=args.batch_size,
                                         lr=args.learning_rate,
                                         momentum=args.momentum,
                                         decay=args.decay,
                                         data_aug=args.data_aug,
                                         train=args.train,
                                         alpha=args.alpha,
                                         job_id=args.job_id,
                                         add_name=args.add_name,
                                         patchup_mode=args.patchup_type, keep_prob=args.keep_prob, gamma=args.gamma,
                                         k=args.k, dropblock_all=args.drop_block_all)

    exp_dir = os.path.join(args.root_dir, exp_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    copy_script_to_folder(os.path.abspath(__file__), exp_dir)

    result_png_path = os.path.join(exp_dir, 'results.png')

    global best_acc
    global best_model

    log = open(os.path.join(exp_dir, 'log.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(exp_dir), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_workers = 0

    if device == torch.device("cuda"):
        num_workers = 2

    per_img_std = False
    train_loader, valid_loader, _, test_loader, num_classes = load_data_subset(args.data_aug, args.batch_size,
                                                                               num_workers, args.dataset,
                                                                               args.data_dir,
                                                                               labels_per_class=args.labels_per_class,
                                                                               valid_labels_per_class=args.valid_labels_per_class)

    if args.dataset == 'tiny-imagenet-200':
        stride = 2
    else:
        stride = 1

    drop_block = args.drop_block
    keep_prob = args.keep_prob
    gamma = args.gamma
    patchup_block = args.patchup_block
    patchup_prob = args.patchup_prob

    print_log("=> creating model '{}'".format(args.arch), log)
    net = models.__dict__[args.arch](num_classes, args.dropout, per_img_std, stride, drop_block, keep_prob, gamma,
                                     patchup_block).to(device)

    print_log("=> network :\n {}".format(net), log)
    args.num_classes = num_classes

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = recorder.max_accuracy(False)
            print_log(
                "=> loaded checkpoint '{}' accuracy={} (epoch {})".format(args.resume, best_acc, checkpoint['epoch']),
                log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        validate(test_loader, net, criterion, log)
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    # Main loop
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.step_factors, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate) \
            + ' [Best Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        tr_acc, tr_acc5, tr_los = train(train_loader, net, optimizer, epoch, args, log)

        # evaluate on validation set
        val_acc, top5_avg, error1, val_los = validate(valid_loader, net, log)

        train_loss.append(tr_los)
        train_acc.append(tr_acc)
        test_loss.append(val_los)
        test_acc.append(val_acc)

        dummy = recorder.update(epoch, tr_los, tr_acc, val_los, val_acc)

        # save the updated model if its performance is better that the best model that
        # we have had up now based on the validation performance comparison.
        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc, best_top5_avg, best_error1, best_val_los = val_acc, top5_avg, error1, val_los
            #TODO
            best_model = models.__dict__[args.arch](num_classes, args.dropout, per_img_std, stride).to(device)
            best_model.load_state_dict(net.state_dict())
            best_model.eval()

        if args.checkpoint:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }, is_best, exp_dir, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(result_png_path)
        train_log = OrderedDict()
        train_log['train_loss'] = train_loss
        train_log['train_acc'] = train_acc
        train_log['test_loss'] = test_loss
        train_log['test_acc'] = test_acc

        pickle.dump(train_log, open(os.path.join(exp_dir, 'log.pkl'), 'wb'))
        plotting(exp_dir)

    print_log('best model stat on validation set:', log)
    validate(valid_loader, best_model, log)

    print_log('best model stat on test set:', log)
    validate(test_loader, best_model, log)

    # following lines used for deformation test performance comparison after training the regularized model.
    if args.affine_test:
        affine_data_loaders = load_transformed_test_sets(args.affine_path, batch_size=100, workers=2)
        for t_loader in affine_data_loaders:
            print_log(f'model performance on {t_loader.transformer} test set:', log)
            validate(t_loader, best_model, log)

    # following lines used for evaluate regularized model performance and robustness to Adversarial Examples
    # created through FGSM attacks.
    if args.fsgm_attack:
        # epsilons controls the perturbation in FGSM attack.
        epsilons = [0, .05, .1, .12, .15, .18, .2]
        accuracies = []
        for eps in epsilons:
            result = run_test_adversarial(best_model, test_loader, eps)
            accuracies.append(result)
            print_log(result, log)
        print_log('the FSGM result :', log)
        print_log(accuracies, log)

    log.close()


if __name__ == '__main__':
    main()

#PR_2

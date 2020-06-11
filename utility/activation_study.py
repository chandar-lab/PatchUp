from __future__ import division
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import matplotlib as mpl

mpl.use('Agg')

from utility.utils import *
import models
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
from data_loader import *
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
parser.add_argument('--drop_block', type=int, default=7, help='block size of dropblock')
parser.add_argument('--arch', metavar='ARCH', default='preactresnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: preactresnet18)')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--job_id', type=str, default='')
parser.add_argument('--checkpoint', type=str, default='False', help='get checkpoint and save best model')
parser.add_argument('--methods', type=str, nargs='+', default=['cutout', 'cutmix', 'manifold', 'soft', 'hard'],
                    help='regularization methods')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--dropout', action='store_true', default=False,
                        help='whether to use dropout or not in final layer')

args = parser.parse_args()
args.checkpoint = False if args.checkpoint == 'False' else True

out_str = str(args)
print(out_str)


def experiment_name(dataset='cifar10',
                              arch=''):
    exp_name = dataset
    exp_name += '_arch_' + str(arch)
    print('experiement name: ' + exp_name)
    return exp_name


def get_activation_sum(activation_lst):
    sum_acts = []
    for activation in activation_lst:
        summation = [activation.features[0, i].sum().item() for i in range(activation.features.shape[1])]
        sum_acts.append(summation)
    return sum_acts


def get_activation_magnitude(val_loader, model):
    model.to(device)
    # First we should register each block in the forward hook while we are doing inference.
    activations_1 = SaveFeatures(list(model.children())[1])
    activations_3 = SaveFeatures(list(model.children())[3])
    activations_5 = SaveFeatures(list(model.children())[5])
    activations_7 = SaveFeatures(list(model.children())[7])

    activation_lst = [activations_1, activations_3, activations_5, activations_7]
    # initialize the sum of activation to zero
    sum_act = np.array([np.zeros(shape=(16,)), np.zeros(shape=(160,)), np.zeros(shape=(320,)), np.zeros(shape=(640,))])

    model.eval()

    # iterate over mini-batches of the validation set.
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        with torch.no_grad():
            input_var = Variable(input)

        model(input_var)
        # after inference we have access to the activations of blocks since we registered them in the forward hook.
        result = get_activation_sum(activation_lst)
        sum_act = np.add(sum_act, result)
    #  average them for all samples
    sum_act = sum_act / float(len(val_loader.dataset.targets))
    return sum_act


best_acc = 0


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.clone().detach().requires_grad_(True).to(device)
    def close(self):
        self.hook.remove()


def main():
    """
    This method is responsible to load a pre-trained regularized model for the ablation study of analysis of
    regularization techniques Effect on activations (described in section 4.4 in manuscript of the paper and Appendix-E
    It will save the result as a numpy array in a given path.
    """
    exp_name = experiment_name(dataset=args.dataset, arch=args.arch)
    exp_dir = os.path.join(args.root_dir, exp_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    copy_script_to_folder(os.path.abspath(__file__), exp_dir)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    num_workers = 0

    if device == torch.device("cuda"):
        num_workers = 2

    per_img_std = False
    # load the datasets
    train_loader, valid_loader, _, test_loader, num_classes = load_data_subset(1,
                                                                               args.batch_size,
                                                                               num_workers, args.dataset,
                                                                               args.data_dir,
                                                                               labels_per_class=args.labels_per_class,
                                                                               valid_labels_per_class=args.valid_labels_per_class)

    stride = 1
    drop_block = args.drop_block
    keep_prob = .7
    gamma = .7
    patchup_block = 7
    for method in args.methods:
        # following line are responsible to load the best pre-trained model for each regularize mode.
        log_method = os.path.join(exp_dir, f'log_{method}.npy')
        net = models.__dict__[args.arch](num_classes, args.dropout, per_img_std, stride, drop_block, keep_prob, gamma,
                                         patchup_block).to(device)

        resume = os.path.join(args.root_dir, f'{method}/model_best.pth.tar')
        if resume:
            if os.path.isfile(resume):
                checkpoint = torch.load(resume, map_location=torch.device(device))
                recorder = checkpoint['recorder']
                args.start_epoch = checkpoint['epoch']
                net.load_state_dict(checkpoint['state_dict'])
                best_acc = recorder.max_accuracy(False)
        # calculate the magnitudes of activations after each block in the pre-trained model.
        magnitudes = get_activation_magnitude(test_loader, net)
        # we save the average activations for validation samples as an numpy array.
        np.save(log_method, magnitudes)


if __name__ == '__main__':
    main()

#PR

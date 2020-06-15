import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import torch
from torch.autograd import Variable, grad
from utility.utils import to_var, device


def fgsm(classifier, x, loss_func, epsilon, initial_perturb=None):
    """
    This function is responsible to creat adversarial samples using fGSM attack approach for a given mini-batch.
    FSGM creates adversarial examples using x′=x +epsilon × sign (∇xJ(θ,x,y)).
    Args:
        classifier: A cnn model
        x: mini-batch samples
        loss_func: model loss
        epsilon: a float number in [0, 1]. epsilon is the magnitude that controls the perturbation.
        initial_perturb: a float number in [0, 1] if you need a simple perturbation before applying FGSM perturbation.
        it permute input x as follow: x = x + initial_perturb

    Returns:
        permutted mini-batch samples using FGSM attack approach.
    """
    x_inp = Variable(x.data, requires_grad=True)

    minv = x_inp.min().cpu().detach().numpy().tolist()
    maxv = x_inp.max().cpu().detach().numpy().tolist()

    if initial_perturb is not None and x_inp.size(0) == initial_perturb.size(0):
        xstart = x_inp + initial_perturb
    else:
        xstart = x_inp * 1.0
        if initial_perturb is not None:
            pass
            # print('perturb not same size!', x_inp.size(0), initial_perturb.size(0))
        else:
            pass
            # print('perturb is none')

    c_pre = classifier(xstart)
    loss = loss_func(c_pre)
    nx_adv = x_inp + epsilon * torch.sign(grad(loss, xstart)[0])

    nx_adv = torch.clamp(nx_adv, minv, maxv)
    perturb = nx_adv - x_inp
    perturb = torch.clamp(perturb, -1.0 * epsilon, epsilon)
    nx_adv = x_inp + perturb
    x_adv = to_var(nx_adv.data)

    return x_adv


def run_test_adversarial(net, loader, epsilon):
    """
    This function is responsible to evaluate the model robustness against an FGSM attack on the given data loader.
    Args:
        net: a CNN model.
        loader: a pytorch data loader that load either validation or test set samples
        epsilon: a float number in [0, 1]. epsilon is the magnitude that controls the perturbation.

    Returns:
        t_accuracy: the model performance against the FGSM attack.
        t_loss: the model average loss facing the FGSM attack.
    """
    correct = 0
    total = 0
    t_loss = 0

    loss = 0.0
    softmax = torch.nn.Softmax().cuda()
    bce_loss = torch.nn.BCELoss().cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        # create adversarial samples for the mini-batch samples
        adv_data = fgsm(net, data, lambda pred: criterion(pred, target), epsilon)

        output = net(adv_data)
        loss = criterion(output, target)

        # sum up batch loss
        t_loss += loss.item() * target.size(0)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().numpy().sum()
        total += target.size(0)

    t_accuracy = 100. * correct * 1.0 / total
    t_loss = t_loss / total
    return t_accuracy, t_loss

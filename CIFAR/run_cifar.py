import torch
import random
import torch.nn.functional as F
import argparse
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from cifar_model import WideResNet, WideResNet_madras
from losses import css, my_CrossEntropyLoss, madras_loss
from torch.autograd import Variable
from utils import AverageMeter, accuracy, metrics_print

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, args):
    """Train for one epoch on the training set with deferral"""
    n_classes = args.n_classes
    alpha = args.alpha


    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.to(device)
        input = input.to(device)

        # compute output
        if args.method == 'MixOfExp':
            output, rej = model(input)
            output.detach()
        else:
            output = model(input)

        if args.method == 'LCE':
            # get expert  predictions and costs
            batch_size = output.size()[0]  # batch_size
            m = expert_fn(target)
            m2 = [0] * batch_size
            for j in range(0, batch_size):
                if m[j] == target[j].item():
                    m[j] = 1
                    m2[j] = alpha
                else:
                    m[j] = 0
                    m2[j] = 1
            m = torch.tensor(m)
            m = m.to(device)
            m2 = torch.tensor(m2)
            m2 = m2.to(device)
            # done getting expert predictions and costs
            # compute loss
            loss = css(output, target, n_classes, m, m2)
        elif args.method == 'LearnedOracle':
            if expert_fn == 0:
                loss = my_CrossEntropyLoss(output, target)
            else:
                batch_size = output.size(0)
                m = [0] * batch_size
                for j in range(0, batch_size):
                    if target[j].item() <= expert_fn:
                        m[j] = 1
                    else:
                        m[j] = 0
                m = torch.tensor(m)
                m = m.to(device)
                loss = my_CrossEntropyLoss(output, m)
        elif args.method == 'Confidence':
            if expert_fn == 0:
                loss = my_CrossEntropyLoss(output, target)
            else:
                batch_size = output.size(0)
                m = expert_fn(target)
                for j in range(0, batch_size):
                    m[j] = 1 - (m[j] == target[j].item())
                m = torch.tensor(m)
                m = m.to(device)
                loss = my_CrossEntropyLoss(output, m)
        elif args.method == 'MixOfExp':
            batch_size = output.size(0)
            exp_pred = expert_fn(target)
            exp_pred_n = [[0.005555]*10] * batch_size
            for j in range(0, batch_size):
                if exp_pred[j] == target[j].item():
                    exp_pred_n[j][target[j].item()] = 1 - 10e-12
                else:
                    exp_pred_n[j] = [0.1] * 10
            exp_pred_n = torch.tensor(exp_pred_n)
            exp_pred_n = exp_pred_n.to(device)
            loss = madras_loss(output, rej, target, exp_pred_n)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

def run_reject(model, expert_fn, args):
    '''
    model: WideResNet model
    data_aug: boolean to use data augmentation in training
    n_dataset: number of classes
    expert_fn: expert model
    epochs: number of epochs to train
    '''
    # Data loading code
    n_dataset = args.n_classes
    epochs = args.epochs
    alpha = args.alpha

    if n_dataset == 10:
        dataset = 'cifar10'
        data_aug = False
    elif n_dataset == 100:
        dataset = 'cifar100'
        data_aug = True
    else:
        print('error : args.n_classes parameter is not 10 or 100')


    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if data_aug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])


    kwargs = {'num_workers': 3, 'pin_memory': True}

    train_dataset_all = datasets.__dict__[dataset.upper()]('./data', train=True, download=True,
                                                           transform=transform_train)
    train_size = int(0.90 * len(train_dataset_all))
    test_size = len(train_dataset_all) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset_all, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=128, shuffle=True, **kwargs)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)

    # optionally resume from a checkpoint

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9, nesterov=True,
                                weight_decay=5e-4)

    # cosine learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs)

    for epoch in range(0, epochs):
        # train for one epoch
        if args.method == 'LCE':
            train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, args)
            if epoch % 10 == 0:
                metrics_print(model, expert_fn, test_loader, args)
        else:
            train_reject(train_loader, model, optimizer, scheduler, epoch, expert_fn, args)


def evaluate(model, expert_fn, args, model2=None):
    n_dataset = args.n_classes
    if n_dataset == 10:
        dataset = 'cifar10'
    elif n_dataset == 100:
        dataset = 'cifar100'
    else:
        print('error : args.n_classes parameter is not 10 or 100')

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    kwargs = {'num_workers': 1, 'pin_memory': True}

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[dataset.upper()]('../data', train=False, transform=transform_test, download=True),
        batch_size=128, shuffle=True, **kwargs)

    if args.method == 'LCE' or args.method == 'MixOfExp':
        metrics_print(model, expert_fn, val_loader, args)
    else:
        metrics_print(model, expert_fn, val_loader, args, model2)


class synth_expert:
    '''
    simple class to describe our synthetic expert on CIFAR-10
    ----
    k: number of classes expert can predict
    n_classes: number of classes (10 for CIFAR-10)
    '''
    def __init__(self, k, n_classes):
        self.k = k
        self.n_classes = n_classes

    def predict(self, labels):
        batch_size = labels.size()[0]  # batch_size
        outs = [0] * batch_size
        for i in range(0, batch_size):
            if labels[i].item() <= self.k:
                outs[i] = labels[i].item()
            else:
                prediction_rand = random.randint(0, self.n_classes - 1)
                outs[i] = prediction_rand
        return outs

def main(args):
    expert = synth_expert(args.k, args.n_classes)
    if args.method == 'LCE':
        model = WideResNet(28, args.n_classes + 1, 4, dropRate=0)
        run_reject(model, expert.predict, args)
        evaluate(model, expert.predict, args)
    elif args.method == 'LearnedOracle':
        model_classifier = WideResNet(28, args.n_classes, 4, dropRate=0)
        run_reject(model_classifier, 0, args)
        model_rejector = WideResNet(10, 2, 4, dropRate=0)
        run_reject(model_rejector, args.k, args)
        evaluate(model_classifier, expert.predict, args, model_rejector)
    elif args.method == 'Confidence':
        model_classifier = WideResNet(28, args.n_classes, 4, dropRate=0)
        run_reject(model_classifier, 0, args)
        model_expert = WideResNet(10, 2, 4, dropRate=0)
        run_reject(model_expert, expert.predict, args)
        evaluate(model_classifier, expert.predict, args, model_expert)
    elif args.method == 'MixOfExp':
        model = WideResNet_madras(10, args.n_classes, 4, dropRate=0)
        run_reject(model, expert.predict, args)
        evaluate(model, expert.predict, args)
    else:
        print('error : args.method param is not in [LCE, LearnedOracle, Confidence, MixOfExp]')

    return 0


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--k', type=int, default=5, help='number of classes expert can predict')
    parse.add_argument('--n_classes', type=int, default=10, help='cifar10')
    parse.add_argument('--alpha', type=float, default=1)
    parse.add_argument('--epochs', type=int, default=200)
    # method: LCE, LearnedOracle, Confidence, MixOfExp
    parse.add_argument('--method', type=str, default='MixOfExp')
    args = parse.parse_args()
    args.device = device
    print(args)

    main(args)


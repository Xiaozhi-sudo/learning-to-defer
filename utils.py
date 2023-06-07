import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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


def metrics_print(net, expert_fn, loader, args, net2=None):
    '''
    Computes metrics for deferal
    -----
    Arguments:
    net: model
    expert_fn: expert model predict function
    n_classes: number of classes
    loader: data loader
    '''
    n_classes = args.n_classes
    device = args.device
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            if args.method == 'LCE':
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
            elif args.method == 'LearnedOracle':
                outputs = net(images)
                outputs_rej = net2(images)
                _, predicted = torch.max(outputs.data, 1)
                _, predicted_rej = torch.max(outputs_rej.data, 1)
            elif args.method == 'Confidence':
                outputs = net(images)
                outputs_exp = net2(images)
                _, predicted = torch.max(outputs.data, 1)
                _, predicted_exp = torch.max(outputs_exp.data, 1)
            else:
                outputs, rej = net(images)
                _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size
            exp_prediction = expert_fn(labels)
            for i in range(0, batch_size):
                if args.method == 'LCE':
                    r = (predicted[i].item() == n_classes)
                elif args.method == 'LearnedOracle':
                    r = (predicted_rej[i].item() == 1)
                elif args.method == 'Confidence':
                    r_score = 1 - outputs.data[i][predicted[i].item()].item()
                    r_score = r_score - outputs_exp.data[i][1].item()
                    r = 0
                    if r_score >= 0:
                        r = 1
                    else:
                        r = 0
                else:
                    r = (rej[i][0].item() >= 0.5)
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001)}
    print(to_print)

def metrics_print(net, expert_fn, loader, args, net2=None):
    '''
    Computes metrics for deferal
    -----
    Arguments:
    net: model
    expert_fn: expert model predict function
    n_classes: number of classes
    loader: data loader
    '''
    n_classes = args.n_classes
    device = args.device
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size
            exp_prediction = expert_fn(labels)
            for i in range(0, batch_size):
                r = (predicted[i].item() == n_classes)
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1
    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001)}
    print(to_print)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
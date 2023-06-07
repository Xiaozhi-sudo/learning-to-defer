import torch
def css(outputs, labels, n_classes, m, m2):
    batch_size = outputs.size(0)
    defer = [n_classes] * batch_size
    outputs = -m2 * torch.log2(outputs[range(batch_size),labels])\
              -m * torch.log2(outputs[range(batch_size), defer])
    return torch.sum(outputs) / batch_size


def my_CrossEntropyLoss(outputs, labels):
    batch_size = outputs.size()[0]  # batch_size
    # pick the values corresponding to the labels
    outputs = - torch.log2(outputs[range(batch_size), labels])
    return torch.sum(outputs) / batch_size


def madras_loss(outputs, rej, labels, expert, eps = 10e-12):
    # MixOfExperts loss of Madras et al. 2018
    batch_size = outputs.size()[0]
    output_no_grad = outputs.detach()
    net_loss_no_grad = -torch.log2(output_no_grad[range(batch_size), labels]+eps)
    net_loss = -torch.log2(outputs[range(batch_size), labels]+eps)
    exp_loss = -torch.log2(expert[range(batch_size), labels]+eps)
    system_loss =  (rej[range(batch_size),0])  *  net_loss_no_grad + rej[range(batch_size),1]  * exp_loss
    system_loss += net_loss
    return torch.sum(system_loss)/batch_size


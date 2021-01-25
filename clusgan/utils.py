from __future__ import print_function

try:
    import os
    import numpy as np
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    from itertools import chain as ichain

except ImportError as e:
    print(e)
    raise ImportError

global cuda
cuda = True if torch.cuda.is_available() else False


# Nan-avoiding logarithm
def tlog(x):
    return torch.log(x + 1e-8)


# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)


# Cross Entropy loss with two vector inputs
def cross_entropy(pred, soft_targets):
    log_softmax_pred = torch.nn.functional.log_softmax(pred, dim=1)
    return torch.mean(torch.sum(- soft_targets * log_softmax_pred, 1))


# Save a provided model to file
def save_model(models=[], out_dir=''):
    # Ensure at least one model to save
    assert len(models) > 0, "Must have at least one model to save."

    # Save models to directory out_dir
    for model in models:
        filename = model.name + '.pth.tar'
        outfile = os.path.join(out_dir, filename)
        torch.save(model.state_dict(), outfile)


# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Sample a random latent space vector
# 随机生成噪声 隐空间上的向量
def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False):  # shape就是img[0], 也就是batch_size

    assert (fix_class == -1 or (fix_class >= 0 and fix_class < n_c)), "Requested class %i outside bounds." % fix_class

    # 将Tensor放到GPU上
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Sample noise as generator input, zn
    # 生成噪声zn
    """
    Tensor是基本数据类型，Variable是对Tensor的封装。
    Variable引入了计算图实现自动求导，backward(). Pytorch默认只进行一次自动求导，然后将计算图丢弃，如果想要多次求导需要retain_graph来保留计算图。
    """
    zn = Variable(Tensor(0.75 * np.random.normal(0, 1, (shape, latent_dim))), requires_grad=req_grad)

    ######### zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation
    # 生成噪声zc 和 zc_idx
    zc_FT = Tensor(shape, n_c).fill_(0)  # FloatTensor类型的张量，是empty的特例。初始化形状，用0填充。torch.zeros
    zc_idx = torch.empty(shape, dtype=torch.long)  # empty创建任意类型的张量，使用参数才可以确定形状和数据类型。
    """
    scatter(dim, index, src)
    将src中的数据根据index中的索引按照dim的方向填入self.
    
    index必须是LongTensor类型，所以初始化zc_idx的时候创建的是torch.long类型的Tensor。
    
    Ex:
    class_num = 10
    batch_size = 4
    label = torch.LongTensor(batch_size, 1).random_() % class_num
    #tensor([[6],
    #        [0],
    #        [3],
    #        [2]])
    torch.zeros(batch_size, class_num).scatter_(1, label, 1)
    #tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    #        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
    """

    """
    unsqueeze(dim)
    扩展维度
    
    [0, 1, 2, 3]
    在dim上扩展，如dim=0在0维上加1。[[0, 1, 2, 3]]
    dim=1在1维上加1[[0],
                    [1],
                    [2],
                    [3]]
    dim=-1是在最后一个维度上加1，也就是dim=1
    dim=-2是在倒数第二个维度上加1，也就是dim=0
    """

    """
    _xxx
    in place操作
    
    inplace操作是指将新值赋到原变量地址上的操作。
    """
    if (fix_class == -1):
        zc_idx = zc_idx.random_(n_c).cuda() if cuda else zc_idx.random_(n_c)  # 用一个离散均匀分布来填充当前的张量。
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.)  # 用于one-hot编码 zc_idx扩展为2维 zc_FT也就生成了二维上的one-hot编码
        # zc_idx = torch.empty(shape, dtype=torch.long).random_(n_c).cuda()
        # zc_FT = Tensor(shape, n_c).fill_(0).scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

        zc_idx = zc_idx.cuda() if cuda else zc_idx
        zc_FT = zc_FT.cuda() if cuda else zc_idx

    zc = Variable(zc_FT, requires_grad=req_grad)

    ## Gaussian-noisey vector generation
    # zc = Variable(Tensor(np.random.normal(0, 1, (shape, n_c))), requires_grad=req_grad)
    # zc = softmax(zc)
    # zc_idx = torch.argmax(zc, dim=1)

    # Return components of latent space variable
    # zn[batch_size, 隐空间dim] zc[batch_size, 聚类个数c_n] zc_idx[batch_size]
    return zn, zc, zc_idx


def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda() if cuda else alpha

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda() if cuda else interpolated

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda() if cuda else torch.ones(
                               prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()

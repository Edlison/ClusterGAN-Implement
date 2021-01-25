from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    from itertools import chain as ichain
    from clusgan.definitions import DATASETS_DIR, RUNS_DIR
    from clusgan.models import Generator_CNN, Encoder_CNN, Discriminator_CNN
    from clusgan.utils import save_model, calc_gradient_penalty, sample_z, cross_entropy
    from clusgan.datasets import get_dataloader, dataset_list
    from clusgan.plots import plot_train_loss

    torch.backends.cudnn.enabled = False  # 关闭cudnn

except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusgan', help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=1024, type=int, help="Batch size")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-w", "--wass_metric", dest="wass_metric", action='store_true',
                        help="Flag for Wasserstein metric")
    parser.add_argument("-g", "-–gpu", dest="gpu", default=5, type=int, help="GPU id to use")
    parser.add_argument("-k", "-–num_workers", dest="num_workers", default=1, type=int,
                        help="Number of dataset workers")
    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name
    device_id = args.gpu
    num_workers = args.num_workers

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    test_batch_size = 5000
    lr = 1e-4
    b1 = 0.5
    b2 = 0.9  # 99
    decay = 2.5 * 1e-5
    n_skip_iter = 1  # 5
    img_size = 28
    channels = 1

    # Latent space info
    latent_dim = 30  # 用于生成图片的隐空间向量dim
    n_c = 10  # 聚类个数
    betan = 10  # 计算zn损失时对应的参数
    betac = 10  # 计算zc损失时对应的参数

    # Wasserstein metric flag
    # Wasserstein metric flag
    wass_metric = args.wass_metric  # 是否使用W-GAN
    mtype = 'van'
    if (wass_metric):
        mtype = 'wass'

    # Make directory structure for this run
    sep_und = '_'
    run_name_comps = ['%iepoch' % n_epochs, 'z%s' % str(latent_dim), mtype, 'bs%i' % batch_size, run_name]
    run_name = sep_und.join(run_name_comps)

    # 生成输出的文件夹路径
    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)  # ./runs/mnist
    data_dir = os.path.join(DATASETS_DIR, dataset_name)  # ./datasets/mnist
    imgs_dir = os.path.join(run_dir, 'images')  # ./runs/mnist/images
    models_dir = os.path.join(run_dir, 'models')  # ./runs/mnist/models

    # 如果文件夹不存在创建文件夹
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print('\nResults to be saved in directory %s\n' % (run_dir))

    x_shape = (channels, img_size, img_size)

    cuda = True if torch.cuda.is_available() else False
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 使用的gpu的id
    if cuda: torch.cuda.set_device(device_id)

    # Loss function
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator TODO 三个网络的结构
    generator = Generator_CNN(latent_dim, n_c, x_shape)
    encoder = Encoder_CNN(latent_dim, n_c)
    discriminator = Discriminator_CNN(wass_metric=wass_metric)

    # 如果使用cuda 将网络结构 与 损失函数放到gpu上
    if cuda:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()

    # 如果使用GPU则转换为GPU上的张量类型
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Configure training data loader
    # 训练集
    dataloader = get_dataloader(dataset_name=dataset_name,
                                data_dir=data_dir,
                                batch_size=batch_size,
                                num_workers=num_workers)

    # Test data loader
    # 测试集
    testdata = get_dataloader(dataset_name=dataset_name, data_dir=data_dir, batch_size=test_batch_size, train_set=False)
    test_imgs, test_labels = next(iter(testdata))
    test_imgs = Variable(test_imgs.type(Tensor))

    # 将 生成器 与 Encoder 的参数连在一起 TODO 两个网络联合训练
    # 需要更新的网络可以看作两个 G+E 与 D
    ge_chain = ichain(generator.parameters(),
                      encoder.parameters())
    optimizer_GE = torch.optim.Adam(ge_chain, lr=lr, betas=(b1, b2),
                                    weight_decay=decay)  # (待优化的参数， lr: 学习率， betas: (一阶矩，二阶矩)平滑函数， weight_decay: 参数的值修改偏导数)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))  # 可以加上weight_decay参数

    # ----------
    #  Training
    # ----------
    # 记录测试时的loss
    ge_l = []  # G+E 的loss
    d_l = []  # D 的loss
    c_zn = []  # zn 的loss
    c_zc = []  # zc 的loss
    c_i = []  # 测试时的loss

    from tqdm import tqdm
    # Training loop 
    print('\nBegin training session with %i epochs...\n' % (n_epochs))
    for epoch in tqdm(range(n_epochs)):  # 在epoch上迭代
        for i, (imgs, itruth_label) in tqdm(enumerate(dataloader)):  # 在batch上迭代
            # Ensure generator/encoder are trainable
            # 训练Generator和Encoder
            generator.train()
            encoder.train()
            # Zero gradients for models
            # 梯度清零
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()

            # Configure input
            # 真实的照片
            real_imgs = Variable(imgs.type(Tensor))  # 为什么要用Variable储存？但是此时不需要梯度！

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------

            # 梯度清零
            optimizer_GE.zero_grad()

            # Sample random latent variables
            # 随机生成噪声 (zn, zc, zc_idx) 返回的是随机生成的具有梯度的Tensor
            zn, zc, zc_idx = sample_z(shape=imgs.shape[0],
                                      latent_dim=latent_dim,  # 隐空间维度 用一个30dim的隐空间向量表示一个sample
                                      n_c=n_c)

            # Generate a batch of images
            # 生成器生成的照片
            gen_imgs = generator(zn, zc)

            # Discriminator output from real and generated samples
            # 同时将生成的照片与真实的照片进入 判别器
            # 拿到各自的输出后面会用来 计算损失
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)

            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            # n个skip_iter后进一次encoder
            if (i % n_skip_iter == 0):
                # Encode the generated images
                # 生成器生成的gen_imgs放入encoder
                enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)

                # Calculate losses for z_n, z_c
                """
                Loss
                
                zn与encode后的enc_gen_zn
                zc与encode后的enc_gen_zc
                """
                zn_loss = mse_loss(enc_gen_zn, zn)  # zn使用MSELoss
                zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)  # zc使用交叉熵 因为one-hot，所以不用zc做损失？

                # Check requested metric
                if wass_metric:
                    # Wasserstein GAN loss
                    # W-GAN的loss
                    ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss
                else:
                    # Vanilla GAN loss
                    # V-GAN的loss
                    # TODO valid 与 v_loss 的意义？
                    valid = Variable(Tensor(gen_imgs.size(0), 1).fill_(1.0), requires_grad=False)  # 不需要梯度
                    v_loss = bce_loss(D_gen, valid)  # 生成器生成的图片的输出 与 valid 的loss
                    ge_loss = v_loss + betan * zn_loss + betac * zc_loss  # G+E loss 对应论文中的损失函数中的生成器与Encoder部分的loss

                # ge_loss中的参数进行反向传播 TODO 层上的权重？
                ge_loss.backward(retain_graph=True)  # 保留计算图为了之后训练 判别器
                # G+E网络中的参数进行梯度下降 TODO 神经元上的参数？
                optimizer_GE.step()  # 梯度下降

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # 梯度清零
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            if wass_metric:
                # Gradient penalty term
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)

                # Wasserstein GAN loss w/gradient penalty
                # W-GAN的损失
                d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty

            else:
                # Vanilla GAN loss
                # V-GAN的损失
                fake = Variable(Tensor(gen_imgs.size(0), 1).fill_(0.0), requires_grad=False)
                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                d_loss = (real_loss + fake_loss) / 2

            # 反向传播
            d_loss.backward()
            # 梯度下降
            optimizer_D.step()

        # ----
        # 一个epoach结束后

        # Save training losses
        # 保存一个epoach训练后的 G+E 与 D 的loss信息
        ge_l.append(ge_loss.item())
        d_l.append(d_loss.item())

        # Generator in eval mode
        # 进行eval
        generator.eval()
        encoder.eval()

        # Set number of examples for cycle calcs
        # 一个batch的前25个 绘制一个5*5的图 用于
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp

        ## Cycle through test real -> enc -> gen
        ## test_img -> E -> G -> fake_img loss test_img
        # 拿到测试集上的图片
        t_imgs, t_label = test_imgs.data, test_labels
        # r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        # Encode sample real instances
        # 将真实的图片进入Encoder
        e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
        # Generate sample instances from encoding
        # 使用Encoder的输出将隐向量e_tzn(encoder_test_zn), 与 e_tzc(encoder_test_zc)进入 生成器
        teg_imgs = generator(e_tzn, e_tzc)
        # Calculate cycle reconstruction loss
        # 计算测试集上的图片 与 encoder回去的隐变量生成的图片 之间的loss
        img_mse_loss = mse_loss(t_imgs, teg_imgs)
        # Save img reco cycle loss
        # 保存测试过程中的关于img的loss
        c_i.append(img_mse_loss.item())

        ## Cycle through randomly sampled encoding -> generator -> encoder
        ## zn_sample, zc_sample -> G -> fake_img -> E -> zn, zc loss zn_sample, zc_sample
        # 随机生成一个噪声
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp,
                                                 latent_dim=latent_dim,
                                                 n_c=n_c)
        # Generate sample instances
        # zn, zc通过G 生成 fake_img
        gen_imgs_samp = generator(zn_samp, zc_samp)
        # Encode sample instances
        # fake_img 通过 E 生成 zn_encoder, zc_encoder
        zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp)
        # Calculate cycle latent losses
        # zn, zc 与 zn_encoder, zc_encoder 计算loss
        lat_mse_loss = mse_loss(zn_e, zn_samp)
        lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)
        # lat_xe_loss = cross_entropy(zc_e_logits, zc_samp)
        # Save latent space cycle losses
        # 保存测试过程中的关于zn, zc的loss
        c_zn.append(lat_mse_loss.item())
        c_zc.append(lat_xe_loss.item())

        # Save cycled and generated examples!
        # 保存一个batch前n_samp个图片 TODO 为什么不直接用测试集上的？ 前面eval的时候已经有了test_imgs 与 fake_imgs
        r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        e_zn, e_zc, e_zc_logits = encoder(r_imgs)
        reg_imgs = generator(e_zn, e_zc)
        # 保存real_imgs 训练集上的
        save_image(r_imgs.data[:n_samp],
                   '%s/real_%06i.png' % (imgs_dir, epoch),
                   nrow=n_sqrt_samp, normalize=True)
        # 保存real_imgs -> E -> G -> reg_imgs 训练集上的
        save_image(reg_imgs.data[:n_samp],
                   '%s/reg_%06i.png' % (imgs_dir, epoch),
                   nrow=n_sqrt_samp, normalize=True)
        # 保存通过随机生成的噪声生成的 gen_imgs
        save_image(gen_imgs_samp.data[:n_samp],
                   '%s/gen_%06i.png' % (imgs_dir, epoch),
                   nrow=n_sqrt_samp, normalize=True)

        ## Generate samples for specified classes
        # 生成特定类别的样本
        stack_imgs = []
        for idx in range(n_c):  # 从0 - 10
            # Sample specific class
            # 随机产生噪声 此时固定类别 fix_class = idx!!!
            # TODO fix_class
            zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_c,  # shape = 10
                                                     latent_dim=latent_dim,
                                                     n_c=n_c,
                                                     fix_class=idx)  # fix_class = idx TODO 聚类的过程只能判别不同的类别，与index(0, 9)实际上是对不上的？

            # Generate sample instances
            # 通过生成器生成gen_imgs_samp
            gen_imgs_samp = generator(zn_samp, zc_samp)

            # 将在10个类别上生成的图片进行拼接
            if (len(stack_imgs) == 0):
                stack_imgs = gen_imgs_samp
            else:
                stack_imgs = torch.cat((stack_imgs, gen_imgs_samp), 0)

        # Save class-specified generated examples!
        # 保存最后生成的图片
        save_image(stack_imgs,
                   '%s/gen_classes_%06i.png' % (imgs_dir, epoch),
                   nrow=n_c, normalize=True)

        # 打印loss信息
        print("[Epoch %d/%d] \n" \
              "\tModel Losses: [D: %f] [GE: %f]" % (epoch,
                                                    n_epochs,
                                                    d_loss.item(),
                                                    ge_loss.item())
              )

        print("\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]" % (img_mse_loss.item(),
                                                               lat_mse_loss.item(),
                                                               lat_xe_loss.item())
              )

    # Save training results
    # 保存训练的参数
    train_df = pd.DataFrame({
        'n_epochs': n_epochs,
        'learning_rate': lr,
        'beta_1': b1,
        'beta_2': b2,
        'weight_decay': decay,
        'n_skip_iter': n_skip_iter,
        'latent_dim': latent_dim,
        'n_classes': n_c,
        'beta_n': betan,
        'beta_c': betac,
        'wass_metric': wass_metric,
        'gen_enc_loss': ['G+E', ge_l],
        'disc_loss': ['D', d_l],
        'zn_cycle_loss': ['$||Z_n-E(G(x))_n||$', c_zn],
        'zc_cycle_loss': ['$||Z_c-E(G(x))_c||$', c_zc],
        'img_cycle_loss': ['$||X-G(E(x))||$', c_i]
    })

    train_df.to_csv('%s/training_details.csv' % (run_dir))

    # Plot some training results
    # 绘制loss图
    plot_train_loss(df=train_df,
                    arr_list=['gen_enc_loss', 'disc_loss'],
                    figname='%s/training_model_losses.png' % (run_dir)
                    )

    plot_train_loss(df=train_df,
                    arr_list=['zn_cycle_loss', 'zc_cycle_loss', 'img_cycle_loss'],
                    figname='%s/training_cycle_loss.png' % (run_dir)
                    )

    # Save current state of trained models
    # 保存models参数
    model_list = [discriminator, encoder, generator]
    save_model(models=model_list, out_dir=models_dir)


if __name__ == "__main__":
    main()

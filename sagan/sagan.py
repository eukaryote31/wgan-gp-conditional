import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.nn.init import xavier_normal_
from torch.autograd import Variable, grad
import torchvision.utils as vutils
import torch.utils.data as data
import random
import os
import json
import math
from sacred import Experiment
from sacred.observers import MongoObserver
import time
from model import Generator, Discriminator
import inception_score


ex = Experiment('anime-sagan128-fc')
ex.observers.append(MongoObserver.create(url='10.128.0.10:27017', db_name='experiments'))


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@ex.config
def config():
    ysize = 45
    zsize = 100
    isize = 128

    batchsize = 64
    nepochs = 999

    discrim_down_blocks = 4
    discrim_self_attn = [0, 0, 0, 1, 0]
    discrim_filters = 16
    discrim_resblocks = 3

    gen_init_resblocks = 12
    gen_up_blocks = 4
    gen_inp_planes = 32
    gen_resblocks = 2
    gen_filters = 256
    gen_self_attn_init = [0] * (gen_init_resblocks + 1)
    gen_self_attn_up = [0, 1, 0, 0, 0]

    # number of discriminator updates per iter
    dratio = 5

    # number of generator updates per iter
    genperiter = 1

    # discriminator must be below this threshold for next generator iteration
    dcutoffthreshold = 999

    # Generator Adam parameters
    G_adam_alpha = 0.00005
    G_adam_beta1 = 0
    G_adam_beta2 = 0.9

    # Discriminator Adam parameters
    D_adam_alpha = 0.00008
    D_adam_beta1 = 0
    D_adam_beta2 = 0.9

    # number of iterations to disable generator at start
    discrim_only_iters = 50

    half = False

    resume = False
    resume_partial = False

    fakepoolsize = 2
    P_pool = 0.2

    loss_D_coeff = 1

    W_clip = 0.2
    W_weight = 5 / math.sqrt(W_clip)
    W_decay_exp = 0.9
    W_epochs = 0

    lambda_ = 3

    # interval to sample images generated
    sample_iter_interval = 100

    dataset_transforms = transforms.Compose([
        transforms.Resize((isize, isize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])


@ex.main
def main(_run, discrim_only_iters,
         sample_iter_interval, resume, resume_partial, dataset_transforms,
         dratio, genperiter, nepochs, batchsize, isize, ysize, zsize,
         D_adam_alpha, D_adam_beta1, D_adam_beta2, G_adam_alpha, G_adam_beta1,
         G_adam_beta2, dcutoffthreshold, gen_up_blocks, gen_inp_planes,
         gen_resblocks, gen_filters, gen_self_attn_up, gen_self_attn_init, discrim_down_blocks, discrim_self_attn,
         discrim_filters, half, discrim_resblocks, fakepoolsize, P_pool, gen_init_resblocks, loss_D_coeff,
         W_weight, W_decay_exp, W_epochs):
    print('Using', dev)
    dataset = CustomDatasetFolder(
        root='images', transform=dataset_transforms)
    init_vecs()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                             shuffle=True, num_workers=4)

    noise = torch.FloatTensor(batchsize, zsize).to(dev)
    yfake = torch.FloatTensor(batchsize, ysize).to(dev)
    yfixed = torch.FloatTensor(batchsize, ysize).to(dev)
    noisefixed = torch.FloatTensor(batchsize, zsize).normal_(0, 1).to(dev)

    loss = torch.nn.BCELoss()

    netG = Generator(isize, ysize, zsize, gen_up_blocks, gen_inp_planes,
                                     gen_resblocks, gen_filters, gen_self_attn_up, gen_self_attn_init, gen_init_resblocks)
    netD = Discriminator(ysize, zsize, isize, discrim_down_blocks, discrim_self_attn,
                                         discrim_filters, discrim_resblocks)

    # Initialize optimizers
    optimizerD = optim.Adam(
        netD.parameters(), lr=D_adam_alpha, betas=(D_adam_beta1, D_adam_beta2))
    optimizerG = optim.Adam(
        netG.parameters(), lr=G_adam_alpha, betas=(G_adam_beta1, G_adam_beta2))
    xavier_init(netG)
    xavier_init(netD)

    netG = netG.to(dev)
    netD = netD.to(dev)

    # Make directories
    try:
        os.mkdir('samples')
        os.mkdir('checkpoints')
    except Exception:
        pass

    # generate fixed y for samples
    gen_condit_vec_(yfixed)

    if resume:
        print('resume')
        # Load checkpoint from file
        checkpoint = torch.load(
            './checkpoints/checkpoint-{}.pth.tar'.format(input()))
        netD.load_state_dict(checkpoint['D_state_dict'])
        netG.load_state_dict(checkpoint['G_state_dict'])
#        optimizerG.load_state_dict(checkpoint['optimizerG'])
#        optimizerD.load_state_dict(checkpoint['optimizerD'])
        noisefixed = checkpoint['noisefixed'].to(dev)
        yfixed = checkpoint['yfixed'].to(dev)
        startepoch = checkpoint['epoch']
        gen_iters = checkpoint['gen_iters']
    elif resume_partial:
        checkpoint = torch.load('checkpoints/checkpoint-transfer.pth.tar')
        D_dict = netD.state_dict()
        D_pretrain_dict = checkpoint['D_state_dict']
        D_pretrain_dict = {k: v for k,
                           v in D_pretrain_dict.items() if k in D_dict}
        D_dict.update(D_pretrain_dict)
        netD.load_state_dict(D_pretrain_dict)

        G_dict = netG.state_dict()
        G_pretrain_dict = checkpoint['G_state_dict']
        G_pretrain_dict = {k: v for k,
                           v in G_pretrain_dict.items() if k in G_dict}
        G_dict.update(G_pretrain_dict)
        netG.load_state_dict(G_pretrain_dict)

    else:
        startepoch = 0
        gen_iters = 0


    labelsn1 = torch.FloatTensor(batchsize, 1).to(dev)
    labels1 = torch.FloatTensor(batchsize, 1).to(dev)
    fakepool = []
    labelsn1.data.fill_(-1.)
    labels1.data.fill_(1.)
    discr_iters = 0
    starttime = time.time()
    step = 0
    ct_discr = 0
    ct_gen = 0
    for epoch in range(startepoch, nepochs):
        ct_discr = 0
        ct_gen = 0
        _run.log_scalar('generator.mpenaltyweight', float(W_weight), step)

        data_iter = iter(dataloader)
        i = 0
        # Train on all batches
        while i < len(data_iter) - 1:
            # Clear gradient
            netD.zero_grad()
            # Train on real samples
            real, yreal = data_iter.next()

            real = real.to(dev)
            yreal = yreal.to(dev)

            usepooled = random.uniform(0, 1) < P_pool
            if usepooled and len(fakepool) > 0:
                fake = random.choice(fakepool)
            else:
                # Train on fake samples
                noise.normal_(0, 1)
                with torch.no_grad():
                    # compute G(z, y)
                    fake = netG(noise, yreal).detach()
                    fakepool.append(fake)
                    if len(fakepool) > fakepoolsize:
                        del fakepool[0]
            D_real = netD(real, yreal)
            D_fake = netD(fake, yreal)

            mean_D_fake = nn.functional.relu(1 + D_fake).mean()
            mean_D_real = nn.functional.relu(1 - D_real).mean() * loss_D_coeff

            realacc = labels1.eq(
                (D_real / D_real.abs()).round()).sum().float() / batchsize
            fakeacc = labelsn1.eq(
                (D_fake / D_fake.abs()).round()).sum().float() / batchsize

            gradpnlty = calc_gradient_penalty(netD, real, fake, yreal, yfake)

            cumulacc = (realacc + fakeacc) / 2
            loss_D = mean_D_fake + mean_D_real + gradpnlty
            loss_D.backward()

            optimizerD.step()
            ct_discr += 1

            D_gammas = [float(s.gamma) for s in netD.sa]

            print('D epoch %d batch %s | ( real %.4f   \t racc %.2f%%\t fake %.4f\t facc %.2f%%\t cumul %.4f)%s|%sDγ: %s|' % (
                epoch, str(i).rjust(4), mean_D_real.item(), realacc * 100., mean_D_fake.item(), fakeacc * 100., loss_D,
                '*' if usepooled else ' ', ' ' * 22, ', '.join(['%+.5f' % _ for _ in D_gammas])))
            i += 1
            _run.log_scalar('discrim.loss.real',
                            float(mean_D_real.item()), step)
            _run.log_scalar('discrim.loss.fake',
                            float(mean_D_fake.item()), step)
            _run.log_scalar('discrim.loss.cumul', float(loss_D), step)
            _run.log_scalar('discrim.accuracy.real', float(realacc), step)
            _run.log_scalar('discrim.accuracy.fake', float(fakeacc), step)
            _run.log_scalar('discrim.gradpnlty', float(gradpnlty), step)
            _run.log_scalar('discrim.loss.raw', float(mean_D_real.item() + mean_D_fake.item()), step)
            _run.log_scalar('discrim.accuracy.cumul', float(cumulacc), step)
            for gidx, g in enumerate(D_gammas):
                _run.log_scalar('discrim.gamma.' + str(gidx), float(g), step)
            step += 1

            if discr_iters >= discrim_only_iters and i % dratio == 0 and float(loss_D) < dcutoffthreshold:
                G_i = 0
                for p in netD.parameters():
                    p.requires_grad_(False)
                for G_i in range(genperiter):
                    # Update Generator

                    netG.zero_grad()
                    noise.normal_(0, 1)
                    genimg = netG(noise, yreal)
                    dg_z = netD(genimg, yreal)

                    if half:
                        dg_z = dg_z.float()

                    if W_weight > 0.01 and epoch < W_epochs:
                        mpenalty = calc_mode_penalty(netG, yreal, genimg) * W_weight
                    else:
                        mpenalty = 0

                    loss_G = -dg_z.mean() + mpenalty
                    loss_G.backward()
                    optimizerG.step()
                    Gacc = labels1.eq((dg_z / dg_z.abs()).round()
                                      ).sum().float() / batchsize

                    G_gammas = [float(s.gamma) for s in netG.sa]

                    for gidx, g in enumerate(G_gammas):
                        _run.log_scalar('generator.gamma.' + str(gidx), float(g), step)

                    print('G epoch {} iter {} |{} {}|  {}Gγ: {} | ({}) gp: {}'.format(
                        epoch,
                        str(gen_iters).rjust(5),
                        ' ' * 80,
                        '| {} | acc {}%'.format(('%.5f' % loss_G), int(Gacc * 100)).ljust(22),
                        ' ' * 12,
                        (', '.join(['%+.5f' % _ for _ in G_gammas])),
                        '%.3f' % float(W_weight),
                        float(gradpnlty),
                        ))

                    _run.log_scalar('generator.loss',
                                    float(loss_G.item()), step)
                    _run.log_scalar('generator.accuracy', float(Gacc), step)
                    _run.log_scalar('generator.mpenalty', float(mpenalty), step)
                    gen_iters += 1
                    step += 1
                    G_i += 1
                    ct_gen += 1

                    if gen_iters % sample_iter_interval == 0:
                        _run.log_scalar('generator.accuracy', float(Gacc), step)
                        netG.eval()
                        with torch.no_grad():
                            fake = netG(noisefixed, yfixed)
                        netG.train()
#                        incepmean, incepstd = inception_score.inception_score(fake, batch_size=batchsize)
#                        _run.log_scalar('inception.mean', float(incepmean), step)
#                        _run.log_scalar('inception.stddeviation', float(incepstd), step)
#                        print('INCEPTION m: {}'.for.clamp(W_clip)mat(incepmean))

                        rows = int(math.floor(math.sqrt(batchsize)))

                        vutils.save_image(fake[:rows ** 2],
                        'samples/fake_samples_epoch_%d_genit_%d.png' % (epoch, gen_iters), nrow=rows) # int(math.ceil(math.sqrt(batchsize)))
                        ex.add_artifact(
                        'samples/fake_samples_epoch_%d_genit_%d.png' % (epoch, gen_iters))


                for p in netD.parameters():
                    p.requires_grad_(True)

            discr_iters += 1
        _run.log_scalar('discrpergen', float(ct_discr / ct_gen), step)

        # decay mpenalty
        W_weight *= W_decay_exp
        # do checkpointing
        save_checkpoint({
            'epoch': epoch + 1,
            'G_state_dict': netG.state_dict(),
            'D_state_dict': netD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'noisefixed': noisefixed,
            'yfixed': yfixed,
            'gen_iters': gen_iters,
        }, epoch)

        now = time.time()
        elapsed = now - starttime
        perepoch = elapsed / (epoch + 1)
        remaining = perepoch * (nepochs - epoch - 1)
        print('Epoch {}/{} done, {}s elapsed ({}s/epoch, {}s remaining)'
              .format(epoch, nepochs, elapsed, perepoch, remaining))


def xavier_init(model):
    for param in model.parameters():
        if len(param.size()) == 2:
            xavier_normal_(param)


def save_checkpoint(state, epoch):
    try:
        filename = 'checkpoints/checkpoint-%d.pth.tar' % epoch
        torch.save(state, filename)
#        os.remove('checkpoints/checkpoint-latest.pth.tar')
    except Exception:
        pass
    else:
        if epoch > 2:
            os.remove('checkpoints/checkpoint-%d.pth.tar' % (epoch - 2))


@ex.capture
def init_vecs(batchsize):
    global eyecolor_onehot
    global hairlen_onehot
    global haircolor_onehot
    global eco
    global hlo
    global hco

    eyecolor_onehot = torch.LongTensor(batchsize)
    hairlen_onehot = torch.LongTensor(batchsize)
    haircolor_onehot = torch.LongTensor(batchsize)

    eco = torch.ones(batchsize, 11)
    hlo = torch.ones(batchsize, 3)
    hco = torch.ones(batchsize, 12)

    eyecolor_onehot = eyecolor_onehot.to(dev)
    hairlen_onehot = hairlen_onehot.to(dev)
    haircolor_onehot = haircolor_onehot.to(dev)
    eco = eco.to(dev)
    hlo = hlo.to(dev)
    hco = hco.to(dev)


@ex.capture
def gen_condit_vec_(y, batchsize):
    y.fill_(0)
    # gender
    y[:, 0].random_(0, 2)
    y[:, 1] = 1 - y[:, 0]

    eyecolor_onehot.random_(0, 11)
    hairlen_onehot.random_(0, 3)
    haircolor_onehot.random_(0, 12)

    # encode onehot
    y[:, 2:13].scatter_(1, eyecolor_onehot.expand(
        11, batchsize).transpose(0, 1), eco)
    y[:, 13:16].scatter_(1, hairlen_onehot.expand(
        3, batchsize).transpose(0, 1), hlo)
    y[:, 16:28].scatter_(1, haircolor_onehot.expand(
        12, batchsize).transpose(0, 1), hco)

    y[:, 28:44].random_(1, 5)
    y[:, 28:44] /= 4
    y[:, 28:44].floor_()

    # mask out gender specific
    y[:, 30] *= y[:, 1]
    y[:, 32] *= y[:, 1]

    y[:, 44].uniform_(0, 1)

    return y


@ex.capture
def gen_condit_vec(batchsize, ysize):
    y = FloatTensor(ysize)
    y.fill_(0)
    # gender
    y[:, 0].random_(0, 2)
    y[:, 1] = 1 - y[:, 0]

    eyecolor_onehot.random_(0, 11)
    hairlen_onehot.random_(0, 3)
    haircolor_onehot.random_(0, 12)

    # encode onehot
    y[:, 2:13].scatter_(1, eyecolor_onehot.expand(
        11, batchsize).transpose(0, 1), eco)
    y[:, 13:16].scatter_(1, hairlen_onehot.expand(
        3, batchsize).transpose(0, 1), hlo)
    y[:, 16:28].scatter_(1, haircolor_onehot.expand(
        12, batchsize).transpose(0, 1), hco)

    y[:, 28:44].random_(1, 5)
    y[:, 28:44] /= 4
    y[:, 28:44].floor_()

    # mask out gender specific
    y[:, 30] *= y[:, 1]
    y[:, 32] *= y[:, 1]

    y[:, 44].uniform_(0, 1)

    return y


@ex.capture
def calc_mode_penalty(netG, y, g2, zsize, ysize, W_clip, W_weight, batchsize, isize):
    z1 = torch.FloatTensor(batchsize, zsize).normal_(0, 1).to(dev)

    g1 = netG(z1, y)
    diff = g1.view(g2.numel()) - g2.view(g2.numel())
    diff = diff.abs()
    diff = diff.clamp(min=1e-8, max=W_clip)
    diff = diff.sqrt()
    return -diff.mean()


class CustomDatasetFolder(data.Dataset):
    def __init__(self, root, loader=dset.folder.default_loader, transform=None, target_transform=None):
        with ex.open_resource('vectors.json') as fh:
            imgdict = json.load(fh)
        self.root = root
        self.loader = loader

        samples = []
        for f in os.listdir(root):
            samples.append((f, imgdict[f[:-4] + '.jpg']))

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        target = torch.FloatTensor(target)
        sample = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


@ex.capture
def calc_gradient_penalty(netD, real_data, fake_data, yreal, yfake, batchsize, lambda_, isize):
    # via https://github.com/caogang/wgan-gp/
    alpha1 = torch.rand(batchsize, 1)
    alpha = alpha1.expand(batchsize, real_data.nelement(
    ) // batchsize).contiguous().view(batchsize, 3, isize, isize)
    alpha = alpha.cuda().to(dev)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates.to(dev).requires_grad_(True)
    D_interpolates = netD(interpolates, yreal)

    gradients = grad(outputs=D_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(D_interpolates.size()).to(dev),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty


if __name__ == '__main__':
    ex.run_commandline()

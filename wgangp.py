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
import os
import json
from sacred import Experiment
from sacred.observers import MongoObserver
import time

ex = Experiment('anime-wgangp')
ex.observers.append(MongoObserver.create(url='10.128.0.10:27017',
                                         db_name='experiments'))


@ex.config
def config():
    ysize = 45
    zsize = 100
    isize = 64
    batchsize = 64
    nepochs = 100

    gen_extra_blocks = 16
    gen_up_count = 3
    discrim_extra_blocks = 0
    discrim_blk_per_seg = 3

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
    D_adam_alpha = 0.00005
    D_adam_beta1 = 0
    D_adam_beta2 = 0.9

    # gradient penalty weight
    lambda_ = 10

    # number of iterations to disable generator at start
    discrim_only_iters = 75

    # interval to sample images generated
    sample_epoch_interval = 1

    cuda = torch.cuda.is_available()
    resume = False
    resume_partial = False

    dataset_transforms = transforms.Compose([
        transforms.Resize((isize, isize)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])


@ex.main
def main(_run, lambda_, discrim_only_iters,
         sample_epoch_interval, cuda, resume, resume_partial, dataset_transforms,
         dratio, genperiter, nepochs, batchsize, isize, ysize, zsize,
         D_adam_alpha, D_adam_beta1, D_adam_beta2, G_adam_alpha, G_adam_beta1,
         G_adam_beta2, dcutoffthreshold):
    if cuda:
        print('Using CUDA.')
    else:
        print('Using CPU.')
    starttime = time.time()
    dataset = CustomDatasetFolder(
        root='cropped256', transform=dataset_transforms)
    init_vecs()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                             shuffle=False, num_workers=2)

    noise = torch.FloatTensor(batchsize, zsize)
    yfake = torch.FloatTensor(batchsize, ysize)
    yfixed = torch.FloatTensor(batchsize, ysize)
    noisefixed = torch.FloatTensor(batchsize, zsize).normal_(0, 1)

    loss = torch.nn.BCELoss()

    netG = Generator()
    netD = Discriminator()

    # Initialize optimizers
    optimizerD = optim.Adam(
        netD.parameters(), lr=D_adam_alpha, betas=(D_adam_beta1, D_adam_beta2))
    optimizerG = optim.Adam(
        netG.parameters(), lr=G_adam_alpha, betas=(G_adam_beta1, G_adam_beta2))
    xavier_init(netG)
    xavier_init(netD)

    one = torch.FloatTensor([1])
    mone = one * -1

    # Make directories
    try:
        os.mkdir('samples')
        os.mkdir('checkpoints')
    except Exception:
        pass

#    train_log = open('checkpoints/train.log', 'w')
    if cuda:
        netG = netG.cuda()
        netD = netD.cuda()
        one = one.cuda()
        mone = mone.cuda()
        noise = noise.cuda()
        noisefixed = noisefixed.cuda()
        yfake = yfake.cuda()
        yfixed = yfixed.cuda()

    # generate fixed y for samples
    gen_condit_vec_(yfixed)

    if resume:
        # Load checkpoint from file
        checkpoint = torch.load('checkpoints/checkpoint-latest.pth.tar')
        netD.load_state_dict(checkpoint.D_state_dict)
        netG.load_state_dict(checkpoint.G_state_dict)
        optimizerD.load_state_dict(checkpoint.optimizerD)
        optimizerG.load_state_dict(checkpoint.optimizerG)
        noisefixed = checkpoint.noisefixed
        yfixed = checkpoint.yfixed
        if cuda:
            noisefixed = noisefixed.cuda()
            yfixed = yfixed.cuda()
        startepoch = checkpoint.epoch
    elif resume_partial:
        checkpoint = torch.load('checkpoints/checkpoint-transfer.pth.tar')
        D_dict = netD.state_dict()
        D_pretrain_dict = checkpoint.D_state_dict
        D_pretrain_dict = {k: v for k, v in D_pretrain_dict.items() if k in D_dict}
        D_dict.update(D_pretrain_dict)
        netD.load_state_dict(D_pretrain_dict)

        G_dict = netG.state_dict()
        G_pretrain_dict = checkpoint.G_state_dict
        G_pretrain_dict = {k: v for k, v in G_pretrain_dict.items() if k in G_dict}
        G_dict.update(G_pretrain_dict)
        netG.load_state_dict(G_pretrain_dict)

    else:
        startepoch = 0

    labelsn1 = torch.FloatTensor(batchsize, 1)
    labels1 = torch.FloatTensor(batchsize, 1)

    if cuda:
        labelsn1 = labelsn1.cuda()
        labels1 = labels1.cuda()
    labelsn1.data.fill_(-1.)
    labels1.data.fill_(1.)
    discr_iters = 0
    gen_iters = 0
    step = 0
    for epoch in range(startepoch, nepochs):

        data_iter = iter(dataloader)
        i = 0
        # Train on all batches
        while i < len(data_iter) - 1:
            # Clear gradient
            netD.zero_grad()
            # Train on real samples
            real, yreal = data_iter.next()
            if cuda:
                real = real.cuda()
                yreal = yreal.cuda()
            batchsz = real.size(0)

            if cuda:
                real = real.cuda()

            # Train on fake samples
            noise.normal_(0, 1)
            with torch.no_grad():
                # compute G(z, y)
                gen_condit_vec_(yfake)
                fake = netG(noise, yreal).detach()

            D_real = netD(real, yreal)
            D_fake = netD(fake, yreal)

            mean_D_fake = D_fake.mean()
            mean_D_real = D_real.mean()

            realacc = labels1.eq((D_real / D_real.abs()).round()).sum().float() / batchsize
            fakeacc = labelsn1.eq((D_fake / D_fake.abs()).round()).sum().float() / batchsize

            gradient_penalty = calc_gradient_penalty(netD, real, fake, yreal, yfake)
            cumulacc = (realacc + fakeacc) / 2
            loss_D = mean_D_fake - mean_D_real + gradient_penalty
            loss_D.backward(one)
            W_loss = mean_D_fake - mean_D_real

            optimizerD.step()

            print('D epoch %d batch %s | BCE ( real %.4f   \t racc %.2f%%\t fake %.4f\t facc %.2f%%\t cumul %.4f\t gradpnlty %.4f\t wass %.4f )' % (
                epoch, str(i).rjust(4), mean_D_real.item(), realacc * 100., mean_D_fake.item(), fakeacc * 100., loss_D, gradient_penalty, W_loss))
#            train_log.write('D epoch %d batch %d | BCE ( real %f\tfake %f\tcumul %f )\n' % (
#                epoch, i, mean_D_real.item(), mean_D_fake.item(), mean_D_real.item() + mean_D_fake.item()))
#            train_log.flush()
            i += 1
            _run.log_scalar('discrim.loss.real', float(mean_D_real.item()), step)
            _run.log_scalar('discrim.loss.fake', float(mean_D_fake.item()), step)
            _run.log_scalar('discrim.loss.wasserstein', float(W_loss), step)
            _run.log_scalar('discrim.loss.cumul', float(loss_D), step)
            _run.log_scalar('discrim.accuracy.real', float(realacc), step)
            _run.log_scalar('discrim.accuracy.fake', float(fakeacc), step)
            _run.log_scalar('discrim.accuracy.cumul', float(cumulacc), step)
            _run.log_scalar('discrim.grad_penalty', float(gradient_penalty), step)
            step += 1

            if discr_iters >= discrim_only_iters and i % dratio == 0 and loss_D < dcutoffthreshold:
                G_i = 0
#                dcutoffthreshold = (dcutoffthreshold + loss_D) / 2

                for p in netD.parameters():
                    p.requires_grad_(False)
                for G_i in range(genperiter):
                    # Update Generator

                    gen_condit_vec_(yfake)
                    netG.zero_grad()
                    noise.normal_(0, 1)
                    genimg = netG(noise, yfake)
                    dg_z = netD(genimg, yfake)

                    loss_G = dg_z.mean()
                    loss_G.backward(mone)
                    optimizerG.step()
                    Gacc = labels1.eq((dg_z / dg_z.abs()).round()).sum().float() / batchsize

                    print('G epoch {} iter {} | BCE {} | acc {}%'.format(
                        epoch, str(gen_iters).rjust(5), loss_G, int(Gacc * 100)))
#                    train_log.write(
#                        'G epoch {} iter {} | BCE {} | acc {}%\n'.format(epoch, G_i, loss_G.item(), Gacc * 100))
#                    train_log.flush()

                    _run.log_scalar('generator.loss', float(loss_G.item()), step)
                    _run.log_scalar('generator.accuracy', float(Gacc), step)
                    gen_iters += 1
                    step += 1
                    G_i += 1
#                    if loss_G.item() < 0.3:
#                        break

                for p in netD.parameters():
                    p.requires_grad_(True)

                # save samples
#                vutils.save_image(fake.data.view(batchsize, 3, isize, isize),
#                                  'samples/fake_samples_epoch_%d_step_%d.png' % epoch, step)
#                ex.add_artifact('samples/fake_samples_epoch_%d_step_%d.png' % epoch, step)


            discr_iters += 1
        if epoch % sample_epoch_interval == 0:
            fake = netG(noisefixed, yfixed)
            vutils.save_image(fake.data.view(batchsize, 3, isize, isize),
                              'samples/fake_samples_epoch_%d_genit_%d.png' % (epoch, gen_iters))
            ex.add_artifact('samples/fake_samples_epoch_%d_genit_%d.png' % (epoch, gen_iters))

            # do checkpointing
            save_checkpoint({
                'epoch': epoch + 1,
                'G_state_dict': netG.state_dict(),
                'D_state_dict': netD.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'noisefixed': noisefixed,
                'yfixed': yfixed,
            }, epoch)

        now = time.time()
        elapsed = now - starttime
        perepoch = elapsed / (epoch + 1)
        remaining = perepoch * (nepochs - epoch - 1)
        print('Epoch {}/{} done, {}s elapsed ({}s/epoch, {}s remaining)'
              .format(epoch, nepochs, elapsed, perepoch, remaining))

class DiscrimBlock(nn.Module):
    @ex.capture
    def __init__(self, in_channels, sidedim):
        super(DiscrimBlock, self).__init__()
        self.pad1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3)
        self.ln1 = nn.LayerNorm((in_channels, sidedim, sidedim))
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.pad2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3)
        self.ln2 = nn.LayerNorm((in_channels, sidedim, sidedim))
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = x
        out = self.ln1(out)
        out = self.pad1(out)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.ln2(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out += residual
        out = self.relu2(out)
        return out


class DiscrimSegment(nn.Module):
    def __init__(self, in_channels, kern, sidedim, nblocks):
        super(DiscrimSegment, self).__init__()
        main = nn.Sequential()

        for i in range(nblocks):
            main.add_module('discrim_block_{}'.format(i), DiscrimBlock(in_channels, sidedim))
        if kern == 4:
            main.add_module('discrim_conv_pad',
                            nn.ReflectionPad2d((1, 2, 1, 2)))
        else:
            main.add_module('discrim_conv_pad', nn.ReflectionPad2d(1))

        main.add_module('discrim_conv', nn.Conv2d(
            in_channels, in_channels * 2, kern, 2))
        main.add_module('discrim_lnorm', nn.LayerNorm((in_channels * 2, sidedim // 2, sidedim // 2)))
        main.add_module('discrim_relu', nn.LeakyReLU(inplace=True))
        self.main = main

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    @ex.capture
    def __init__(self, ysize, isize, discrim_extra_blocks, discrim_blk_per_seg):
        super(Discriminator, self).__init__()
        main = nn.Sequential()

        main.add_module('init_conv_pad', nn.ReplicationPad2d(1))
        main.add_module('init_conv', nn.Conv2d(3 + ysize, 32, 4, 2))
        main.add_module('init_relu', nn.LeakyReLU(inplace=True))

        for i in range(discrim_extra_blocks):
            main.add_module('discrim_extra_block_{}'.format(i), DiscrimBlock(32, isize // 2))

        filters = 32
        sidedim = isize // 2
        n_segs = 0
        while sidedim > 4:
            main.add_module('discrimseg_{}'.format(
                filters), DiscrimSegment(filters, 3 if n_segs > 1 else 4, sidedim, discrim_blk_per_seg))
            sidedim //= 2
            filters *= 2
            n_segs += 1

        score = nn.Sequential()
        self.fc_size = (isize // (2 ** (n_segs + 1))) ** 2 * (filters)
        self.ln = nn.LayerNorm((self.fc_size))
        score.add_module('fc', nn.Linear(
            self.fc_size, 1))
#        score.add_module('tanh', nn.Tanh())

        self.main = main
        self.score = score

    @ex.capture
    def forward(self, x, y, isize, ysize):
        y = y.view(-1, ysize, 1, 1).expand(-1, ysize, isize, isize)
        x = torch.cat((x, y), 1)
        x = self.main(x)
        x = x.view(-1, self.fc_size)
        x = self.ln(x)
        x = self.score(x)

        return x


class GenResBlock(nn.Module):
    def __init__(self, in_channels):
        super(GenResBlock, self).__init__()

        self.pad1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pad2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class GenUpBlock(nn.Module):
    def __init__(self, in_channels):
        super(GenUpBlock, self).__init__()

        main = nn.Sequential()

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, 256, 3)
        self.shuf = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.shuf(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Generator(nn.Module):
    @ex.capture
    def __init__(self, isize, ysize, zsize, gen_extra_blocks, gen_up_count):
        super(Generator, self).__init__()
        self.latent = nn.Linear(
            zsize + ysize, 64 * (isize // (2 ** gen_up_count)) * (isize // (2 ** gen_up_count)))
        self.bn_1 = nn.BatchNorm2d(64)
        self.relu_1 = nn.ReLU(inplace=True)
        main = nn.Sequential()

        for i in range(gen_extra_blocks):
            main.add_module('res_block_{}'.format(i), GenResBlock(64))

        main.add_module('mid_bn', nn.BatchNorm2d(64))
        main.add_module('mid_relu', nn.ReLU())
        self.main = main

        upscale = nn.Sequential()
        for i in range(gen_up_count):
            upscale.add_module('up_{}'.format(i), GenUpBlock(64))
        upscale.add_module('padding', nn.ZeroPad2d(4))
        upscale.add_module('convf', nn.Conv2d(64, 3, 9))
        upscale.add_module('sigmoid', nn.Sigmoid())
        self.upscale = upscale

    @ex.capture
    def forward(self, z, y, isize, gen_up_count):
        # concatenate latent noise vector and conditional vector
        z = self.latent(torch.cat((z, y), 1))
        z = z.view(-1, 64, (isize // (2 ** gen_up_count)), (isize // (2 ** gen_up_count)))
        z = self.bn_1(z)
        z = self.relu_1(z)
        residual = z
        z = self.main(z)
        z += residual
        z = self.upscale(z)

        return z


def xavier_init(model):
    for param in model.parameters():
        if len(param.size()) == 2:
            xavier_normal_(param)


def save_checkpoint(state, epoch):
    filename = 'checkpoints/checkpoint-%d.pth.tar' % epoch
    torch.save(state, filename)
    try:
        os.remove('checkpoints/checkpoint-latest.pth.tar')
    except Exception:
        pass
    os.symlink(filename, 'checkpoints/checkpoint-latest.pth.tar')
#    ex.add_artifact(filename)


@ex.capture
def calc_gradient_penalty(netD, real_data, fake_data, yreal, yfake, batchsize, cuda, lambda_, isize):
    # via https://github.com/caogang/wgan-gp/
    alpha1 = torch.rand(batchsize, 1)
    alpha = alpha1.expand(batchsize, real_data.nelement(
    ) / batchsize).contiguous().view(batchsize, 3, isize, isize)
    alpha = alpha.cuda() if cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if cuda:
        interpolates = interpolates.cuda()

    D_interpolates = netD(interpolates, yreal)

    gradients = grad(outputs=D_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(D_interpolates.size()).cuda() if cuda else torch.ones(
                                  D_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty


@ex.capture
def init_vecs(batchsize, cuda):
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
    hco = torch.ones(batchsize, 13)

    if cuda:
        eyecolor_onehot = eyecolor_onehot.cuda()
        hairlen_onehot = hairlen_onehot.cuda()
        haircolor_onehot = haircolor_onehot.cuda()
        eco = eco.cuda()
        hlo = hlo.cuda()
        hco = hco.cuda()


@ex.capture
def gen_condit_vec_(y, batchsize):
    y.fill_(0)
    # gender
    y[:, 0].random_(0, 2)
    y[:, 1] = 1 - y[:, 0]

    eyecolor_onehot.random_(0, 11)
    hairlen_onehot.random_(0, 3)
    haircolor_onehot.random_(0, 13)

    # encode onehot
    y[:, 2:13].scatter_(1, eyecolor_onehot.expand(
        11, batchsize).transpose(0, 1), eco)
    y[:, 13:16].scatter_(1, hairlen_onehot.expand(
        3, batchsize).transpose(0, 1), hlo)
    y[:, 16:29].scatter_(1, haircolor_onehot.expand(
        13, batchsize).transpose(0, 1), hco)

    y[:, 29:45].random_(0, 2)

    # mask out gender specific
    y[:, 31] *= y[:, 1]
    y[:, 33] *= y[:, 1]

    return y


class CustomDatasetFolder(data.Dataset):
    def __init__(self, root, loader=dset.folder.default_loader, transform=None, target_transform=None):
        with ex.open_resource('vectors.json') as fh:
            imgdict = json.load(fh)
        self.root = root
        self.loader = loader

        samples = []
        for f in os.listdir(root):
            samples.append((f, imgdict[f]))

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


if __name__ == '__main__':
    ex.run_commandline()

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

ex = Experiment('anime-wgangp-fc')
ex.observers.append(MongoObserver.create(url='10.128.0.10:27017', db_name='experiments'))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@ex.config
def config():
    ysize = 45
    zsize = 100
    isize = 128
    batchsize = 64
    nepochs = 1000

    gen_extra_blocks = 16
    gen_up_count = 3
    discrim_extra_blocks = 0
    discrim_blk_per_seg = 2

    # number of discriminator updates per iter
    dratio = 2

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
    discrim_only_iters = 0

    # interval to sample images generated
    sample_epoch_interval = 1

    cuda = torch.cuda.is_available()
    resume = True
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
        root='images', transform=dataset_transforms)
    init_vecs()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                             shuffle=False, num_workers=2)

    noise = torch.FloatTensor(batchsize, zsize).to(dev)
    yfixed = torch.FloatTensor(batchsize, ysize).to(dev)
    noisefixed = torch.FloatTensor(batchsize, zsize).normal_(0, 1).to(dev)

    netG = nn.DataParallel(Generator()).to(dev)
    netD = nn.DataParallel(Discriminator()).to(dev)

    # Initialize optimizers
    optimizerD = optim.Adam(
        netD.parameters(), lr=D_adam_alpha, betas=(D_adam_beta1, D_adam_beta2))
    optimizerG = optim.Adam(
        netG.parameters(), lr=G_adam_alpha, betas=(G_adam_beta1, G_adam_beta2))
    xavier_init(netG)
    xavier_init(netD)

    # Make directories
    try:
        os.mkdir('samples')
        os.mkdir('checkpoints')
    except Exception:
        pass

#    train_log = open('checkpoints/train.log', 'w')

    # generate fixed y for samples
    gen_condit_vec_(yfixed)

    if resume:
        print('resume')
        # Load checkpoint from file
        checkpoint = torch.load('./checkpoints/checkpoint-{}.pth.tar'.format(input()))
        netD.load_state_dict(checkpoint['D_state_dict'])
        netG.load_state_dict(checkpoint['G_state_dict'])
#        optimizerD.load_state_dict(checkpoint['optimizerD'])
#        optimizerG.load_state_dict(checkpoint['optimizerG'])
        noisefixed = checkpoint['noisefixed'].to(dev)
        yfixed = checkpoint['yfixed'].to(dev)
        startepoch = checkpoint['epoch']
    elif resume_partial:
        checkpoint = torch.load('checkpoints/checkpoint-transfer.pth.tar')
        D_dict = netD.state_dict()
        D_pretrain_dict = checkpoint['D_state_dict']
        D_pretrain_dict = {k: v for k, v in D_pretrain_dict.items() if k in D_dict}
        D_dict.update(D_pretrain_dict)
        netD.load_state_dict(D_pretrain_dict)

        G_dict = netG.state_dict()
        G_pretrain_dict = checkpoint['G_state_dict']
        G_pretrain_dict = {k: v for k, v in G_pretrain_dict.items() if k in G_dict}
        G_dict.update(G_pretrain_dict)
        netG.load_state_dict(G_pretrain_dict)

    else:
        startepoch = 0

    labelsn1 = torch.FloatTensor(batchsize, 1).to(dev)
    labels1 = torch.FloatTensor(batchsize, 1).to(dev)

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
            real = real.to(dev)
            yreal = yreal.to(dev)

            # Train on fake samples
            noise.normal_(0, 1)
            with torch.no_grad():
                # compute G(z, y)
                fake = netG(noise, yreal).detach()

            D_real = netD(real, yreal)
            D_fake = netD(fake, yreal)

            loss_D_fake = nn.functional.relu(1 + D_fake).mean()
            loss_D_real = nn.functional.relu(1 - D_real).mean()

            realacc = labels1.eq((D_real / D_real.abs()).round()).sum().float() / batchsize
            fakeacc = labelsn1.eq((D_fake / D_fake.abs()).round()).sum().float() / batchsize

            gradient_penalty = calc_gradient_penalty(netD, real, fake, yreal)
            cumulacc = (realacc + fakeacc) / 2
            loss_D = loss_D_fake + loss_D_real + gradient_penalty
            W_loss = loss_D_fake + loss_D_real
            loss_D.backward()

            optimizerD.step()

            D_gamma = float(netD.module.sa.gamma)

            print('D epoch %d batch %s | BCE ( real %.4f   \t racc %.2f%%\t fake %.4f\t facc %.2f%%\t cumul %.4f\t gradpnlty %.4f\t wass %.4f\t gamma %.5f )' % (
                epoch, str(i).rjust(4), loss_D_real.item(), realacc * 100., loss_D_fake.item(), fakeacc * 100., loss_D, gradient_penalty, W_loss, D_gamma))
#            train_log.write('D epoch %d batch %d | BCE ( real %f\tfake %f\tcumul %f )\n' % (
#                epoch, i, loss_D_real.item(), loss_D_fake.item(), loss_D_real.item() + loss_D_fake.item()))
#            train_log.flush()
            i += 1
            _run.log_scalar('discrim.loss.real', float(loss_D_real.item()), step)
            _run.log_scalar('discrim.loss.fake', float(loss_D_fake.item()), step)
            _run.log_scalar('discrim.loss.wasserstein', float(W_loss), step)
            _run.log_scalar('discrim.loss.cumul', float(loss_D), step)
            _run.log_scalar('discrim.accuracy.real', float(realacc), step)
            _run.log_scalar('discrim.accuracy.fake', float(fakeacc), step)
            _run.log_scalar('discrim.accuracy.cumul', float(cumulacc), step)
            _run.log_scalar('discrim.grad_penalty', float(gradient_penalty), step)
            _run.log_scalar('discrim.gamma', float(D_gamma), step)
            step += 1

            if (discr_iters >= discrim_only_iters) and i % dratio == 0 and loss_D < dcutoffthreshold:
                G_i = 0
#                dcutoffthreshold = (dcutoffthreshold + loss_D) / 2

                for p in netD.parameters():
                    p.requires_grad_(False)
                for G_i in range(genperiter):
                    # Update Generator

                    netG.zero_grad()
                    noise.normal_(0, 1)
                    genimg = netG(noise, yreal)
                    dg_z = netD(genimg, yreal)

                    loss_G = dg_z.mean()
                    (-loss_G).backward()
                    optimizerG.step()
                    Gacc = labels1.eq((dg_z / dg_z.abs()).round()).sum().float() / batchsize

                    G_gamma = float(netG.module.sa.gamma)
                    print('G epoch {} iter {} | BCE {} | acc {}% | gamma {}'.format(
                        epoch, str(gen_iters).rjust(5), loss_G, int(Gacc * 100), G_gamma))
#                    train_log.write(
#                        'G epoch {} iter {} | BCE {} | acc {}%\n'.format(epoch, G_i, loss_G.item(), Gacc * 100))
#                    train_log.flush()

                    _run.log_scalar('generator.loss', float(loss_G.item()), step)
                    _run.log_scalar('generator.gamma', float(G_gamma), step)
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
            with torch.no_grad():
                netG.eval()
                fake = netG(noisefixed, yfixed)
                netG.train()

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
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3)
        self.ln1 = nn.LayerNorm((in_channels, sidedim, sidedim))
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.pad2 = nn.ReflectionPad2d(1)
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

        main.add_module('init_conv_pad', nn.ReflectionPad2d(1))
        main.add_module('init_conv', nn.Conv2d(3 + ysize, 32, 4, 2))
        main.add_module('init_relu', nn.LeakyReLU(inplace=True))

        for i in range(discrim_extra_blocks):
            main.add_module('discrim_extra_block_{}'.format(i), DiscrimBlock(32, isize // 2))

        filters = 32
        sidedim = isize // 2
        n_segs = 0
        while sidedim > 4:
            if n_segs == 3:
                self.sa = Self_Attn(filters)
                main.add_module('sa', self.sa)
            main.add_module('discrimseg_{}'.format(
                filters), DiscrimSegment(filters, 3, sidedim, discrim_blk_per_seg))

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

        return x.float()


class GenResBlock(nn.Module):
    def __init__(self, in_channels):
        super(GenResBlock, self).__init__()

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pad2 = nn.ReflectionPad2d(1)
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
        self.sa = Self_Attn(64)
        for i in range(gen_up_count):
            if i == 1:
                upscale.add_module('sa', self.sa)
            upscale.add_module('up_{}'.format(i), GenUpBlock(64))
#        upscale.add_module('resf', GenResBlock(64))
        upscale.add_module('padding', nn.ReflectionPad2d(4))
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
def calc_gradient_penalty(netD, real_data, fake_data, yreal, batchsize, lambda_, isize):
    # via https://github.com/caogang/wgan-gp/
    alpha1 = torch.rand(batchsize, 1)
    alpha = alpha1.expand(batchsize, real_data.nelement(
    ) // batchsize).contiguous().view(batchsize, 3, isize, isize)
    alpha = alpha.to(dev)

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).to(dev)

    interpolates.requires_grad_(True)
    D_interpolates = netD(interpolates, yreal)

    gradients = grad(outputs=D_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(D_interpolates.size()).to(dev),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty


@ex.capture
def init_vecs(batchsize):
    global eyecolor_onehot
    global hairlen_onehot
    global haircolor_onehot
    global eco
    global hlo
    global hco

    eyecolor_onehot = torch.LongTensor(batchsize).to(dev)
    hairlen_onehot = torch.LongTensor(batchsize).to(dev)
    haircolor_onehot = torch.LongTensor(batchsize).to(dev)

    eco = torch.ones(batchsize, 11).to(dev)
    hlo = torch.ones(batchsize, 3).to(dev)
    hco = torch.ones(batchsize, 12).to(dev)


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


def save_checkpoint(state, epoch):
    try:
        filename = 'checkpoints/checkpoint-%d.pth.tar' % epoch
        torch.save(state, filename)
#        os.remove('checkpoints/checkpoint-latest.pth.tar')
    except Exception:
        pass
    else:
        if epoch > 2:
            try:
                os.remove('checkpoints/checkpoint-%d.pth.tar' % (epoch - 2))
            except Exception:
                pass


# (Modified) From https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()

        self.kernel_size = 1

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=self.kernel_size)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=self.kernel_size)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=self.kernel_size)
        self.gamma = nn.Parameter(torch.FloatTensor(1).fill_(0.5))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        px = x
#        px = nn.functional.pad(px, (1, 1, 1, 1), mode='reflect')

        proj_query = self.query_conv(px).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(px).view(
            m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(px).view(
            m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


if __name__ == '__main__':
    ex.run_commandline()

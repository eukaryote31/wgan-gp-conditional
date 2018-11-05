import torch
import torch.nn as nn
from external import Self_Attn


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Discriminator(nn.Module):
    def __init__(self, ysize, zsize, isize, discrim_down_blocks, discrim_self_attn,
                 discrim_filters, discrim_resblocks):
        super(Discriminator, self).__init__()
        self.sa = []
        main = nn.Sequential()

        self.ysize = ysize

        main.add_module('init_conv_pad', nn.ReflectionPad2d(1))
        main.add_module('init_conv', nn.utils.spectral_norm(nn.Conv2d(3 + ysize, discrim_filters, 4, 2)))
        main.add_module('init_relu', nn.LeakyReLU(0.2))

        if discrim_self_attn[0]:
            sa = Self_Attn(discrim_filters)
            self.sa.append(sa)
            main.add_module('sa_{}'.format(0), sa)

        down = nn.ModuleList()
        isddb = []
        sidedim = isize // 2
        for i in range(discrim_down_blocks):
            mod = DiscrimDownBlock(discrim_filters, 4, discrim_resblocks, ysize, sidedim)
            isddb.append(True)

            down.append(mod)

            if discrim_self_attn[i + 1]:
                sa = Self_Attn(discrim_filters * 2)
                self.sa.append(sa)
                down.append(sa)
                isddb.append(False)

            discrim_filters *= 2
            sidedim //= 2

        self.main = main
#        self.latmerge = LatMerge(discrim_filters, ysize, sidedim)
        self.fc_size = sidedim ** 2 * discrim_filters
        self.fc = nn.Linear(self.fc_size, 1)
        self.isddb = isddb
        self.down = down

    def forward(self, x, y):
        origy = y
        ysize = self.ysize
        y = y.view(-1, ysize, 1, 1).expand(-1, ysize, x.shape[2], x.shape[3])
        x = torch.cat((x, y), 1)
        x = self.main(x)
        for mod, ddb in zip(self.down, self.isddb):
            if ddb:
                x = mod(x, origy)
            else:
                x = mod(x)
#        x = self.latmerge(x, origy)
        x = x.view(-1, self.fc_size)
        x = self.fc(x)

        return x


class LatMerge(nn.Module):
    def __init__(self, in_channels, lat_channels, sidedim):
        super(LatMerge, self).__init__()
        internal_planes = lat_channels
        self.in_channels = in_channels
        self.lat_channels = lat_channels
        self.internal_planes = internal_planes
        self.sidedim = sidedim
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels + internal_planes, in_channels, 1))
#        self.res = ResBlock(in_channels, False)

    def forward(self, x, lat):
        skip = x
        latp = lat.clone().view(-1, self.internal_planes, 1, 1) \
            .expand(-1, self.internal_planes, self.sidedim, self.sidedim)
        x = torch.cat((x, latp), 1)
        x = self.conv(x)
#        x = self.res(x)
        x += skip
        x = nn.functional.leaky_relu(x, negative_slope=0.05)
        return x


class Generator(nn.Module):
    def __init__(self, isize, ysize, zsize, gen_up_blocks, gen_inp_planes,
                 gen_resblocks, gen_filters, gen_self_attn_up, gen_self_attn_init, gen_init_resblocks):
        super(Generator, self).__init__()
        self.sa = []
        sidedim = isize // (2 ** gen_up_blocks)
        self.sidedim = sidedim
        self.latent = nn.Linear(zsize + ysize, (sidedim ** 2) * gen_inp_planes)
        main = nn.Sequential()
        resb = nn.Sequential()

        main.add_module('pad_init', nn.ReflectionPad2d(1))
        main.add_module('conv_init', nn.utils.spectral_norm(nn.Conv2d(gen_inp_planes, gen_filters, 3)))

        if gen_self_attn_init[0]:
            sa = Self_Attn(gen_filters)
            self.sa.append(sa)
            resb.add_module('sa_{}'.format(0), sa)

        for i in range(gen_init_resblocks):
            resb.add_module('res_{}'.format(i), ResBlock(gen_filters, True))

            if gen_self_attn_init[i + 1]:
                sa = Self_Attn(gen_filters)
                self.sa.append(sa)
                resb.add_module('sa_{}'.format(i + 1), sa)

        resb.add_module('relu', nn.LeakyReLU(0.05))
        main.add_module('init_res', SkipWrapper(resb))

        if gen_self_attn_up[0]:
            sa = Self_Attn(gen_filters)
            self.sa.append(sa)
            main.add_module('sa_{}'.format(0), sa)

        up = nn.ModuleList()
        isgenup = []
        for i in range(gen_up_blocks):
            mod = GenUpBlock(gen_filters, gen_filters // 2, gen_resblocks, ysize + zsize, sidedim)

            up.append(mod)
            isgenup.append(True)

            gen_filters //= 2
            sidedim *= 2

            if gen_self_attn_up[i + 1]:
                sa = Self_Attn(gen_filters)
                self.sa.append(sa)
                up.append(sa)
                isgenup.append(False)


#        self.latmerge = LatMerge(gen_filters, zsize + ysize, isize)
        final = nn.Sequential()
#        final.add_module('res', ResBlock(gen_filters, True))
        final.add_module('padding', nn.ReflectionPad2d(4))
        final.add_module('conv_final', nn.utils.spectral_norm(nn.Conv2d(gen_filters, 3, 9)))
        final.add_module('sigmoid', nn.Sigmoid())
        self.gen_inp_planes = gen_inp_planes
        self.main = main
        self.up = up
        self.final = final
        self.isgenup = isgenup

    def forward(self, z, y):
        oz = z
        z = self.latent(torch.cat((z, y), 1))
        z = z.view(-1, self.gen_inp_planes, self.sidedim, self.sidedim)
        z = self.main(z)
        for up, isgenup in zip(self.up, self.isgenup):
            if isgenup:
                z = up(z, torch.cat((oz, y), 1))
            else:
                z = up(z)
#        z = self.latmerge(z, torch.cat((oz, y), 1))
        z = self.final(z)

        return z


class SkipWrapper(nn.Module):
    def __init__(self, mod):
        super(SkipWrapper, self).__init__()

        self.mod = mod

    def forward(self, x):
        skip = x
        y = self.mod(x)
        y += skip
        return y


class ResBlock(nn.Module):
    def __init__(self, channels, bn):
        super(ResBlock, self).__init__()

        self.bn = bn

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3))
        if bn:
            self.bn1 = nn.BatchNorm2d(channels)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3))
        if bn:
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.pad1(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = nn.functional.leaky_relu(out, negative_slope=0.2)
        out = self.pad2(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        out += residual
        out = nn.functional.leaky_relu(out, negative_slope=0.2)
        return out


class GenUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resblocks, yzsize, sidedim):
        super(GenUpBlock, self).__init__()

        self.latmerge = LatMerge(in_channels, yzsize, sidedim)
        self.res = nn.Sequential()
        for i in range(resblocks):
            self.res.add_module('res_{}'.format(i), ResBlock(in_channels, True))
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels * 4, 3))
        self.shuf = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.01)


    def forward(self, x, yz):
        skip = x
        x = self.latmerge(x, yz)
        x = self.res(x)
        x += skip
        x = self.pad(x)
        x = self.conv(x)
        x = self.shuf(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DiscrimDownBlock(nn.Module):
    def __init__(self, in_channels, kern, nblocks, yzsize, sidedim):
        super(DiscrimDownBlock, self).__init__()
        main = nn.Sequential()

        for i in range(nblocks):
            main.add_module('discrim_block_{}'.format(i), ResBlock(in_channels, False))
        if kern == 4:
            main.add_module('discrim_conv_pad',
                            nn.ReflectionPad2d((1, 2, 1, 2)))
        else:
            main.add_module('discrim_conv_pad', nn.ReflectionPad2d(1))

        main.add_module('discrim_conv', nn.utils.spectral_norm(nn.Conv2d(
            in_channels, in_channels * 2, kern, 2)))
        main.add_module('discrim_relu', nn.LeakyReLU(0.2))
        self.latmerge = LatMerge(in_channels, yzsize, sidedim)
        self.main = main

    def forward(self, x, yz):
        x = self.latmerge(x, yz)
        return self.main(x)

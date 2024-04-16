import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class UpSampling(nn.Module):
    def __init__(self, in_channels, with_conv) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

class DownSampling(nn.Module):
    def __init__(self, in_channels, with_conv) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        # out = (input + 2 * padding - kernel_size) / stride + 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if temb_channels:
            self.temb_proj = nn.Linear(
                temb_channels,
                out_channels,
            )

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if out_channels != in_channels:
            if self.use_conv_shortcut:
                self.conv3 = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.conv3 = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb):
        h = x
        x = self.norm1(x)
        x = x * torch.sigmoid(x)
        x = self.conv1(x)

        if temb is not None:
            x += self.temb_proj(temb * torch.sigmoid(temb))[:, :, None, None]
        
        x = self.norm2(x)
        x = x * torch.sigmoid(x)
        x = self.dropout(x)
        x = self.conv2(x)

        if self.in_channels != self.out_channels:
            h = self.conv3(h)

        return x + h
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.proj_out = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        h = x
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b c (h w)')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij, bik->bjk', q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        x = torch.einsum('bik,bjk->bji', w_, v)
        x = rearrange(x, 'b c (h w)-> b c h w', h=h, w=w)

        x = self.proj_out(x)

        return x + h
    
class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, n_heads) -> None:
        super().__init__()
        assert in_channels % n_heads == 0
        self.n_heads = n_heads
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.proj_out = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        h_ = x
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b (c heads) h w -> b heads (h w) c', heads=self.n_heads)
        k = rearrange(k, 'b (c heads) h w -> b heads c (h w)', heads=self.n_heads)
        w_ = torch.einsum('bhij, bhjk->bhik', q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b (c heads) h w -> b heads c (h w)', heads=self.n_heads)
        x = torch.einsum('bhik,bhjk->bhji', w_, v)
        x = rearrange(x, 'b heads c (h w)-> b (c heads) h w', h=h, w=w)

        x = self.proj_out(x)

        return x + h_
    
class LinearAttnBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, n_heads) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.conv1 = nn.Conv2d(
            in_channels,
            hidden_size * 3,
            kernel_size=3, 
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            hidden_size,
            in_channels,
            kernel_size=3, 
            stride=1,
            padding=1
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.conv1(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', qkv=3, heads=self.n_heads)
        k = k.softmax(dim=-1)
        weights = torch.einsum('bhdn, bhen->bhde', q, k)
        out = torch.einsum('bhde,bhdn->bhen', weights, v)
        out = rearrange(out, 'b heads c (h w)->b (heads c) h w', heads=self.n_heads, h=h, w=w)
        return self.conv2(out)
    
def make_attn(in_channels, attn_type='vanilla', hidden_size=1024, n_heads=8):
    assert attn_type in ['vanilla', 'multihead', 'linear', 'None']
    if attn_type == 'vanilla':
        return AttnBlock(in_channels)
    elif attn_type == 'multihead' and n_heads is not None:
        return MultiHeadAttnBlock(in_channels, n_heads)
    elif attn_type == 'linear':
        return LinearAttnBlock(in_channels, hidden_size, n_heads)
    else:
        return nn.Identity(in_channels)
    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type='vanilla',
                 **ignore_kwargs) -> None:
        super().__init__()
        if use_linear_attn: attn_type = 'linear'
        self.ch = ch
        self.temb_ch = 0
        self.num_resolution = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            in_channels,
            ch,
            kernel_size=
            3,
            stride=1,
            padding=1,
        )
        curr_res = resolution
        in_ch_multi = (1,) + tuple(ch_mult)
        self.in_ch_multi = in_ch_multi
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolution):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_multi[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolution - 1:
                down.downsample = DownSampling(block_in, resamp_with_conv)
            self.down.append(down)
        
        # mid
        self.mid = nn.Module()
        self.mid.block1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout
        )

        # end
        self.norm = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        temb = None

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolution):
            for i_blcok in range(self.num_res_blocks):
                h = self.down[i_level].block[i_blcok](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_blcok](h)
                hs.append(h)
            if i_level != self.num_resolution - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        # middle
        h = hs[-1]
        h = self.mid.block1(h, temb)
        h = self.mid.attn1(h)
        h = self.mid.block2(h, temb)

        # end
        h = self.norm(h)
        h = h * torch.sigmoid(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type='vanilla', **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpSampling(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = h * torch.sigmoid(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class AutoEncoderKL(nn.Module):
    def __init__(self,
                embed_dim,
                double_z=True, 
                z_channels=16,
                resolution=256,
                in_channels=3,
                out_ch=3,
                ch=128,
                ch_mult=[ 1,1,2,2,4],  # num_down = len(ch_mult)-1
                num_res_blocks=2,
                attn_resolutions=[16],
                dropout=0.0,
                attn_type='multihead',
                ) -> None:
        super().__init__()
        self.encoder = Encoder(
                double_z=double_z, 
                z_channels=z_channels,
                resolution=resolution,
                in_channels=in_channels,
                out_ch=out_ch,
                ch=ch,
                ch_mult=ch_mult,  # num_down = len(ch_mult)-1
                num_res_blocks=num_res_blocks,
                attn_resolutions=attn_resolutions,
                dropout=dropout,
                attn_type=attn_type
            )
        self.decoder = Decoder(
                double_z=double_z, 
                z_channels=z_channels,
                resolution=resolution,
                in_channels=in_channels,
                out_ch=out_ch,
                ch=ch,
                ch_mult=ch_mult,  # num_down = len(ch_mult)-1
                num_res_blocks=num_res_blocks,
                attn_resolutions=attn_resolutions,
                dropout=dropout,
                attn_type=attn_type
            )
        assert double_z
        self.quant_conv = nn.Conv2d(2*z_channels, 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

model = AutoEncoderKL(128)
x = torch.randn(4,3,256,256)
y, _ = model(x)
print(y.shape)

# double_z: True
# z_channels: 16
# resolution: 256
# in_channels: 3
# out_ch: 3
# ch: 128
# ch_mult: [ 1,1,2,2,4]  # num_down = len(ch_mult)-1
# num_res_blocks: 2
# attn_resolutions: [16]
# dropout: 0.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init

from kornia.filters import *

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100 * (1 - super(SSIM_Loss, self).forward(img1, img2))

# from taming.modules.losses.vgg import Vgg16
# def content_loss_vgg( inputs, reconstructions):
#         dtype = torch.cuda.FloatTensor
#         # load vgg network
#         vgg = Vgg16().type(dtype)
#         loss_mse = torch.nn.MSELoss()
#         CONTENT_WEIGHT = 1e0
#
#
#         # get vgg features
#         y_c_features = vgg(inputs)
#         y_hat_features = vgg(reconstructions)
#         # calculate content loss (h_relu_2_2)
#         recon = y_c_features[1]
#         recon_hat = y_hat_features[1]
#         return CONTENT_WEIGHT * loss_mse(recon_hat, recon)


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        # self.ssim_criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=3)
        # self.ms_ssim_criterion = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)



    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight

        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        # reconstruction loss
        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            ## Gradient loss component
            GDL_loss = self.gradient_loss(reconstructions,inputs.detach())
            ## Sobel loss component
            sobel_loss = self.sobel_edge(inputs.detach(),reconstructions)
            ## SSim loss component
            # ssim_loss = self.ssim_criterion(reconstructions, inputs.detach())
            ## Content loss component << never tried
            # content_loss = content_loss_vgg(inputs.detach(),reconstructions)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + GDL_loss
                        # + GDL_loss
                        # + sobel_loss
                        # + 10*GDL_loss
                        # + content_loss
                        # + ssim_loss
                        # + self.codebook_weight * codebook_loss.mean() <<original from VQGAN

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log



    def sobel_edge(self,inputs,reconstructions):
        sobel_loss_input = sobel(inputs)
        sobel_loss_reconstruction = sobel(reconstructions)

        return torch.mean((sobel_loss_input-sobel_loss_reconstruction).pow(2))

    def gradient_loss(self,gen_frames, gt_frames, alpha=1):

        def gradient(x):
            # idea from tf.image.image_gradients(image)
            # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
            # x: (b,c,h,w), float32 or float64
            # dx, dy: (b,c,h,w)

            h_x = x.size()[-2]
            w_x = x.size()[-1]
            # gradient step=1
            left = x
            right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
            top = x
            bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

            # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
            dx, dy = right - left, bottom - top
            # dx will always have zeros in the last column, right-left
            # dy will always have zeros in the last row,    bottom-top
            dx[:, :, :, -1] = 0
            dy[:, :, -1, :] = 0

            return dx, dy

        # gradient
        gen_dx, gen_dy = gradient(gen_frames)
        gt_dx, gt_dy = gradient(gt_frames)
        #
        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        # condense into one tensor and avg
        return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

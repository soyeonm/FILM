from typing import Dict, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from . import segmentation_definitions as segdef


DEPTH_BINS = 50
DEPTH_MAX = 5.0

DISTR_DEPTH = True
VEC_HEAD = True

TRAIN_DEPTH_ONLY = False
TRAIN_SEG_ONLY = True
TRAIN_INV_ONLY = False


def _multidim_sm(x: torch.tensor, dims: Tuple[int, ...], log: bool):
    #_check_dims_consecutive(dims)
    init_shape = x.shape
    dims = list(sorted(dims))
    new_shape = [d for i, d in enumerate(init_shape) if i not in dims]
    new_shape = new_shape[:dims[0]] + [-1] + new_shape[dims[0]:]
    x = x.reshape(new_shape)
    dim = dims[0]

    if log:
        x = F.log_softmax(x, dim=dim)
    else:
        x = F.softmax(x, dim=dim)

    x = x.reshape(init_shape)
    return x

def multidim_softmax(x: torch.tensor, dims: Tuple[int, ...]) -> torch.tensor:
    return _multidim_sm(x, dims, log=False)


def multidim_logsoftmax(x: torch.tensor, dims: Tuple[int, ...]) -> torch.tensor:
    return _multidim_sm(x, dims, log=True)


class DepthEstimate():

    def __init__(self, depth_pred, num_bins, max_depth):
        # depth_pred: BxCxHxW tensor of depth probabilities over C depth bins
        # Convert logprobabilities to probabilities
        if depth_pred.max() < 0:
            depth_pred = torch.exp(depth_pred)

        if ((depth_pred.sum(dim=1) - 1).abs() < 1e-2).all():
            pass
        else:
            pickle.dump(depth_pred, open("depth_pred.p", "wb"))
            assert ((depth_pred.sum(dim=1) - 1).abs() < 1e-2).all(), "Depth prediction needs to be a simplex at each pixel, current sum is " + str(depth_pred.sum(dim=1))

        self.depth_pred = depth_pred
        self.num_bins = num_bins
        self.max_depth = max_depth

    def to(self, device):
        depth_pred = self.depth_pred.to(device)
        return DepthEstimate(depth_pred, self.num_bins, self.max_depth)

    def domain(self, res=None):
        if res is None:
            res = self.num_bins
        return torch.arange(0, res, 1, device=self.depth_pred.device)[None, :, None, None]

    def domain_image(self, res=None):
        if res is None:
            res = self.num_bins
        domain = self.domain(res)
        domain_image = domain.repeat((1, 1, self.depth_pred.shape[2], self.depth_pred.shape[3])) * (self.max_depth / res)
        return domain_image

    def mle(self):
        mle_depth = self.depth_pred.argmax(dim=1, keepdim=True).float() * (self.max_depth / self.num_bins)
        return mle_depth

    def expectation(self):
        expected_depth = (self.domain() * self.depth_pred).sum(dim=1, keepdims=True) * (self.max_depth / self.num_bins)
        return expected_depth

    def spread(self):
        spread = (self.mle() - self.expectation()).abs()
        return spread

    def percentile(self, which):
        domain = self.domain()
        cumsum = self.depth_pred.cumsum(dim=1)
        pctlbin = ((cumsum < which) * domain).max(dim=1, keepdim=True).values
        pctldepth = pctlbin * (self.max_depth / self.num_bins)
        return pctldepth

    def get_trustworthy_depth(self, include_mask=None, confidence=0.9, max_conf_int_width_prop=0.30, include_mask_prop=1.0):
        conf_int_lower = self.percentile((1 - confidence) / 2)
        conf_int_upper = self.percentile(1 - (1 - confidence) / 2)

        spread = conf_int_upper - conf_int_lower
        max_conf_int_width = self.expectation() * max_conf_int_width_prop
        trusted_mask = spread < max_conf_int_width

        accept_mask = trusted_mask * (1-include_mask.bool().float()).bool()

        if include_mask is not None:
            # Apply looser criteria for objects that the agent is actively looking for
            include_mask_criteria = spread < include_mask_prop * self.expectation()
            include_mask_solid = include_mask_criteria * include_mask
            accept_mask = accept_mask.bool() + include_mask_solid.bool()

        est_depth = self.mle()
        trustworthy_depth = est_depth * accept_mask
        return trustworthy_depth


class DoubleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, cmid=None, stride2=1):
        super(DoubleConv, self).__init__()
        if cmid is None:
            cmid = cin
        self.conv1 = nn.Conv2d(cin, cmid, k, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(cmid, cout, k, stride=stride2, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img):
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        return x


class UpscaleDoubleConv(torch.nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super(UpscaleDoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(cin, cout, k, stride=1, padding=padding)
        #self.upsample1 = Upsample(scale_factor=2, mode="nearest")
        self.conv2 = nn.Conv2d(cout, cout, k, stride=1, padding=padding)

    def init_weights(self):
        torch.nn.init.kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        torch.nn.init.kaiming_uniform(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, img, output_size):
        x = self.conv1(img)
        x = F.leaky_relu(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2(x)
        if x.shape[2] > output_size[2]:
            x = x[:, :, :output_size[2], :]
        if x.shape[3] > output_size[3]:
            x = x[:, :, :, :output_size[3]]
        return x


class SimpleUNEt(torch.nn.Module):
    def __init__(self, distr_depth, vec_head, segonly):
        super(SimpleUNEt, self).__init__()

        self.num_c = segdef.get_num_objects()
        self.depth_bins = 50

        class objectview(object):
            def __init__(self, d):
                self.__dict__ = d
        params = {
            "in_channels": 3,
            "hc1": 256 if distr_depth else 256,
            "hc2": 256 if distr_depth else 256,
            "out_channels": self.num_c + DEPTH_BINS if distr_depth else self.num_c + 1,
            "out_vec_length": self.num_c + 1,
            "stride": 2
        }

        self.p = objectview(params)
        self.distr_depth = distr_depth
        self.vec_head = vec_head

        DeconvOp = UpscaleDoubleConv
        ConvOp = DoubleConv

        # inchannels, outchannels, kernel size
        self.conv1 = ConvOp(self.p.in_channels, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv2 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv3 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv4 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv5 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.conv6 = ConvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)

        self.deconv1 = DeconvOp(self.p.hc1, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv2 = DeconvOp(self.p.hc1 * 2, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv3 = DeconvOp(self.p.hc1 * 2, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv4 = DeconvOp(self.p.hc1 * 2, self.p.hc1, 3, stride=self.p.stride, padding=1)
        self.deconv5 = DeconvOp(self.p.hc1 * 2, self.p.hc2, 3, stride=self.p.stride, padding=1)
        self.deconv6 = DeconvOp(self.p.hc1 + self.p.hc2, self.p.out_channels, 3, stride=self.p.stride, padding=1)

        if self.vec_head:
            self.linear1 = nn.Linear(self.p.hc1, self.p.hc1)
            self.linear2 = nn.Linear(self.p.hc1, self.p.out_vec_length)

        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.norm2 = nn.InstanceNorm2d(self.p.hc1)
        self.norm3 = nn.InstanceNorm2d(self.p.hc1)
        self.norm4 = nn.InstanceNorm2d(self.p.hc1)
        self.norm5 = nn.InstanceNorm2d(self.p.hc1)
        self.norm6 = nn.InstanceNorm2d(self.p.hc1)
        # self.dnorm1 = nn.InstanceNorm2d(in_channels * 4)
        self.dnorm2 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm3 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm4 = nn.InstanceNorm2d(self.p.hc1)
        self.dnorm5 = nn.InstanceNorm2d(self.p.hc2)

    def init_weights(self):
        self.conv1.init_weights()
        self.conv2.init_weights()
        self.conv3.init_weights()
        self.conv4.init_weights()
        self.conv5.init_weights()
        self.deconv1.init_weights()
        self.deconv2.init_weights()
        self.deconv3.init_weights()
        self.deconv4.init_weights()
        #self.deconv5.init_weights()

    def forward(self, input):
        x1 = self.norm2(self.act(self.conv1(input)))
        x2 = self.norm3(self.act(self.conv2(x1)))
        x3 = self.norm4(self.act(self.conv3(x2)))

        x3 = self.dropout(x3)

        x4 = self.norm5(self.act(self.conv4(x3)))
        x5 = self.norm6(self.act(self.conv5(x4)))
        x6 = self.act(self.conv6(x5))

        x6 = self.dropout(x6)

        y5 = self.act(self.deconv1(x6, output_size=x5.size()))
        xy5 = torch.cat([x5, y5], 1)

        y4 = self.dnorm3(self.act(self.deconv2(xy5, output_size=x4.size())))
        xy4 = torch.cat([x4, y4], 1)
        y3 = self.dnorm4(self.act(self.deconv3(xy4, output_size=x3.size())))
        xy3 = torch.cat([x3, y3], 1)
        y2 = self.dnorm4(self.act(self.deconv4(xy3, output_size=x2.size())))
        xy2 = torch.cat([x2, y2], 1)

        xy2 = self.dropout(xy2)

        y1 = self.dnorm5(self.act(self.deconv5(xy2, output_size=x1.size())))
        xy1 = torch.cat([x1, y1], 1)
        out = self.deconv6(xy1, output_size=input.size())

        out_a = out[:, :self.num_c]
        out_b = out[:, self.num_c:]

        out_a = multidim_logsoftmax(out_a, dims=(1,))

        if self.distr_depth:
            out_b = multidim_logsoftmax(out_b, dims=(1,))

        if self.vec_head:
            vec_head = self.linear2(self.act(self.linear1(x6.mean(dim=2).mean(dim=2))))
            vec_head = F.log_softmax(vec_head, dim=1)
        else:
            vec_head = None

        return out_a, out_b, vec_head


class AlfredSegmentationAndDepthModel(nn.Module):

    """
    Given a current state s_t, proposes an action distribution that makes sense.
    """
    def __init__(self, hparams = None, distr_depth=DISTR_DEPTH, vec_head=VEC_HEAD, segonly=False):
        super().__init__()
        self.hidden_dim = 128
        self.semantic_channels = segdef.get_num_objects()
        self.distr_depth = distr_depth
        self.vec_head = vec_head

        self.net = SimpleUNEt(distr_depth, vec_head, segonly)

        self.iter = nn.Parameter(torch.zeros([1], dtype=torch.double), requires_grad=False)

        self.nllloss = nn.NLLLoss(reduce=True, size_average=True)
        self.celoss = nn.CrossEntropyLoss(reduce=True, size_average=True)
        self.mseloss = nn.MSELoss(reduce=True, size_average=True)
        self.act = nn.LeakyReLU()

    def predict(self, rgb_image):
        with torch.no_grad():
            if self.distr_depth:
                DEPTH_TEMPERATURE_BETA = 0.5# 0.5 # 1.0# 0.3
                SEG_TEMPERATURE_BETA = 1.0 # 1.5

                seg_pred, depth_pred, vec_head = self.forward_model(rgb_image)
                seg_pred = torch.exp(seg_pred * SEG_TEMPERATURE_BETA)
                depth_pred = torch.exp(depth_pred * DEPTH_TEMPERATURE_BETA)
                depth_pred = depth_pred / (depth_pred.sum(dim=1, keepdim=True))

                depth_pred = DepthEstimate(depth_pred, DEPTH_BINS, DEPTH_MAX)

                # Filter segmentations
                good_seg_mask = seg_pred > 0.3
                seg_pred = seg_pred * good_seg_mask
                seg_pred = seg_pred / (seg_pred.sum(dim=1, keepdims=True) + 1e-10)
            else:
                seg_pred, depth_pred, vec_head = self.forward_model(rgb_image)
                seg_pred = torch.exp(seg_pred)

                good_seg_mask = seg_pred > 0.3
                good_depth_mask = (seg_pred > 0.5).sum(dim=1, keepdims=True) * (depth_pred > 0.9)
                seg_pred = seg_pred * good_seg_mask
                seg_pred = seg_pred / (seg_pred.sum(dim=1, keepdims=True) + 1e-10)
                depth_pred = depth_pred * good_depth_mask

        return seg_pred, depth_pred

    def forward_model(self, rgb_image: torch.tensor):
        return self.net(rgb_image)

    def get_name(self) -> str:
        return "alfred_segmentation_and_depth_model"

    def loss(self, batch: Dict):
        # This is now forward
        return self.forward(batch)

    def forward(self, batch: Dict):
        observations = batch["observations"]

        rgb_image = observations.rgb_image.float()
        seg_gt = observations.semantic_image.float().clone()
        depth_gt = observations.depth_image.float()
        inv_gt = observations.inventory_vector.float()
        inv_gt_int = (inv_gt.argmax(dim=1) + 1) * (inv_gt.max(1).values > 0.5)

        # Switch to a one-hot segmentation representation
        observations.uncompress()
        seg_gt_oh = observations.semantic_image.float()

        b, c, h, w = seg_gt.shape

        seg_pred, depth_pred, vec_head = self.forward_model(rgb_image)

        c = seg_pred.shape[1]

        seg_flat_pred = seg_pred.permute((0, 2, 3, 1)).reshape([b * h * w, c])
        seg_flat_gt = seg_gt.permute((0, 2, 3, 1)).reshape([b * h * w]).long()

        seg_loss = self.nllloss(seg_flat_pred, seg_flat_gt)

        if VEC_HEAD:
            inv_loss = self.nllloss(vec_head, inv_gt_int)

        if self.distr_depth:
            depth_flat_pred = depth_pred.permute((0, 2, 3, 1)).reshape([b * h * w, DEPTH_BINS])
            depth_flat_gt = depth_gt.permute((0, 2, 3, 1)).reshape([b * h * w])
            depth_flat_gt = ((depth_flat_gt / DEPTH_MAX).clamp(0, 0.999) * DEPTH_BINS).long()
            depth_loss = self.nllloss(depth_flat_pred, depth_flat_gt)

            depth_pred_mean = (torch.arange(0, DEPTH_BINS, 1, device=depth_pred.device)[None, :, None, None] * torch.exp(depth_pred)).sum(dim=1)
            depth_mae = (depth_pred_mean.view([-1]) - depth_flat_gt).abs().float().mean() * (DEPTH_MAX / DEPTH_BINS)

        else:
            depth_flat_pred = depth_pred.reshape([b, h * w])
            depth_flat_gt = depth_gt.reshape([b, h * w])
            depth_loss = self.mseloss(depth_flat_pred, depth_flat_gt)
            depth_mae = (depth_flat_pred - depth_flat_gt).abs().mean()

        seg_pred_distr = torch.exp(seg_pred)

        if TRAIN_DEPTH_ONLY:
            loss = depth_loss
        elif TRAIN_SEG_ONLY:
            loss = seg_loss
        else:
            loss = seg_loss + depth_loss

        if VEC_HEAD:
            loss = loss + inv_loss

        metrics = {}
        metrics["loss"] = loss.item()
        metrics["seg_loss"] = seg_loss.item()
        metrics["depth_loss"] = depth_loss.item()
        metrics["depth_mae"] = depth_mae.item()
        if VEC_HEAD:
            metrics["inv_loss"] = inv_loss.item()

        self.iter += 1

        return loss, metrics

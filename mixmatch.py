import sys
from itertools import cycle
import logging

import numpy as np
from tqdm import tqdm
import albumentations as A

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from utils import (NUM_CLASSES, base_tf, collate_fn, get_datasets,
                   apply_transform, validate)
from ema import ModelEMA

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger()

aug_tf = A.Compose([
    A.ShiftScaleRotate(),
    A.OneOf([
        A.CLAHE(),
        A.Solarize(),
        A.ColorJitter(),
        A.ToGray(),
        A.ToSepia(),
        A.RandomBrightness(),
        A.RandomGamma(),
    ]),
    A.CoarseDropout(max_height=4, max_width=4, max_holes=3, p=0.25), base_tf
])


def train_epoch_mixmatch(epoch,
                         model,
                         ema_model,
                         train_dl_l,
                         train_dl_ul,
                         optimizer,
                         _get_w,
                         num_augs,
                         T,
                         α,
                         bar_label_prefix=''):

    torch.cuda.reset_peak_memory_stats()

    epoch_loss = 0.

    bar_label = f'{bar_label_prefix}{"Training":10s}'
    model.train()
    with tqdm(
            zip(cycle(train_dl_l), train_dl_ul),
            desc=bar_label,
            total=len(train_dl_ul)) as bar:
        for i, ((x_l, y_l), (x_ul, _)) in enumerate(bar):

            w_u = _get_w(epoch * len(train_dl_ul) + i)

            # --------------
            # create pseudo labels
            # --------------

            # (num_augs * batch_size, C, H, W)
            x_ul_aug = torch.cat(
                [apply_transform([_x, _x], None, tf=aug_tf)[0] for _x in x_ul],
                dim=0)
            x_ul_aug = x_ul_aug.cuda()

            with torch.no_grad():
                model.eval()
                # (batch_size * num_augs, num_classes)
                out_ul_aug = model(x_ul_aug).detach()
                model.train()

            # (batch_size, num_augs, num_classes)
            out_ul_aug = out_ul_aug.reshape(-1, num_augs, out_ul_aug.shape[-1])
            # (batch_size, num_classes)
            out_ul_aug = out_ul_aug.mean(dim=1).softmax(dim=-1)
            out_ul_aug_sharp = sharpen(out_ul_aug, T=T, dim=-1)
            out_ul_aug_sharp = out_ul_aug_sharp.cuda()
            y_ul = out_ul_aug_sharp

            # --------------
            # mixup and forward pass
            # --------------

            x_ul, _ = apply_transform(x_ul, None, tf=aug_tf)
            x_ul = x_ul.cuda()

            x_l, y_l = apply_transform(x_l, y_l, tf=base_tf)
            x_l = x_l.cuda()
            x = torch.cat([x_l, x_ul], dim=0).cuda()

            inds = torch.randperm(len(x)).cuda()

            x = x[inds]

            L, λ_l_1, λ_l_2 = mixup(α, α, x_l, x[:len(x_l)])
            U, λ_ul_1, λ_ul_2 = mixup(α, α, x_ul, x[len(x_l):])

            out_l = model(L)
            out_ul = model(U).softmax(dim=-1)

            # --------------
            # loss and backward pass
            # --------------

            y_l = y_l.cuda()
            y_l_new = torch.zeros_like(out_l)
            y_l_new[torch.arange(len(y_l)), y_l] = 1.

            y = torch.cat([y_l_new, y_ul], dim=0).contiguous()
            y = y[inds]

            y_l_1 = y_l_new
            y_l_2 = y[:len(x_l)]

            y_ul_1 = y_ul
            y_ul_2 = y[len(x_l):]

            y_sup = λ_l_1.unsqueeze(-1) * y_l_1 + λ_l_2.unsqueeze(-1) * y_l_2
            supervised_loss = -torch.log_softmax(out_l, dim=-1)
            supervised_loss *= y_sup
            supervised_loss = supervised_loss.sum(dim=-1).mean()

            y_unsup = λ_ul_1.unsqueeze(-1) * y_ul_1 + λ_ul_2.unsqueeze(
                -1) * y_ul_2
            unsupervised_loss = F.mse_loss(out_ul, y_unsup)

            loss = supervised_loss + w_u * unsupervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_model.update(model)

            epoch_loss += loss.detach().item()
            bar.set_postfix({
                'w': f'{w_u:.03f}',
                'sup_loss': supervised_loss.item(),
                'unsup_loss': unsupervised_loss.item(),
            })

    return epoch_loss, -1


def train_mixmatch(model,
                   ema_model,
                   train_dl_l,
                   train_dl_ul,
                   val_dl,
                   optimizer,
                   sched,
                   params,
                   num_augs,
                   T,
                   α,
                   w_scale,
                   rampup_epochs=80,
                   rampdown_epochs=50,
                   start_epoch=0,
                   eval_interval=1):
    num_epochs = params['epochs']

    def _get_w(step):
        N = len(train_dl_ul)
        rampdown_start = (num_epochs - rampdown_epochs) * N
        rampup_steps = rampup_epochs * N
        rampdown_steps = rampdown_epochs * N
        if step <= rampup_steps:
            # ramp up
            w = w_scale * np.exp(-5 * (1 - (step / rampup_steps))**2)
        elif step >= rampdown_start:
            # ramp down
            w = w_scale * np.exp(-((
                (step - rampdown_start) * 0.5)**2) / rampdown_steps)
        else:
            w = w_scale
        return w

    for epoch in range(start_epoch, num_epochs):

        bar_label_prefix = f'Epoch {epoch+1}/{num_epochs}: '

        # train
        epoch_loss, train_acc = train_epoch_mixmatch(
            epoch,
            model,
            ema_model,
            train_dl_l,
            train_dl_ul,
            optimizer,
            _get_w,
            num_augs,
            T,
            α,
            bar_label_prefix=bar_label_prefix)

        if (epoch + 1) % eval_interval == 0:
            # validate
            val_loss, val_acc = validate(
                ema_model,
                nn.CrossEntropyLoss(),
                val_dl,
                bar_label_prefix=bar_label_prefix)

            lr = optimizer.param_groups[0]['lr']
            print(f'\nepoch: {epoch+1:3d}, lr: {lr:0.6f}, '
                  f'epoch_loss: {epoch_loss:4.4f}, val_loss: {val_loss:4.4f}, '
                  f'train_acc: {train_acc:0.4f}, val_acc: {val_acc:0.4f}')

        sched.step()


def mixup(α, β, x1, x2):
    assert len(x1) == len(x2)
    N = len(x1)
    λ = torch.distributions.beta.Beta(α, β).sample(sample_shape=(N, 1, 1,
                                                                 1)).cuda()
    λ1, λ2 = λ, (1 - λ)
    mixed_up = (λ1 * x1) + (λ2 * x2)
    return mixed_up, λ1.flatten(), λ2.flatten()


def sharpen(x, T, dim=-1):
    x = x**(1 / T)
    return x / x.sum(dim=dim, keepdims=True)


def run():
    model = torch.hub.load(
        'AdeelH/WideResNet-pytorch:torch_hub',
        'WideResNet',
        depth=28,
        num_classes=NUM_CLASSES,
        widen_factor=2)
    model = model.cuda()
    ema_model = ModelEMA(model, decay=0.999)

    train_params = {}
    train_params['batch_size'] = 100
    train_params['val_batch_size'] = 256

    train_ds, train_subset_ds, val_ds = get_datasets(subset_size=4000)
    train_dl_l = torch.utils.data.DataLoader(
        train_subset_ds,
        batch_size=train_params['batch_size'],
        pin_memory=True,
        num_workers=4,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn)
    train_dl_ul = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_params['batch_size'],
        pin_memory=True,
        num_workers=4,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn)
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=train_params['val_batch_size'],
        pin_memory=True,
        num_workers=2,
        drop_last=False,
        collate_fn=collate_fn)

    train_params['epochs'] = 300
    train_params['learning_rate'] = 3e-4

    optimizer = optim.Adam(
        model.parameters(),
        lr=train_params['learning_rate'],
        betas=(0.9, 0.999))
    sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        train_params['epochs'],
        eta_min=train_params['learning_rate'] / 10)

    train_mixmatch(
        model,
        ema_model,
        train_dl_l,
        train_dl_ul,
        val_dl,
        optimizer,
        sched,
        train_params,
        num_augs=2,
        T=0.5,
        α=0.75,
        w_scale=100,
        rampup_epochs=int(train_params['epochs'] * .4),
        rampdown_epochs=int(train_params['epochs'] / 6),
        start_epoch=0)


if __name__ == '__main__':
    run()

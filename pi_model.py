import numpy as np
from tqdm import tqdm
import albumentations as A

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from utils import (NUM_CLASSES, base_tf, collate_fn, get_datasets,
                   apply_transform, validate)

aug_tf = A.Compose([
    A.Compose([A.RandomCrop(30, 30, p=0.5),
               A.Resize(32, 32)]),
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


def train_cl_epoch(epoch,
                   model,
                   train_dl,
                   criterion,
                   optimizer,
                   _get_w,
                   bar_label_prefix=''):

    torch.cuda.reset_peak_memory_stats()

    epoch_loss = 0.
    train_corrects = 0

    model.train()
    bar_label = f'{bar_label_prefix}{"Training":10s}'
    with tqdm(train_dl, desc=bar_label, total=len(train_dl)) as bar:
        for i, (x, y) in enumerate(bar):

            w = _get_w(epoch * len(train_dl) + i)

            x_aug, _ = apply_transform(x, None, tf=aug_tf)
            x_aug = x_aug.cuda()

            x, y = apply_transform(x, y, tf=base_tf)
            x = x.cuda()
            y = y.cuda()

            x_all = torch.cat((x, x_aug), dim=0)

            out = model(x_all)
            out_x = out[:len(x)]
            out_aug = out[len(x):]

            supervised_loss = criterion(out_x, y)

            consistency_loss = F.mse_loss(
                out_x.detach().softmax(dim=-1), out_aug.softmax(dim=-1))
            loss = supervised_loss + w * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
            preds = out_x.detach().argmax(dim=-1)
            train_corrects += (preds == y).cpu().float().sum()

            bar.set_postfix({
                'w': f'{w:.03f}',
                'sup_loss': f'{supervised_loss.item():.03f}',
                'unsup_loss': f'{consistency_loss.item():.03f}',
            })

    train_acc = train_corrects / len(train_dl.sampler)
    return epoch_loss, train_acc


def train_cl(model,
             train_dl,
             val_dl,
             optimizer,
             sched,
             params,
             w_scale=1,
             rampup_epochs=80,
             rampdown_epochs=50,
             eval_interval=1):

    num_epochs = params['epochs']

    def _get_w(step):
        N = len(train_dl)
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

    for epoch in range(num_epochs):

        bar_label_prefix = f'Epoch {epoch+1}/{num_epochs}: '

        epoch_loss, train_acc = train_cl_epoch(
            epoch,
            model,
            train_dl,
            nn.CrossEntropyLoss(),
            optimizer,
            _get_w,
            bar_label_prefix=bar_label_prefix)

        if (epoch + 1) % eval_interval == 0:
            # validate
            val_loss, val_acc = validate(
                model,
                nn.CrossEntropyLoss(),
                val_dl,
                bar_label_prefix=bar_label_prefix)

            lr = optimizer.param_groups[0]['lr']
            print(f'\nepoch: {epoch+1:3d}, lr: {lr:0.6f}, '
                  f'epoch_loss: {epoch_loss:4.4f}, val_loss: {val_loss:4.4f}, '
                  f'train_acc: {train_acc:0.4f}, val_acc: {val_acc:0.4f}')

        sched.step()


def run():
    model = torch.hub.load(
        'AdeelH/WideResNet-pytorch:torch_hub',
        'WideResNet',
        depth=28,
        num_classes=NUM_CLASSES,
        widen_factor=2)
    model = model.cuda()
    # ema_model = ModelEMA(model, decay=0.999)
    train_params = {}
    train_params['batch_size'] = 100
    train_params['val_batch_size'] = 512

    train_ds, train_subset_ds, val_ds = get_datasets(subset_size=4000)

    train_dl = torch.utils.data.DataLoader(
        train_subset_ds,
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

    train_cl(
        model,
        train_dl,
        val_dl,
        optimizer,
        sched,
        train_params,
        w_scale=20,
        rampup_epochs=80,
        rampdown_epochs=50)


if __name__ == '__main__':
    run()

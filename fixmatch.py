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

weak_aug_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(scale_limit=0, rotate_limit=12.5, p=0.5), base_tf
])

strong_aug_tf = A.Compose([
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


def train_epoch_fixmatch(epoch,
                         model,
                         ema_model,
                         train_dl_l,
                         train_dl_ul,
                         optimizer,
                         sched,
                         conf_thresh,
                         w_u,
                         bar_label_prefix=''):

    torch.cuda.reset_peak_memory_stats()

    epoch_loss = 0.
    train_corrects = 0
    total = 0

    bar_label = f'{bar_label_prefix}{"Training":10s}'
    model.train()
    it = zip(cycle(train_dl_l), train_dl_ul)
    with tqdm(it, desc=bar_label, total=len(train_dl_ul)) as bar:
        for i, ((x_l, y_l), (x_ul, _)) in enumerate(bar):

            x_l, y_l = apply_transform(x_l, y_l, tf=base_tf)

            x_l = x_l.cuda()
            y_l = y_l.cuda()

            out_l = model(x_l)
            supervised_loss = F.cross_entropy(out_l, y_l)

            loss = 0.
            loss += supervised_loss

            x_weak_aug, _ = apply_transform(x_ul, y=None, tf=weak_aug_tf)
            x_weak_aug = x_weak_aug.cuda()

            with torch.no_grad():
                # model.eval()
                out_weak_aug = model(x_weak_aug).detach().softmax(dim=-1)
                # model.train()
                # out_weak_aug = ema_model(x_weak_aug).detach().softmax(dim=-1)

            max_vals, max_inds = out_weak_aug.cpu().max(dim=-1)
            high_confidence_mask = max_vals >= conf_thresh
            unsupervised_loss = None
            if high_confidence_mask.any():
                y_ul = max_inds[high_confidence_mask]
                x_ul = x_ul[high_confidence_mask]

                x_strong_aug, y_ul = apply_transform(
                    x_ul, y=y_ul, tf=strong_aug_tf)
                x_strong_aug = x_strong_aug.cuda()
                y_ul = y_ul.cuda()

                out_ul = model(x_strong_aug)

                unsupervised_loss = F.cross_entropy(out_ul, y_ul)
                loss += w_u * unsupervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_model.update(model)
            sched.step()

            epoch_loss += loss.detach().item()
            preds = out_l.detach().argmax(dim=-1)
            train_corrects += (preds == y_l).detach().cpu().float().sum()
            total += len(y_l)
            bar.set_postfix({
                'sup_loss':
                f'{supervised_loss.item():.03f}',
                'unsup_loss':
                f'{unsupervised_loss.item():.03f}'
                if unsupervised_loss is not None else '0.000',
                'pseudo_used':
                f'{high_confidence_mask.float().mean().item():.03f}'
            })

    train_acc = train_corrects / total
    return epoch_loss, train_acc


def train_fixmatch(model,
                   ema_model,
                   train_dl_l,
                   train_dl_ul,
                   val_dl,
                   optimizer,
                   sched,
                   params,
                   conf_thresh=0.95,
                   w_u=1.,
                   start_epoch=0,
                   eval_interval=1):
    num_epochs = params['epochs']
    for epoch in range(start_epoch, num_epochs):

        bar_label_prefix = f'Epoch {epoch+1}/{num_epochs}: '

        # train
        epoch_loss, train_acc = train_epoch_fixmatch(
            epoch,
            model,
            ema_model,
            train_dl_l,
            train_dl_ul,
            optimizer,
            sched,
            conf_thresh=conf_thresh,
            w_u=w_u,
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


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        current_step -= num_warmup_steps
        training_steps = (num_training_steps - num_warmup_steps)
        t = (1. * current_step) / (max(1, training_steps))
        return max(0., np.cos(np.pi * num_cycles * t))

    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


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
    train_params['batch_size_l'] = 64
    train_params['batch_size_ul_prop'] = 7
    train_params['batch_size_ul'] = (
        train_params['batch_size_l'] * train_params['batch_size_ul_prop'])
    train_params['val_batch_size'] = 256

    train_ds, train_subset_ds, val_ds = get_datasets(subset_size=4000)
    train_dl_l = torch.utils.data.DataLoader(
        train_subset_ds,
        batch_size=train_params['batch_size_l'],
        pin_memory=True,
        num_workers=4,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn)
    train_dl_ul = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_params['batch_size_ul'],
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
    train_params['learning_rate'] = 3e-2

    optimizer = optim.SGD(
        model.parameters(),
        lr=train_params['learning_rate'],
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4)
    # sched = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=train_params['epochs'], eta_min=0)
    sched = get_cosine_schedule_with_warmup(
        optimizer, 0, train_params['epochs'] * len(train_dl_l))

    train_fixmatch(
        model,
        ema_model,
        train_dl_l,
        train_dl_ul,
        val_dl,
        optimizer,
        sched,
        train_params,
        conf_thresh=0.95,
        w_u=1,
        start_epoch=0,
        eval_interval=1)


if __name__ == '__main__':
    run()

import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from utils import test_single_volume
import pickle

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator, Synapse_dataset_valid
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_valid = Synapse_dataset_valid(base_dir=args.root_path, list_dir=args.list_dir, split="valid",transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    validloader = DataLoader(db_valid, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    lower_loss = 10000
    val_dice_loss = []
    val_multi_loss = []
    val_dice_acc = []
    train_dice_loss = []
    train_multi_loss = []
    train_dice_acc = []
    if args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids=[0,1,2,3,4])
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    params = model.parameters()
    optimizer = optim.Adam(params, lr=base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            # gt = torch.eye(2).cuda()[label_batch.type(torch.long)].permute(0, 3, 1, 2)
            # pred = outputs
            # # print(f'')
            # # print('gt' +  str(gt.shape))
            # # print(pred.shape)
            # intersection = (pred * gt).sum()
            # new_dice = (2. * intersection + 1) / (pred.sum() + gt.sum() + 1)

            # train_dice_acc.append(new_dice)
            train_multi_loss.append(loss.item())
            train_dice_loss.append(loss_dice.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/dice_loss', loss_dice, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(f'iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(validloader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice

                # gt = torch.eye(2).cuda()[label_batch.type(torch.long)].permute(0, 3, 1, 2)
                # pred = outputs
                # print(f'')
                # print('gt' +  str(gt.shape))
                # print(pred.shape)
                # intersection = (pred * gt).sum()
                # new_dice = (2. * intersection + 1) / (pred.sum() + gt.sum() + 1)

                # val_dice_acc.append(new_dice)
                val_multi_loss.append(loss.item())
                val_dice_loss.append(loss_dice.item())

                writer.add_scalar('info/val_dice_loss', loss_dice, iter_num)
                writer.add_scalar('info/val_loss_ce', loss_ce, iter_num)

                logging.info(f'iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

                if loss.item() < lower_loss:
                    lower_loss = loss.item()
                    save_mode_path = os.path.join(snapshot_path, 'epoch_'+ str(epoch_num)+ 'loss_' + str(loss.item())  + '.pth')
                    torch.save(model.state_dict(), save_mode_path)




            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    log_dict = {
        'val_dice_loss': val_dice_loss,
        'val_multi_loss': val_multi_loss,
        # 'val_dice_acc': val_dice_acc,
        'train_dice_loss': train_dice_loss,
        'train_multi_loss': train_multi_loss,
        # 'train_dice_acc': train_dice_acc,
    }
    pickle.dump(log_dict, open('log_dict.pkl', 'wb'))

    writer.close()
    return "Training Finished!"
import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
from copy import deepcopy


import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, \
    add_weight_decay

# from models.model_Trimodal import buildTriModalTransformer
from models.LCNet import buildLLCDTransformer

from utils.logger import setup_logger
from utils.slconfig import get_raw_dict
from utils.misc import clean_state_dict
# from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

#import init_paths
from dataset.get_dataset import get_datasets

def parser_args():
    parser = argparse.ArgumentParser(description='GQ2L Tri-modal decoder')
    parser.add_argument('--dataname', help='dataname', default='coco14', choices=['coco14', 'rope', 'voc'])
    parser.add_argument('--dataset_dir', help='dir of dataset', default='/comp_robot/liushilong/data/COCO14/')
    parser.add_argument('--img_size', default=448, type=int,
                        help='size of input images')
    parser.add_argument('--CropLevels', default=0, type=int,
                        help='size of input images')
    parser.add_argument('--imgInpsize', default=448, type=int,
                        help='size of input images')

    parser.add_argument('--output', metavar='DIR', 
                        help='path to output folder')
    parser.add_argument('--num_class', default=80, type=int,
                        help="Number of query slots")
    
   
    
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False, 
                        help='disable_torch_grad_focal_loss in asl')              
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=4, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                                            help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.05, type=float,
                                            help='scale factor for clip')  

    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=1000, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--print_freq_val', default=1000, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')


    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')


    # data aug
    parser.add_argument('--cutout', action='store_true', default=True,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')              
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ') 

    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')


    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=680, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=680, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")
    parser.add_argument('--remove_aa_jit', action='store_true', default=False,
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")
    
    

    # training
    parser.add_argument('--isTrain', action='store_true', default=False,
                        help='True if training')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    
    # Validation:
    parser.add_argument('--thr', default=0.85, type=float,
                    metavar='N', help='threshold value')

    # ML-Decoder

    parser.add_argument('--data', type=str, default='/home/MSCOCO_2014/')
    # parser.add_argument('--model-name', default='tresnet_l')
    parser.add_argument('--model-path', default='/home/menno/GQ2LTrimodal_V2/tresnet_l.pth', type=str)
    parser.add_argument('--num-classes', default=80)
    parser.add_argument('--image-size', default=448, type=int,
                        metavar='N', help='input image size (default: 448)')

    parser.add_argument('--num_classes', default=80, type=int,
                        help="Number of query slots")
    # parser.add_argument('--model_name', help='dataname', default='tresnet_l', choices=['tresnet_l', 'rope', 'voc'])
    parser.add_argument('--use-ml-decoder', default=0, type=int)
    parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
    parser.add_argument('--decoder-embedding', default=680, type=int)
    parser.add_argument('--zsl', default=0, type=int)


    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args



best_mAP = 0

def main():
    args = get_args()

    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        # single process, useful for debugging
        #   python main.py ...
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    
    logger = setup_logger(output=args.output, distributed_rank=0, color=False, name="GQ2L Trimodal-decoder")
    logger.info("Command: "+' '.join(sys.argv))

    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))


    # Setup model
    logger.info('creating model and backbone: {}.'.format(args.backbone))

    


    # # COCO Data loading
    # instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    # instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    # #data_path_val = args.data
    # #data_path_train = args.data
    # data_path_val = f'{args.data}/val2014'  # args.data
    # data_path_train = f'{args.data}/train2014'  # args.data
    # val_dataset = CocoDetection(data_path_val,
    #                             instances_path_val,
    #                             transforms.Compose([
    #                                 transforms.Resize((args.image_size, args.image_size)),
    #                                 transforms.ToTensor(),
    #                                 # normalize, # no need, toTensor does normalization
    #                             ]))
    # train_dataset = CocoDetection(data_path_train,
    #                               instances_path_train,
    #                               transforms.Compose([
    #                                   transforms.Resize((args.image_size, args.image_size)),
    #                                   CutoutPIL(cutout_factor=0.5),
    #                                   RandAugment(),
    #                                   transforms.ToTensor(),
    #                                   # normalize,
    #                               ]))
    # print("len(val_dataset)): ", len(val_dataset))
    # print("len(train_dataset)): ", len(train_dataset))

    # # Pytorch Data loader
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=False)

    # # Actuall Training
    # train_multi_label_coco(model, train_loader, val_loader, args.lr)

    return main_worker(args, logger)

def main_worker(args, logger):
    global best_mAP

    # Build model:
    model = buildLLCDTransformer(args)
    model = model.cuda()

    # model = create_model(args).cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    summary_writer = SummaryWriter(log_dir=args.output)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(0))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dicr Found!!!")
            logger.info("Omitting {}".format(args.resume_omit))
            # import ipdb; ipdb.set_trace()
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.load_state_dict(state_dict, strict=False)
            # model.module.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    # load datasets:

    # Data loading code
    train_dataset, val_dataset, test_dataset = get_datasets(args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(args.batch_size), shuffle=False,
        num_workers=args.workers, pin_memory=False)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=int(args.batch_size), shuffle=False,
        num_workers=args.workers, pin_memory=False)
    

    


    # set optimizer
    Epochs = args.epochs
    weight_decay = args.weight_decay

    lr = args.lr
    criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=args.loss_clip, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)
    

    # if args.evaluate:
    #     _, mAP = validate(val_loader, model, criterion, args, logger)
    #     logger.info(' * mAP {mAP:.5f}'
    #           .format(mAP=mAP))
    #     return
    
    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, mAPs, losses_ema, mAPs_ema],
        prefix='=> Test Epoch: ')
    
    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    
    torch.cuda.empty_cache()

    ema = ModelEma(model, args.ema_decay)  # 0.9997^641=0.82

    if args.isTrain: 

        for epoch in range(args.start_epoch, args.epochs):
            # train_sampler.set_epoch(epoch)

            if args.ema_epoch == epoch:
                # ema_m = ModelEma(model, args.ema_decay)
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            # train for one epoch
            loss = train(train_loader, model, ema, criterion, optimizer, scheduler, epoch, args, logger)

            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('train_loss', loss, epoch)
                # summary_writer.add_scalar('train_acc1', acc1, epoch)
                summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)


            if  epoch % args.val_interval == 0:
                loss, loss_ema, mAP, mAP_ema = validate(val_loader, model, ema, criterion, args, logger)

                losses.update(loss)
                mAPs.update(mAP)
                losses_ema.update(loss_ema)
                mAPs_ema.update(mAP_ema)
                epoch_time.update(time.time() - end)
                end = time.time()
                eta.update(epoch_time.avg * (args.epochs - epoch - 1))

                regular_mAP_list.append(mAP)
                ema_mAP_list.append(mAP_ema)

                progress.display(epoch, logger)

                if summary_writer:
                    # tensorboard logger
                    summary_writer.add_scalar('val_loss', loss, epoch)
                    summary_writer.add_scalar('val_mAP', mAP, epoch)
                    summary_writer.add_scalar('val_loss_ema', loss_ema, epoch)
                    summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)

                if mAP > best_regular_mAP:
                    best_regular_mAP = max(best_regular_mAP, mAP)
                    best_regular_epoch = epoch
                if mAP_ema > best_ema_mAP:
                    best_ema_mAP = max(mAP_ema, best_ema_mAP)
                
                if mAP_ema > mAP:
                    mAP = mAP_ema
                    state_dict = ema.module.state_dict()
                else:
                    state_dict = model.state_dict()
                is_best = mAP > best_mAP
                if is_best:
                    best_epoch = epoch
                best_mAP = max(mAP, best_mAP)

                logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
                logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))

            
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args,
                    'state_dict': state_dict,
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint.pth.tar'))
                # filename=os.path.join(args.output, 'checkpoint_{:04d}.pth.tar'.format(epoch))

                if math.isnan(loss) or math.isnan(loss_ema):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args,
                        'state_dict': model.state_dict(),
                        'best_mAP': best_mAP,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint_nan.pth.tar'))
                    logger.info('Loss is NaN, break')
                    sys.exit(1)


                # early stop
                if args.early_stop:
                    if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 8:
                        if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                            logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                            if args.kill_stop:
                                filename = sys.argv[0].split(' ')[0].strip()
                                killedlist = kill_process(filename, os.getpid())
                                logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist)) 

    else:
        validate_multi(test_loader, ema, args)




def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger):
    batch_time = AverageMeter('T', ':5.3f')
    data_time = AverageMeter('DT', ':5.3f')
    speed_gpu = AverageMeter('S1', ':.1f')
    speed_all = AverageMeter('SA', ':.1f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)

   

    # print("ALMOST FINISHED 15.2")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, speed_gpu, speed_all, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))
    
    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
        
    lr.update(args.lr)
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    scaler = GradScaler()

    model.train()

    end = time.time()

    for i, (inputData, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # print("Target: ", target)
        # target = target.max(dim=0)[0]


        inputData = inputData.cuda()
        target = target.cuda()
        # target = target.max(dim=1)

        with autocast():  # mixed precision
            output = model(inputData).float()  # sigmoid will be done in loss !

            # print("output: ", output)
            # print("Target: ", target)

        loss = criterion(output, target)


        model.zero_grad()

        # print("Regular outputs: ")
        # print("Train output_regular: ", output)
        # print("Train loss_reg: ", loss)

        losses.update(loss.item(), inputData.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        
        scaler.scale(loss).backward()
        # loss.backward()

        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

        scheduler.step()

        lr.update(get_learning_rate(optimizer))

        ema_m.update(model)
        # store information
        batch_time.update(time.time() - end)
        end = time.time()
        speed_gpu.update(inputData.size(0) / batch_time.val, batch_time.val)
        speed_all.update(inputData.size(0) * 1 / batch_time.val, batch_time.val)

        if i % args.print_freq == 0:

            progress.display(i, logger)

    model.eval()

    return losses.avg
    

# def val(val_loader, model, ema, criterion, args, logger):

#     model.eval()

#     loss, mAP_score = validate(val_loader, model, ema, criterion, args, logger)

#     if mAP_score > highest_mAP:
#         highest_mAP = mAP_score
#         try:
#             torch.save(model.state_dict(), os.path.join(
#                 'models/', 'model-highest.ckpt'))
#         except:
#             pass
#     print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))


def validate(val_loader, model, ema_model, criterion, args, logger):
    model.eval()
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    losses_ema = AverageMeter('Loss_ema', ':5.3f')
    # Acc1 = AverageMeter('Acc@1', ':5.2f')
    # top5 = AverageMeter('Acc@5', ':5.2f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    # mAP = AverageMeter('mAP', ':5.3f', val_only=)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, losses_ema, mem],
        prefix='Test: ')
    

    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []

    end = time.time()

    saved_data = []

    for i, (input, target) in enumerate(val_loader):

        # input = input.cuda()
        # target = target.cuda()

        # target = target.max(dim=0)[0]

        

        # compute output
        with torch.no_grad():
            # with autocast():
            output_regular = Sig(model(input.cuda())).cpu()
            

            # print("output_regular: ", output_regular)
            # print("Target: ", target)

            # print("Regular outputs: ")
            # print("output_regular: ", output_regular)
            # print("loss_reg: ", loss_reg)

            output_ema = Sig(ema_model.module(input.cuda())).cpu()
        loss_reg = criterion(output_regular, target)
        loss_ema = criterion(output_ema, target)

        # print("EMA outputs: ")
        # print("output_ema: ", output_ema)
        # print("loss_ema: ", loss_ema)

        # record loss
        losses.update(loss_reg.item(), input.size(0))
        losses_ema.update(loss_ema.item(), input.size(0))
        # losses.update(loss_ema.item(), input.size(0))

        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

        # print("Shape output: ", output_ema.detach().cpu().shape)
        # print("target output: ", target.detach().cpu().shape)

        _item = torch.cat((output_ema.detach().cpu(), target.detach().cpu()), 1)
        # del output_sm
        # del target
        saved_data.append(_item)
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    saved_data = torch.cat(saved_data, 0).numpy()
    saved_name = 'saved_data_tmp.{}.txt'.format(0)
    np.savetxt(os.path.join(args.output, saved_name), saved_data)

    print("Calculating mAP:")
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))

    
    logger.info("  mAP: {}".format(mAP_score_regular))
    logger.info("  mAP_ema: {}".format(mAP_score_ema))

    loss_avg = losses.avg
    
    loss_avg_ema = losses_ema.avg

    model.train()

    return loss_avg, loss_avg_ema, mAP_score_regular, mAP_score_ema


def validate_multi(val_loader, model, args):

    batch_time = AverageMeter('Time', ':5.3f')
    prec = AverageMeter('Prec', ':5.3f')
    rec = AverageMeter('Rec', ':5.3f')
    mAP_meter = AverageMeter('mAP', ':5.3f')

    Sig = torch.nn.Sigmoid()

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        # target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            # output = Sig(model(input.cuda().half())).cpu()
            output = Sig(model.module(input.cuda())).cpu()

        pred = output.data.gt(args.thr).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        targetarr = target
        outputarr = output

        # for mAP calculation
        preds.append(outputarr.cpu())
        targets.append(targetarr.cpu())

        if i % args.print_freq == 0:
            mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
            # mAP_score = mAP(np.array(targets), np.array(preds))
            print("mAP score:", mAP_score)
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time,
                prec=prec, rec=rec))
            print(
                'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                    .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    print(
        '--------------------------------------------------------------------')
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())

    print("mAP score:", mAP_score)

    return

##################################################################################
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""
    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name, 
                            val=str(datetime.timedelta(seconds=int(self.val))), 
                            sum=str(datetime.timedelta(seconds=int(self.sum))))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

def kill_process(filename:str, holdpid:int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True, cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist

if __name__ == '__main__':
    main()




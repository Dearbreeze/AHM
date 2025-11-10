import math
import sys
import os
import cv2
from typing import Iterable
import torch
import numpy as np
import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import skimage.io as io 
from skimage import img_as_ubyte
import State_test as State
from pixelwise_a3c_test import *


def train_one_epoch(model: torch.nn.Module, Agent: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    Agent.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args['accum_iter']
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    mask_size = int(args['input_size'] * 1.0 / args['Division_number'] + 0.5)
    current_state = State.State((args['batch_size'], 1, args['input_size'], args['input_size']), mask_size)
    agent = PixelWiseA3C_InnerState(args, Agent, args['batch_size'], args['GAMMA'])

    for data_iter_step, (samples, file_name) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        current_state.reset(samples.numpy())
        samples = samples.to(device, non_blocking=True)
        Mask = agent.act(current_state.image, args['mask_ratio'])
        # torch.set_printoptions(threshold=np.inf)
        # print(Mask.size())
        # print(Mask)
        Mask = Mask.repeat(args['input_size'] / mask_size, axis=2).repeat(args['input_size'] / mask_size, axis=3)
        Mask = torch.from_numpy(Mask).to(device)
        with torch.cuda.amp.autocast():
            output, loss, mask, masked_x = model(samples, Mask)
        
      
        img_path = args['base_dir'] + '/visualization_test_maskadaptive/'
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        
        ## display images
        # if epoch == 0 or epoch == 1  or epoch == 200 or epoch == 300 or epoch == 400 :
        if  epoch == 800:
            if not os.path.exists(os.path.join(args['output_dir'],'train','mask')):
                os.makedirs(os.path.join(args['output_dir'],'train','mask'))
            if not os.path.exists(os.path.join(args['output_dir'],'train','recon')):
                os.makedirs(os.path.join(args['output_dir'],'train','recon'))
            if not os.path.exists(os.path.join(args['output_dir'],'train','input')):
                os.makedirs(os.path.join(args['output_dir'],'train','input'))
            # save_image(labelval,os.path.join(args['output_dir'],opt.name,'val','SEG_GT',file_name[0] + '.png'))
            mask = mask.cpu().numpy()
            masked_x = masked_x.cpu().numpy()
            output = output.cpu().detach().numpy()
            for k in range(len(file_name)) :
                cv2.imwrite(os.path.join(args['output_dir'],'train','mask',file_name[k] + '.png'),mask[k,...].squeeze(0)*255)
                cv2.imwrite(os.path.join(args['output_dir'],'train','input',file_name[k] + '.png'),masked_x[k,...].squeeze(0)*255)
                cv2.imwrite(os.path.join(args['output_dir'],'train','recon',file_name[k] + '.png'),np.float32((output[k,...]*mask[k,...] + masked_x[k,...]*(1-mask[k,...])).squeeze(0))*255)

            

        # loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter

        ## type1
        # if (data_iter_step + 1) % accum_iter == 0:
        #     loss.backward()
        #     optimizer.step()
        ## type2
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},loss_value
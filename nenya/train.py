from __future__ import print_function

import time
import os
import h5py
import numpy as np
from tqdm.auto import trange

from nenya import io as nenya_io
from nenya.train_util import set_model, train_model
from nenya.train_util import nenya_loader
from nenya.params import option_preprocess
from nenya.util import set_optimizer, save_model
from nenya.util import adjust_learning_rate


def main(opt_path: str, debug:bool=False, save_file:str=None):
    """
    Trains a model using the specified parameters.

    Args:
        opt_path (str): The path to the parameters JSON file.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        save_file (str, optional): The path to save the trained model. Defaults to None.
    """
    # loading parameters json file
    opt = nenya_io.Params(opt_path)
    if debug:
        opt.epochs = 2
    option_preprocess(opt)

    # Save opts
    opt.save(os.path.join(opt.model_folder, 
                          os.path.basename(opt_path)))
    
    # build model and criterion
    model, criterion = set_model(opt, cuda_use=opt.cuda_use)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    loss_train, loss_step_train, loss_avg_train = [], [], []
    loss_valid, loss_step_valid, loss_avg_valid = [], [], []

    # Loop me
    for epoch in trange(1, opt.epochs + 1): 
        train_loader = nenya_loader(opt)
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, losses_step, losses_avg = train_model(
            train_loader, model, criterion, optimizer, epoch, opt, 
            cuda_use=opt.cuda_use)

        # record train loss
        loss_train.append(loss)
        loss_step_train += losses_step
        loss_avg_train += losses_avg

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # Free up memory
        del train_loader

        # Validate?
        if epoch % opt.valid_freq == 0:
            # Data Loader
            valid_loader = nenya_loader(opt, valid=True)
            #
            epoch_valid = epoch // opt.valid_freq
            time1_valid = time.time()
            loss, losses_step, losses_avg = train_model(
                valid_loader, model, criterion, optimizer, epoch_valid, opt, 
                cuda_use=opt.cuda_use, update_model=False)
           
            # record valid loss
            loss_valid.append(loss)
            loss_step_valid += losses_step
            loss_avg_valid += losses_avg
        
            time2_valid = time.time()
            print('valid epoch {}, total time {:.2f}'.format(epoch_valid, time2_valid - time1_valid))

            # Free up memory
            del valid_loader 

        # Save model?
        if (epoch % opt.save_freq) == 0:
            # Save locally
            save_file = os.path.join(opt.model_folder,
                                     f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)
            
    # save the last model local
    save_file = os.path.join(opt.model_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # Save the losses
    if not os.path.isdir(f'{opt.model_folder}/learning_curve/'):
        os.mkdir(f'{opt.model_folder}/learning_curve/')
        
    losses_file_train = os.path.join(opt.model_folder,'learning_curve',
                                     f'{opt.model_name}_losses_train.h5')
    losses_file_valid = os.path.join(opt.model_folder,'learning_curve',
                                     f'{opt.model_name}_losses_valid.h5')
    
    with h5py.File(losses_file_train, 'w') as f:
        f.create_dataset('loss_train', data=np.array(loss_train))
        f.create_dataset('loss_step_train', data=np.array(loss_step_train))
        f.create_dataset('loss_avg_train', data=np.array(loss_avg_train))
    with h5py.File(losses_file_valid, 'w') as f:
        f.create_dataset('loss_valid', data=np.array(loss_valid))
        f.create_dataset('loss_step_valid', data=np.array(loss_step_valid))
        f.create_dataset('loss_avg_valid', data=np.array(loss_avg_valid))
        
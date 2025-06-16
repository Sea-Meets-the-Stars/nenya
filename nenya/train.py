from __future__ import print_function

import time
import os
import h5py
import numpy as np
from tqdm.auto import trange
import torch

from nenya import io as nenya_io
from nenya.train_util import set_model, train_model
from nenya.train_util import nenya_loader
from nenya.train_util import save_losses
from nenya import params 
from nenya.util import set_optimizer, save_model
from nenya.util import adjust_learning_rate

from IPython import embed


def main(opt_path:str, debug:bool=False, load_epoch:int=None):
    """
    Trains a model using the specified parameters.

    Args:
        opt_path (str): The path to the parameters JSON file.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        load_epoch (bool, optional): If provided, the model will be loaded from this epoch.
        prev_losses_file (str): If load epoch is provided, this file will be used to load previous losses.

    """
    # loading parameters json file
    opt = params.Params(opt_path)
    if debug:
        opt.epochs = 2
    params.option_preprocess(opt)

    # Save opts
    opt.save(os.path.join(opt.model_folder, 
                          os.path.basename(opt_path)))


    # build model and criterion
    model, criterion = set_model(opt, cuda_use=opt.cuda_use)

    # build optimizer
    optimizer = set_optimizer(opt, model)


    # Losses
    loss_train, loss_step_train, loss_avg_train = [], [], []
    loss_valid, loss_step_valid, loss_avg_valid = [], [], []

    # set start_epoch
    start_epoch = 1
    # Load pre-trained model if requested
    if load_epoch is not None:

        # Check for lesses files
        losses_file_train, losses_file_valid = nenya_io.losses_filenames(opt)
        # Local?
        if not os.path.exists(losses_file_train): 
            losses_file_train = os.path.basename(losses_file_train)
            if not os.path.exists(losses_file_train):
                print(f"Warning: Losses file '{losses_file_train}' not found.")
                print(f"Expected path: {losses_file_train} [or the full path]")
                print("Training stopped.")
                return
            losses_file_valid = os.path.basename(losses_file_valid)

        model_file = os.path.join(opt.model_folder, f'ckpt_epoch_{load_epoch}.pth')

        if not os.path.exists(model_file):
            print(f"Warning: Model file '{model_file}' not found.")
            print(f"Expected path: {model_file}")
            print("Training stopped.")
            return

        #try:
        print(f"Loading model from epoch {load_epoch}: {model_file}")
        checkpoint = torch.load(model_file, map_location='cpu' if not opt.cuda_use else 'cuda')

        # Load model state
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Set starting epoch
        start_epoch = load_epoch + 1
        print("Model Loading Sucessfully!")

        # Load losses
        f = h5py.File(losses_file_valid, 'r')
        loss_valid = np.array(f['loss_valid'][:]).tolist()
        loss_step_valid = np.array(f['loss_step_valid'][:]).tolist()
        loss_avg_valid = np.array(f['loss_avg_valid'][:]).tolist()
        f.close()
        f = h5py.File(losses_file_train, 'r')
        loss_train = np.array(f['loss_train'][:]).tolist()
        loss_step_train = np.array(f['loss_step_train'][:]).tolist()
        loss_avg_train = np.array(f['loss_avg_train'][:]).tolist()
        f.close()
        # Check
        assert len(loss_valid) == load_epoch
        print(f"Losses loaded")

    # Adjust total epochs if loading from a checkpoint
    if load_epoch is not None and start_epoch > opt.epochs:
        print(f"Warning: Starting epoch ({start_epoch}) is greater than total epochs ({opt.epochs})")
        print("No additional training needed.")
        return

    # Loop me
    for epoch in trange(start_epoch, opt.epochs + 1): 
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

            # Save losses
            save_losses(opt, loss_train, loss_step_train, loss_avg_train,
                loss_valid, loss_step_valid, loss_avg_valid)
            
            
    # save the last model local
    save_file = os.path.join(opt.model_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

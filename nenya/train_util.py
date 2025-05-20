""" Code related to Nenya training """
import os
import sys
import numpy as np
import time

import skimage.transform
import skimage.filters
import h5py

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import Dataset

from nenya.models.resnet_big import SupConResNet
from nenya.losses import SupConLoss

from nenya.util import TwoCropTransform, AverageMeter
from nenya.util import warmup_learning_rate

from IPython import embed

class Demean:
    """
    Remove the mean of the image
    """
    def __call__(self, image):
        """ Demean

        Args:
            image (np.ndarray): Expected to be a 2D image

        Returns:
            np.ndarray: Demeaned image, 2D
        """
        image -= image.mean()
        # Return
        return image

class ThreeChannel:
    """
    Turn into 3 channel
    """
    def __call__(self, image):
        """ 

        Args:
            image (np.ndarray): Expected to be a 2D image

        Returns:
            np.ndarray: Demeaned image, 3D
        """
        # 3 channels
        image = np.repeat(image, 3, axis=-1)
        #
        return image
    
class RandomRotate:
    """
    Random Rotation Augmentation of the training samples.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
    def __call__(self, image):
        """ 
        Args:
            image (np.ndarray): Expected to be a 2D image

        Returns:
            np.ndarray: Randomly rotated image, 2D
        """
        #if self.verbose:
        #    print(f'Shape of image: {image.shape}')
        rang = np.float32(360*np.random.rand(1))
        if self.verbose:
            print(f'RandomRotate: {rang}')
        # Return
        return (skimage.transform.rotate(image, rang[0])).astype(np.float32)

class RandomFlip:
    """
    Random Rotation Augmentation of the training samples.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
    def __call__(self, image):
        """ 
        Args:
            image (np.ndarray): Expected to be a 3D image 

        Returns:
            np.ndarray: Randomly flipped image, 3D
        """
        if self.verbose:
            print(f'Shape of image: {image.shape}')
        # Left/right
        rflips = np.random.randint(2, size=2)
        if rflips[0] == 1:
            image = image[:, ::-1]
        # Up/down
        if rflips[1] == 1:
            image = image[::-1, :]
        if self.verbose:
            print(f'RandomFlip: {rflips}')
        # Return
        return image
    
class JitterCrop:
    """
    Random Jitter and Crop augmentaion of the training samples.

    Args:
        crop_dim (int): Crop dimension
            Size of the crop
        jitter_lim (int): Jitter limit
        rescale (int): Rescale factor
            Note: this is not implemented yet
        verbose (bool): Verbose flag
    """
    def __init__(self, crop_dim=32, rescale:int=0, jitter_lim=0, 
                 verbose=False):
        self.jitter_lim = jitter_lim
        self.crop_dim = crop_dim

        # Half-size of the final image
        self.offset = self.crop_dim//2
        # Rescale
        self.rescale = rescale
        if self.rescale > 0:
            raise IOError("Rescale not implemented yet")
        self.verbose = verbose

        if self.verbose:
            print(f'Offset: {self.offset}')
        
    def __call__(self, image):
        """ 
        Args:
            image (np.ndarray): Expected to be a 2D image

        Returns:
            np.ndarray: Jittered + Cropped image, 2D
        """
        if self.verbose:
            print(f'Shape of image input to JitterCrop: {image.shape}')
        center_x = image.shape[0]//2
        center_y = image.shape[0]//2
        if self.jitter_lim:
            center_x += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
            center_y += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))

        row0 = max(center_x - self.offset, 0)
        row1 = min(center_x + self.offset, image.shape[0])
        col0 = max(center_y - self.offset, 0)
        col1 = min(center_y + self.offset, image.shape[1])

        image_cropped = image[row0:row1,col0:col1,:]
            #(center_x-self.offset):(center_x+self.offset), 
                              #(center_y-self.offset):(center_y+self.offset), 
                              #:]
        if self.verbose:
            print(f'Shape of image cropped: {image_cropped.shape}')
            print(f'center_x: {center_x}, center_y: {center_y}')

        if self.rescale > 0:
            # THIS WILL PROBABLY BREAK
            #image = skimage.transform.rescale(image_cropped, self.rescale)
            image = np.expand_dims(skimage.transform.rescale(image_cropped, self.rescale), axis=-1)
            # 3 channels (only )
            image = np.repeat(image, 3, axis=-1)
        else:
            image = skimage.transform.resize(image_cropped, image.shape)

        # Verbose?
        if self.verbose:
            print(f'JitterCrop: {center_x, center_y}')
            print(f'Shape exiting JitterCrop: {image.shape}')

        # Return 
        return image
    
class RandomJitterCrop:
    """
    Random Jitter and Crop Augmentation used in v2 

    DEPRECATED
    """
    def __init__(self, crop_lim=5, jitter_lim=5):
        self.crop_lim = crop_lim
        self.jitter_lim = jitter_lim
        
    def __call__(self, image):
        offset_left, offset_right = 0, 0
        offset_low, offset_high = 0, 0
        
        # Random realization
        rand_crop_x = int(np.random.randint(0, self.crop_lim+1, 1))
        rand_crop_y = int(np.random.randint(0, self.crop_lim+1, 1))
        jitter_lim_x = min(rand_crop_x//2, self.jitter_lim)
        jitter_lim_y = min(rand_crop_y//2, self.jitter_lim)
        rand_jitter_x = int(np.random.randint(-jitter_lim_x, jitter_lim_x+1))
        rand_jitter_y = int(np.random.randint(-jitter_lim_y, jitter_lim_y+1))
        
        # Crop in x
        if rand_crop_x > 0:
            offset_left = rand_crop_x // 2
            offset_right = rand_crop_x - offset_left
            if rand_jitter_x != 0:
                offset_left -= rand_jitter_x
                offset_right += rand_jitter_x
                
        # Crop in y
        if rand_crop_y > 0:
            offset_low = rand_crop_y // 2
            offset_high = rand_crop_y - offset_low
            if rand_jitter_y != 0:
                offset_low -= rand_jitter_y
                offset_high += rand_jitter_y
        
        image_width, image_height = image.shape[0], image.shape[1]
        
        ### comment these commands after test
        #assert (offset_left + offset_right) == rand_crop_x, "Crop is Wrong!"
        #assert (offset_low + offset_high) == rand_crop_y, "Crop is Wrong!"
        #assert (offset_low >= 0) and (offset_left >= 0), "Crop is Wrong!"
        #assert (offset_high >= 0) and (offset_right >= 0), "Crop is Wrong!"

        # Crop
        image_cropped = image[offset_left: image_width-offset_right, offset_low: image_height-offset_high]
        image = skimage.transform.resize(image_cropped, image.shape)
        image = np.repeat(image, 3, axis=-1)
        
        # Return
        return image
    
class GaussianNoise:
    """
    Gaussian Noise augmentation used for training samples.
    """
    def __init__(self, instrument_noise=(0, 0.1)):
        self.noise_mean = instrument_noise[0]
        self.noise_std = instrument_noise[1]

    def __call__(self, image):
        noise_shape = image.shape
        noise = np.random.normal(self.noise_mean, self.noise_std, size=noise_shape)
        image += noise

        return image
    
class GaussianBlurring:
    """
    Gaussian Blurring augmentation used for training samples.
    """
    def __init__(self, sigma=1):
        self.sigma = sigma
        
    def __call__(self, image):
        image_blurred = skimage.filters.gaussian(image, sigma=self.sigma, multichannel=False)
    
        return image_blurred
        
class NenyaDataset(Dataset):
    """
    Nenya Dataset used for the training of the model.
    """
    def __init__(self, data_path, transform, data_key='train'):
        self.data_path = data_path
        self.transform = transform
        self.data_key = data_key

    def _open_file(self):
        self.files = h5py.File(self.data_path, 'r')

    def __len__(self):
        self._open_file()
        num_samples = self.files[self.data_key].shape[0]
        return num_samples

    def __getitem__(self, global_idx):     
        self._open_file()
        # This may have shape [1, npix, npix]
        image = self.files[self.data_key][global_idx]
        # Build out
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        # Process
        image_transposed = np.transpose(image, (1, 2, 0))
        image_transformed = self.transform(image_transposed)
        
        return image_transformed

#@profile
def nenya_loader(opt, valid=False):
    """
    This is a function used to create a Nenya data loader.
    
    Args:
        opt: (Params) options for the training process.
        valid: (Bool) flag used to decide if to use the validation set.
        
    Returns:
        loader: (Dataloader) Nenya Dataloader.
    """
    # Construct the Augmentations
    augment_list = []
    #embed(header='294 of train_util')
    if opt.flip:
        augment_list.append(RandomFlip())
    if opt.rotate:
        augment_list.append(RandomRotate())
    if opt.random_cropjitter == 0:
        augment_list.append(JitterCrop())
    else:
        augment_list.append(JitterCrop(crop_dim=opt.random_cropjitter[0],
                                       jitter_lim=opt.random_cropjitter[1],
                                       rescale=0,
                                       verbose=False))
    if opt.demean:
        augment_list.append(Demean())

    # 3-chanel augmentation
    if opt.nchannels == 3:
        print("Using 3 channels")
        augment_list.append(ThreeChannel())
    elif opt.nchannels == 1:
        print("Using 1 channel")
    else:
        raise IOError(f"Number of channels={opt.nchannels} not supported")

    # Tensor
    augment_list.append(transforms.ToTensor())

    # Do it
    transforms_compose = transforms.Compose(augment_list)

    if not valid:
        data_key = opt.train_key
        batch_size = opt.batch_size_train
    else:
        data_key = opt.valid_key
        batch_size = opt.batch_size_valid
    data_file = os.path.join(opt.data_folder, opt.images_file)

    modis_dataset = NenyaDataset(data_file, 
                                 transform=TwoCropTransform(
                                     transforms_compose), 
                                 data_key=data_key)
    sampler = None                                
    loader = torch.utils.data.DataLoader(
                    modis_dataset, batch_size=batch_size, 
                    shuffle=(sampler is None),
                    num_workers=opt.num_workers, 
                    pin_memory=False, sampler=sampler)
    
    return loader

def modis_loader_v2_with_blurring(opt, valid=False):
    """
    This is a function used to create the modis data loader v2 with gaussian
    blurring.
    
    Args:
        opt: (Params) options for the training process.
        valid: (Bool) flag used to decide if to use the validation set.
        
    Returns:
        train_loader: (Dataloader) Modis Dataloader.
    """
    raise IOError("Modify the loader instead!!")
    transforms_compose = transforms.Compose([RandomRotate(),
                                             RandomJitterCrop(),
                                             GaussianBlurring(),
                                             GaussianNoise(instrument_noise=(0, 0.05)),
                                             transforms.ToTensor()])
    if not valid:
        modis_path = opt.data_folder
    else:
        modis_path = opt.valid_folder
    modis_file = os.path.join(modis_path, os.listdir(modis_path)[0])
    #from_s3 = (modis_path.split(':')[0] == 's3')
    #modis_dataset = ModisDataset(modis_path, transform=TwoCropTransform(transforms_compose), from_s3=from_s3)
    modis_dataset = ModisDataset(modis_file, transform=TwoCropTransform(transforms_compose), data_key=opt.data_key)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
                    modis_dataset, batch_size=opt.batch_size_train, shuffle=(train_sampler is None),
                    num_workers=opt.num_workers, pin_memory=False, sampler=train_sampler)
    
    return train_loader

#@profile
def set_model(opt, cuda_use=True):
    """
    This is a function to set up the model.
    
    Args:
        opt: (Params) options for the training process.
        cude_use: (Bool) flag for the cude usage.
        
    Returns:
        model: (torch.nn.Module) model class set up by opt.
        criterion: (torch.Tensor) training loss.
    """
    model = SupConResNet(name=opt.ssl_model, 
                         feat_dim=opt.feat_dim,
                         nchannels=opt.nchannels)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)
    
    if torch.cuda.is_available() and cuda_use:
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

#@profile
def train_model(train_loader, model, criterion, optimizer, 
                epoch, opt, cuda_use=True, update_model=True):
    """
    one epoch training.

    Also used for validation with update_model=False
    
    Args:
        train_loader: (Dataloader) data loader for the training process.
        model: (torch.nn.Module)
        criterion: (torch.nn.Module) loss of the training model.
        optimizer (can be None):
        epoch: (int)
        opt: (Params)
        update_mode: (bool)
            Update the model.  Should be True unless you are running
            on the valid dataset

    Returns:
        losses.avg: (float32) 
        losses_step_list: (list)
        losses_avg_list: (list)
        criterion: (torch.Tensor) loss of the training model.
    """
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    losses_avg_list, losses_step_list = [], []

    for idx, images in enumerate(train_loader):
        data_time.update(time.time() - end)
        print(f"Data time: {data_time.val:.3f} ({data_time.avg:.3f})")

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available() and cuda_use:
            images = images.cuda(non_blocking=True)
            #labels = labels.cuda(non_blocking=True)
        bsz = images.shape[0] // 2

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        #embed(header='456 of train_util')
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.ssl_method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.ssl_method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.ssl_method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        if update_model:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # record losses
        losses_step_list.append(losses.val)
        losses_avg_list.append(losses.avg)
        
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
            
    return losses.avg, losses_step_list, losses_avg_list

def save_losses(opt, loss_train, loss_step_train, loss_avg_train,
                 loss_valid, loss_step_valid, loss_avg_valid):
    """
    Save the losses to h5 files.
    """

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
        
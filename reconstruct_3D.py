'''
Reconstruct 3D from 2D slices.
'''

import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from med_palette_2D.data import dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np

from dataset_brats import NiftiPairImageGenerator
""" try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.') """

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    """ opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file. """

    # set arguments
    dataset_folder='/projects/bcai/BraTS2021_Training_Data'
    input_modality = 'flair_poolx4'
    target_modality = 'flair'

    val_dataset = dataset.Brast_2D(
        dataset_folder = dataset_folder,
        input_modality = input_modality,
        target_modality = target_modality,
        input_size = 192, 
        depth_size = 152,
        slice_direction = 1,
        train = False
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size = 1,
        shuffle = True,
        num_workers = 1,
    ) 
    # extract the arguments
    slice_direction = val_dataset.slice_direction
    if slice_direction == 1:
        slice_len = 152
    else:
        slice_len = 192
    
    # set up the model based on the iter_95000 ckpt
    opt.load_iter = 95000
    opt.eval = True
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()
    real_Bs = []
    fake_Bs = []
    folder_path = '3d_imgs'
    for i, data in enumerate(val_dl):
        """ if i >= 1000:  # only apply our model to opt.num_test images.
            break """
        """ if i==0:
            print(np.shape(data['A']),np.shape(data['B'])) 
            # torch.Size([1, 5, 192, 192]) torch.Size([1, 1, 192, 192])
            # because DataLoader will add a new dimension at the head, which is the batch_size!!!!! """
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        real_A = model.real_A  # input image, torch.Size([1, 5, 192, 192])
        real_B = model.real_B  # gt image, torch.Size([1, 1, 192, 192])
        fake_B = model.fake_B  # output image, torch.Size([1, 1, 192, 192])
        #print(np.shape(real_B), np.shape(fake_B))
        #real_B = real_A[0,:,:,:] + 2*real_B

        # delete the batch_size dimension
        real_A = real_A[0]
        real_B = real_B[0]
        fake_B = fake_B[0]
        #print(np.shape(real_A),np.shape(fake_B))
        #torch.Size([5, 192, 192]) torch.Size([1, 192, 192])
        real_B = 2 * real_B + real_A[0:1,:,:]
        fake_B = 2 * fake_B + real_A[0:1,:,:]
        #print(np.shape(real_B),np.shape(fake_B))
        #torch.Size([1, 192, 192]) torch.Size([1, 192, 192])
        """ if i==0:
            print(np.shape(real_A)) #torch.Size([1, 5, 192, 192])
            print(np.shape(real_A[:,0,:,:])) # torch.Size([1, 192, 192])
            print(np.shape(real_A[:,0:1,:,:])) #torch.Size([1, 1, 192, 192]) """
        idx_3d = i // slice_len
        idx_slice = i % slice_len
        """ if idx_slice == 0:
            imgs = NiftiPairImageGenerator(dataset_folder, 
                                        input_modality = input_modality, 
                                        target_modality = target_modality, 
                                        input_size = 192, 
                                        depth_size = 152, 
                                        input_channel = 8,
                                        residual_training = True,
                                        train = False) """
        real_Bs.append(real_B)
        fake_Bs.append(fake_B)
        print(len(fake_Bs))
        #print(len(fake_Bs), np.shape(fake_Bs[idx_slice]))
        if idx_slice == slice_len - 1: # done with the indexed {idx_3d} 3D-img reconstruction
            real_B_3d = torch.stack(real_Bs, dim=1)
            fake_B_3d = torch.stack(fake_Bs, dim=1)
            #print(np.shape(real_B_3d),np.shape(fake_B_3d))
            #torch.Size([1, 152, 192, 192]) torch.Size([1, 152, 192, 192])
            file_path_real = os.path.join(folder_path, f'real_{idx_3d}.pth')
            file_path_fake = os.path.join(folder_path, f'fake_{idx_3d}.pth')
            torch.save(real_B_3d,file_path_real)
            torch.save(fake_B_3d,file_path_fake)
            real_Bs = []
            fake_Bs = []

        

        


    
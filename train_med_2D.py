"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from med_palette_2D.data import dataset
from torch.utils.data import DataLoader
from collections import OrderedDict

# from data_med_palette_2D/data.dataset import Brast2D

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    train_dataset = dataset.Brast_2D(
        dataset_folder='/projects/bcai/BraTS2021_Training_Data',
        input_modality = 'flair_poolx4',
        target_modality = 'flair',
        slice_direction = 1
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size = 4,
        shuffle = True,
        num_workers = 4,
    )
    """ val_dataset = dataset.Brast_2D(
        dataset_folder='/projects/bcai/BraTS2021_Training_Data',
        input_modality = 'flair_poolx4',
        target_modality = 'flair',
        slice_direction = 1,
        train = False
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size = 4,
        shuffle = True,
        num_workers = 2,
    ) """

    dataset_size = len(train_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    print('The length of training dataloader = %d' % len(train_dl))
    """ print('The number of validation images = %d' % len(val_dataset))
    print('The length of validation dataloader = %d' % len(val_dl)) """
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        #epoch_start_time = time.time()  # timer for entire epoch
        #iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_dl):  # inner loop within one epoch
            #iter_start_time = time.time()  # timer for computation per iteration
            #if total_iters % opt.print_freq == 0:
                #t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            """ if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result) """

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                #t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                model.save_networks('latest')

                ## validation
                """ with torch.no_grad():
                    print('entered validation')
                    count = 0
                    total_losses = OrderedDict()
                    for key in model.get_current_losses().keys():
                        total_losses[key] = 0.0

                    for i, data in enumerate(val_dl):

                        if count >= 1000:
                            break
                        model.set_input(data)
                        current_losses = model.get_current_losses()
                        for key, value in current_losses.items():
                            total_losses[key] += value
                        count +=  1
                    average_losses = OrderedDict()

                    for key, value in total_losses.items():     
                        average_losses[key] = value / 1000
                    print("Average losses over 1000 validation iterations:")
                    for key, value in average_losses.items():
                        print(f"{key}: {value}") """
                    

            #iter_data_time = time.time()
        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            


        print('End of epoch %d / %d' % (epoch, opt.n_epochs + opt.n_epochs_decay))

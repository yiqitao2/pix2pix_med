"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
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

""" try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.') """

def run_test(load_iter_value):
    opt.load_iter = load_iter_value
    opt.eval = True
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    total_losses = OrderedDict()
    total_losses['G_L2'] = 0

    criterionMSE = torch.nn.MSELoss()
    for i, data in enumerate(val_dl):
        """ if i >= 1000:  # only apply our model to opt.num_test images.
            break """
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        real_B = model.real_B
        fake_B = model.fake_B

        l2_loss = criterionMSE(fake_B, real_B)
        total_losses['G_L2'] += l2_loss
        if i % 1000 == 0:
            print(f'loss at iter {i}: l2: {l2_loss}')
        #print(f'loss at iter {i+1}: l2: {l2_loss}')
    average_losses = OrderedDict()
    for key, value in total_losses.items():     
        average_losses[key] = value / len(val_dl)
    print('Average losses of validation set:')
    for key, value in average_losses.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    """ opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file. """
 
    
    val_dataset = dataset.Brast_2D(
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
    ) 
    print(len(val_dl))
    run_test(95000)
    
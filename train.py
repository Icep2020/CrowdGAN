import time
import warnings
import sys
import torch
import os
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.util import set_seed, print_current_errors

warnings.filterwarnings('ignore')

opt = TrainOptions().parse()
set_seed(opt.random_seed, len(opt.gpu_ids)>0)

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training samples = %d' % dataset_size)

model = create_model(opt)
total_steps = 0

for epoch in range(1, 999):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            print_current_errors(epoch, epoch_iter, errors, t)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

        if total_steps > opt.max_steps:
            print('End training!')
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
            sys.exit(0)


    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)


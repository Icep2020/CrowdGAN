import warnings
import os
import torch
import random
import numpy as np
from PIL import Image
from options.test_options import TestOptions

from models.crowdgan import CrowdganModel
from data.fudan_dataset import TestFetcher
warnings.filterwarnings('ignore')

def set_seed(seed,cuda):
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].detach().cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

# Param loading ...
opt = TestOptions().parse()
set_seed(opt.random_seed, len(opt.gpu_ids)>0)
assert(len(opt.gpu_ids) == 1)

model = CrowdganModel()

model.initialize(opt)
model.netG.load_state_dict(torch.load(opt.netG_ckpt))
model.mapG.load_state_dict(torch.load(opt.mapG_ckpt))
model.flowG.load_state_dict(torch.load(opt.flowG_ckpt))

# Iterative prediction
model = model.eval()

fetcher = TestFetcher('samples', opt.n_frames_G, opt.logPara)

for i,val_data in enumerate(fetcher):
    model.set_input(val_data)
    model.forward()
    fake_numpy = tensor2im(model.fake)
    # save
    image_name = val_data['I_path']
    save_path = os.path.join(fetcher.results_dir, 'output_'+ '%03d' %(i+opt.n_frames_G)+'.bmp')
    print(save_path)
    image_pil = Image.fromarray(fake_numpy)
    image_pil.save(save_path)

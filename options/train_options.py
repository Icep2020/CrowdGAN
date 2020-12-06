from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--max_steps', type=int, default=200000, help='# of iterations')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
        self.parser.add_argument('--lambda_GAN', type=float, default=3.0, help='weight of GAN loss')
        self.parser.add_argument('--lambda_GAN_T', type=float, default=0.5, help='weight for temporal GAN loss')
        self.parser.add_argument('--lambda_F', type=float, default=1.0, help='weight for flow loss')
        self.parser.add_argument('--lambda_W', type=float, default=2.0, help='weight for warp loss')
        self.parser.add_argument('--lambda_dw', type=float, default=1.0, help='weight for density map warp loss')

        self.parser.add_argument('--n_frames_D', type=int, default=4, help='number of frames to feed into temporal discriminator')
        self.parser.add_argument('--max_t_step', type=int, default=10, help='max spacing between neighboring sampled frames. If greater than 1, the network may randomly skip frames during training.')
        self.parser.add_argument('--min_t_step', type=int, default=5, help='min spacing between neighboring sampled frames.')

        self.parser.add_argument('--pool_size', type=int, default=40, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--DG_ratio', type=int, default=1, help='how many times for D training after training G once')

        self.isTrain = True

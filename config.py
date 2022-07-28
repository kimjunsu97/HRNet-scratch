import os
import os.path as osp


class Config:
    
    ## dataset
    # training set
    # 2D: MSCOCO 
    trainset = 'MSCOCO'

    # testing set
    testset = 'MSCOCO'

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
 
    ## model setting
    resnet_type = 50 # 50, 101, 152
    
    ## input, output
    input_shape = (256, 256)
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)
    bbox_real = (2000, 2000) # Human36M, MuCo, MuPoTS: (2000, 2000), PW3D: (2, 2)

    ## training config
    lr_dec_epoch = [17]
    end_epoch = 20
    lr = 1e-3
    lr_dec_factor = 10
    batch_size = 32

    ## testing config
    test_batch_size = 32
    use_gt_bbox = True

    ## others
    num_thread = 8
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()
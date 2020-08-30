### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from pix2pixHD.data.base_dataset import BaseDataset, get_params, get_transform, normalize
from pix2pixHD.data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        dir_A = '_stick'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        A_paths = sorted(make_dataset(self.dir_A))

        self.A_paths_pre=A_paths[0:-1]
        self.A_paths=A_paths[1:]

        ### input B (real images)
        if opt.isTrain:
            # self.A_paths_pre = A_paths[0:-4]
            # self.A_paths = A_paths[4:]

            dir_B = '_image'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

            self.B_paths_pre = self.B_paths[0:-1]
            self.B_paths = self.B_paths[1:]
            assert len(self.A_paths) == len(self.B_paths)

        ##background
        if opt.background:
            self.bg_path=opt.bg_path

        self.dataset_size = len(self.A_paths)
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)

        A_path_pre = self.A_paths_pre[index]
        A_pre = Image.open(A_path_pre)

        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))
        A_tensor_pre = transform_A(A_pre.convert('RGB'))


        B_tensor = B_tensor_pre=bg_tensor= 0.
        ### input B (real images)
        if self.opt.isTrain:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')

            B_path_pre = self.B_paths_pre[index]
            B_pre = Image.open(B_path_pre).convert('RGB')

            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)
            B_tensor_pre = transform_B(B_pre)

        ##background
        if self.opt.background:
            bg_img=Image.open(self.bg_path).convert('RGB')
            transform_C = get_transform(self.opt, params)
            bg_tensor=transform_C(bg_img)

        input_dict = {'label': A_tensor, 'label_pre': A_tensor_pre, 'image': B_tensor,
                      'image_pre': B_tensor_pre, 'bg':bg_tensor,'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
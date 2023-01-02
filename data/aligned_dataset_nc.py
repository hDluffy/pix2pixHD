import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, add_aug_transform
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.domain_num=opt.domain_num
        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        self.domain_paths = []

        if opt.isTrain:
            for i in range(self.domain_num):
                dir_domain = '_D' + str(i)
                self.dir_domain = os.path.join(opt.dataroot, opt.phase + dir_domain)  
                self.domain_paths.append(sorted(make_dataset(self.dir_domain)))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            if self.opt.use_online_aug:
                transform_A = add_aug_transform(self.opt, transform_A, "train_A")
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        inst_tensor = feat_tensor = 0
        D_tensors = []
        if self.opt.isTrain:
            for i in range(self.domain_num):
                domain_path = self.domain_paths[i][index]   
                domain = Image.open(domain_path).convert('RGB')
                transform_B = get_transform(self.opt, params)
                if self.opt.use_online_aug:
                    transform_B = add_aug_transform(self.opt, transform_B, "train_D")      
                D_tensors.append(transform_B(domain))

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'domains': D_tensors, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
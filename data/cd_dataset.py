from .transform import Transforms
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def make_dataset(dir):
    img_paths = []
    names = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            img_paths.append(path)
            names.append(fname)

    return img_paths, names

class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt

        if 'SYSU' in opt.dataset:
            folder1, folder2 = 'time1', 'time2'
        elif 'S2Looking' in opt.dataset:
            folder1, folder2 = 'Image1', 'Image2'
        else:
            folder1, folder2 = 'A', 'B'
            
        self.dir1 = os.path.join(opt.dataroot, opt.dataset, opt.phase, folder1)
        t1_paths, t1_names = make_dataset(self.dir1)
        
        self.dir2 = os.path.join(opt.dataroot, opt.dataset, opt.phase, folder2)
        t2_paths, t2_names = make_dataset(self.dir2)
        
        self.dir_label = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'label')
        label_paths, label_names = make_dataset(self.dir_label)
        
        t1_names_set = set(t1_names)
        t2_names_set = set(t2_names)
        label_names_set = set(label_names)
        
        common_names = t1_names_set & t2_names_set & label_names_set
        common_names = sorted(list(common_names))
        
        print(f"Original file count: A={len(t1_names)}, B={len(t2_names)}, label={len(label_names)}")
        print(f"Common file count: {len(common_names)}")
        
        self.t1_paths = []
        self.t2_paths = []
        self.label_paths = []
        self.fnames = []
        
        t1_name_to_path = {name: path for path, name in zip(t1_paths, t1_names)}
        t2_name_to_path = {name: path for path, name in zip(t2_paths, t2_names)}
        label_name_to_path = {name: path for path, name in zip(label_paths, label_names)}
        
        for name in common_names:
            self.t1_paths.append(t1_name_to_path[name])
            self.t2_paths.append(t2_name_to_path[name])
            self.label_paths.append(label_name_to_path[name])
            self.fnames.append(name)
        
        self.dataset_size = len(self.t1_paths)
        
        assert len(self.t1_paths) == len(self.t2_paths) == len(self.label_paths), \
            f"Dataset size mismatch: A={len(self.t1_paths)}, B={len(self.t2_paths)}, label={len(self.label_paths)}"
        
        print(f"Final dataset size: {self.dataset_size}")

        self.normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform = transforms.Compose([Transforms(size=256)])
        self.resize = transforms.Resize((256, 256))
        self.resize_label = transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        t1_path = self.t1_paths[index]
        fname = self.fnames[index]
        img1 = Image.open(t1_path)

        t2_path = self.t2_paths[index]
        img2 = Image.open(t2_path)

        label_path = self.label_paths[index]
        label = np.array(Image.open(label_path)) // 255
        cd_label = Image.fromarray(label)

        if self.opt.phase == 'train':
            _data = self.transform({'img1': img1, 'img2': img2, 'cd_label': cd_label})
            img1, img2, cd_label = _data['img1'], _data['img2'], _data['cd_label']
        else:
            img1 = self.resize(img1)
            img2 = self.resize(img2)
            cd_label = self.resize_label(cd_label)

        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        cd_label = torch.from_numpy(np.array(cd_label))
        input_dict = {'img1': img1, 'img2': img2, 'cd_label': cd_label, 'fname': fname}

        return input_dict


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=opt.phase=='train',
                                                      pin_memory=True,
                                                      drop_last=opt.phase=='train',
                                                      num_workers=int(opt.num_workers)
                                                      )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

class LiTSDataloader(Dataset):
    def __init__(self,dir_csv, label_id):
        """
        :param csv_file:path to all the images
        """
        self.image_dirs = pd.read_csv(dir_csv,header=None).iloc[:, :].values#from DataFrame to array
        self.label_id = label_id

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, item):
        seg = sitk.ReadImage(self.image_dirs[item][0])
        target = (sitk.GetArrayFromImage(seg) >= self.label_id).astype(np.float32)
        seg_array = np.expand_dims(target, 0)
        map_array = np.expand_dims(np.load(self.image_dirs[item][1]), 0)  # C=2,DHW
        origin = torch.Tensor(seg.GetOrigin())
        direction = torch.Tensor(seg.GetDirection())
        space = torch.Tensor(seg.GetSpacing())
        prefix = self.image_dirs[item][0].split('/')[-2]
        subNo = self.image_dirs[item][0].split('/')[-1][:-4]
        return (seg_array, map_array, origin, direction, space, prefix, subNo)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

class LiTSDataloader(Dataset):
    def __init__(self,dir_csv,label_id):
        """
        :param csv_file:path to all the images
        """
        self.image_dirs = pd.read_csv(dir_csv,header=None).iloc[:, :].values#from DataFrame to array
        self.label_id = label_id

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, item):
        # (NCDHW),C must be added and N must be removed since added byitself
        ct = sitk.ReadImage(self.image_dirs[item][0])
        ct_array = np.expand_dims(sitk.GetArrayFromImage(ct), 0)
        seg_load = sitk.GetArrayFromImage(sitk.ReadImage(self.image_dirs[item][1]))
        target = (seg_load >= self.label_id).astype(np.float32)
        seg_array = np.expand_dims(target, 0)
        map_array = np.expand_dims(np.load(self.image_dirs[item][2]), 0)#C=2,DHW

        origin = torch.Tensor(ct.GetOrigin())
        direction = torch.Tensor(ct.GetDirection())
        space = torch.Tensor(ct.GetSpacing())
        prefix = self.image_dirs[item][0].split('/')[-2]
        subNo = self.image_dirs[item][0].split('/')[-1][:-4]
        return (ct_array,seg_array,map_array,origin,direction,space,prefix,subNo)
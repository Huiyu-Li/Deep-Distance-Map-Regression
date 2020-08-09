import os
import shutil
import time
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import generate_binary_structure
import warnings
warnings.simplefilter("ignore")
from utils import *

# odinary one channel distance map
def seg_distanceMap(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)

    for i in range(1):#131
        i = 4
        prefix = 'segmentation-' + str(i)
        print(prefix)
        if os.path.isdir(os.path.join(seg_path,prefix)):#for positive negtive
            sublist = os.listdir(os.path.join(seg_path,prefix))
            num_sub = len(sublist)
            for j in range(num_sub):#
                seg_name = os.path.join(seg_path, prefix, sublist[j])
                seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
                target1 = (seg_array >= label_id)*1.0 #foreground
                disMap1 = distance_transform_edt(target1).astype(np.float32)# foreground

                new_path = os.path.join(saved_path, prefix)
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                saved_name = os.path.join(new_path, sublist[j].replace('nii','npy'))
                #(seg,map)is couple,so need to make sure the name is matching!!!
                print(seg_name)
                print(saved_name)
                np.save(saved_name, disMap1)

# Norm one channel distance map
def seg_NormDistanceMap(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)

    s = generate_binary_structure(3, 3)
    for i in range(1):#131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path,prefix))
        num_sub = len(sublist)
        for j in range(num_sub):
            seg_name = os.path.join(seg_path, prefix, sublist[j])
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
            target1 = (seg_array >= label_id)*1.0#foreground
            disMap1 = distance_transform_edt(target1).astype(np.float32)  # foreground

            label_lesion,num_label = label(target1, structure=s, output=np.int16)#[0]
            # check if label_id==0 is background or not
            if not target1[label_lesion==0].all()==0:
                assert ValueError('label_id==0 is foreground')
            print('num_label=',num_label)
            for id in range(1,num_label+1,1):
                local = np.where(label_lesion==id)
                localMax = disMap1[local].max()
                disMap1[local]/=localMax #care 0!!
            print('max',disMap1.max())

            new_path = os.path.join(saved_path, prefix)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            saved_name = os.path.join(new_path, sublist[j].replace('nii', 'npy'))
            # (seg,map)is couple,so need to make sure the name is matching!!!
            print(seg_name)
            print(saved_name)
            np.save(saved_name, disMap1)

# inverse one channel distance map
# local maximum + 1 - local
def seg_InverseDistanceMap(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)
    s = generate_binary_structure(3, 3)

    for i in range(131):#131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path,prefix))
        num_sub = len(sublist)
        for j in range(num_sub):
            seg_name = os.path.join(seg_path, prefix, sublist[j])
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
            target1 = (seg_array >= label_id)*1.0#foreground
            disMap1 = distance_transform_edt(target1).astype(np.float32)  # foreground

            label_lesion,num_label = label(target1, structure=s, output=np.int16)#[0]
            # check if label_id==0 is background or not
            if not target1[label_lesion==0].all()==0:
                assert ValueError('label_id==0 is foreground')
            print('num_label=',num_label)
            for id in range(1,num_label+1,1):
                local = np.where(label_lesion==id)
                localMax = disMap1[local].max()
                disMap1[local] = localMax+1 - disMap1[local]
            print('max',disMap1.max())

            new_path = os.path.join(saved_path, prefix)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            saved_name = os.path.join(new_path, sublist[j].replace('nii', 'npy'))
            # (seg,map)is couple,so need to make sure the name is matching!!!
            print(seg_name)
            print(saved_name)
            np.save(saved_name, disMap1)

# norm inverse one channel distance map
'''
# 1/local
def seg_NormInverseDistanceMap(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)
    s = generate_binary_structure(3, 3)

    for i in range(131):#131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path,prefix))
        num_sub = len(sublist)
        for j in range(num_sub):
            seg_name = os.path.join(seg_path, prefix, sublist[j])
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
            target1 = (seg_array >= label_id)*1.0#foreground
            disMap1 = distance_transform_edt(target1).astype(np.float32)  # foreground

            label_lesion,num_label = label(target1, structure=s, output=np.int16)#[0]
            # check if label_id==0 is background or not
            if not target1[label_lesion==0].all()==0:
                assert ValueError('label_id==0 is foreground')
            print('num_label=',num_label)
            for id in range(1,num_label+1,1):
                local = np.where(label_lesion==id)
                disMap1[local] = 1./disMap1[local]
            print('max',disMap1.max())

            new_path = os.path.join(saved_path, prefix)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            saved_name = os.path.join(new_path, sublist[j].replace('nii', 'npy'))
            # (seg,map)is couple,so need to make sure the name is matching!!!
            print(seg_name)
            print(saved_name)
            np.save(saved_name, disMap1)
'''
# inverse = maximum + 1 - local
def seg_NormInverseDistanceMap(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)
    s = generate_binary_structure(3, 3)

    for i in range(131):#131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path,prefix))
        num_sub = len(sublist)
        for j in range(num_sub):#
            seg_name = os.path.join(seg_path, prefix, sublist[j])
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
            #target1 = (seg_array >= label_id)*1.0#foreground
            target1 = (seg_array == label_id) * 1.0  # foreground
            disMap1 = distance_transform_edt(target1).astype(np.float32)  # foreground

            label_lesion,num_label = label(target1, structure=s, output=np.int16)#[0]
            # check if label_id==0 is background or not
            if not target1[label_lesion==0].all()==0:
                assert ValueError('label_id==0 is foreground')
            print('num_label=',num_label)
            for id in range(1,num_label+1,1):
                local = np.where(label_lesion==id)
                localMax = disMap1[local].max()
                disMap1[local] = localMax+1 - disMap1[local]
                disMap1[local] /= localMax  # care 0!!
            print('max', disMap1.max())

            new_path = os.path.join(saved_path, prefix)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            saved_name = os.path.join(new_path, sublist[j].replace('nii', 'npy'))
            # (seg,map)is couple,so need to make sure the name is matching!!!
            print(seg_name)
            print(saved_name)
            np.save(saved_name, disMap1)
# norm inverse two channel distance map
def seg_BiNormInverseDistanceMap(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)
    s = generate_binary_structure(3, 3)

    for i in range(131):#131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path,prefix))
        num_sub = len(sublist)
        for j in range(num_sub):
            seg_name = os.path.join(seg_path, prefix, sublist[j])
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
            target1 = (seg_array >= label_id)*1.0#foreground
            disMap1 = distance_transform_edt(target1).astype(np.float32)  # foreground
            disMap0 = distance_transform_edt(np.logical_not(target1)).astype(np.float32)# background

            label_lesion,num_label = label(target1, structure=s, output=np.int16)#[0]
            # check if label_id==0 is background or not
            if not target1[label_lesion==0].all()==0:
                assert ValueError('label_id==0 is foreground')
            print('num_label=',num_label)
            for id in range(1,num_label+1,1):
                local = np.where(label_lesion==id)
                localMax = disMap1[local].max()
                disMap1[local] = localMax+1 - disMap1[local]
                disMap1[local] /= localMax  # care 0!!
            print('max', disMap1.max())
            # check if label_id==0 is background or not
            localMax0 = disMap0.max()
            local0 = np.where(label_lesion == 0)
            localMax00 = disMap1[local0].max()
            if localMax0 != localMax00:
                assert ValueError('localMax0 != localMax00')
            # disMap0 no need to inverse
            disMap0[local0] /= localMax0  # care 0!!
            print('max', disMap0.max())
            disMap = [disMap0,disMap1]# channal 0 is background

            new_path = os.path.join(saved_path, prefix)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            saved_name = os.path.join(new_path, sublist[j].replace('nii', 'npy'))
            # (seg,map)is couple,so need to make sure the name is matching!!!
            print(seg_name)
            print(saved_name)
            np.save(saved_name, disMap)

# sign distance map
def seg_SignDistanceMap(label_id,seg_path,saved_path):
    #Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)

    for i in range(10):#131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        if os.path.isdir(os.path.join(seg_path,prefix)):#for positive negtive
            sublist = os.listdir(os.path.join(seg_path,prefix))
            num_sub = len(sublist)
            for j in range(num_sub):
                seg_name = os.path.join(seg_path, prefix, sublist[j])
                seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
                target1 = (seg_array >= label_id)*1.0 #foreground
                disMap1 = distance_transform_edt(target1).astype(np.float32)# foreground
                disMap0 = distance_transform_edt(np.logical_not(target1)).astype(np.float32)#background
                disMap = disMap1 - disMap0

                new_path = os.path.join(saved_path, prefix)
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                saved_name = os.path.join(new_path, sublist[j].replace('nii','npy'))
                #(seg,map)is couple,so need to make sure the name is matching!!!
                print(seg_name)
                print(saved_name)
                np.save(saved_name, disMap)
# sign norm distance map
def seg_SignNormDistanceMap(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)
    s = generate_binary_structure(3, 3)

    for i in range(131):#131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path,prefix))
        num_sub = len(sublist)
        for j in range(num_sub):
            seg_name = os.path.join(seg_path, prefix, sublist[j])
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
            target1 = (seg_array >= label_id)*1.0#foreground
            disMap1 = distance_transform_edt(target1).astype(np.float32)  # foreground
            disMap0 = distance_transform_edt(np.logical_not(target1)).astype(np.float32)# background

            label_lesion,num_label = label(target1, structure=s, output=np.int16)#[0]
            # check if label_id==0 is background or not
            if not target1[label_lesion==0].all()==0:
                assert ValueError('label_id==0 is foreground')
            print('num_label=',num_label)
            for id in range(1,num_label+1,1):
                local = np.where(label_lesion==id)
                localMax = disMap1[local].max()
                disMap1[local] /= localMax # care 0!!
            print('max', disMap1.max())

            # check if label_id==0 is background or not
            localMax0 = disMap0.max()
            local0 = np.where(label_lesion == 0)
            localMax00 = disMap1[local0].max()
            if localMax0 != localMax00 and disMap0[local0].sum()!= disMap0.sum():
                assert ValueError('localMax0 != localMax00')
            disMap0[local0] /= localMax0 #care 0!!
            disMap = disMap1 - disMap0

            new_path = os.path.join(saved_path, prefix)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            saved_name = os.path.join(new_path, sublist[j].replace('nii', 'npy'))
            # (seg,map)is couple,so need to make sure the name is matching!!!
            print(seg_name)
            print(saved_name)
            np.save(saved_name, disMap)
# sign inverse disatance map
def seg_SignInverseDistanceMap(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)
    s = generate_binary_structure(3, 3)

    for i in range(10):  # 131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path, prefix))
        num_sub = len(sublist)
        for j in range(num_sub):
            seg_name = os.path.join(seg_path, prefix, sublist[j])
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
            target1 = (seg_array >= label_id) * 1.0  # foreground
            disMap1 = distance_transform_edt(target1).astype(np.float32)  # foreground
            disMap0 = distance_transform_edt(np.logical_not(target1)).astype(np.float32)  # background

            label_lesion, num_label = label(target1, structure=s, output=np.int16)  # [0]
            # check if label_id==0 is background or not
            if not target1[label_lesion == 0].all() == 0:
                assert ValueError('label_id==0 is foreground')
            print('num_label=', num_label)
            for id in range(1, num_label + 1, 1):
                local = np.where(label_lesion == id)
                localMax = disMap1[local].max()
                disMap1[local] = localMax + 1 - disMap1[local]

            # disMap0 no need to inverse
            disMap = disMap1 - disMap0

            new_path = os.path.join(saved_path, prefix)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            saved_name = os.path.join(new_path, sublist[j].replace('nii', 'npy'))
            # (seg,map)is couple,so need to make sure the name is matching!!!
            print(seg_name)
            print(saved_name)
            np.save(saved_name, disMap)
'''
# sign norm inverse distance map(norm = 1/)
def seg_SignNormInverseDistanceMap(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)
    s = generate_binary_structure(3, 3)

    for i in range(10):#131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path,prefix))
        num_sub = len(sublist)
        for j in range(num_sub):
            seg_name = os.path.join(seg_path, prefix, sublist[j])
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
            target1 = (seg_array >= label_id)*1.0#foreground
            disMap1 = distance_transform_edt(target1).astype(np.float32)  # foreground
            disMap0 = distance_transform_edt(np.logical_not(target1)).astype(np.float32)# background

            label_lesion,num_label = label(target1, structure=s, output=np.int16)#[0]
            # check if label_id==0 is background or not
            if not target1[label_lesion==0].all()==0:
                assert ValueError('label_id==0 is foreground')
            print('num_label=',num_label)
            for id in range(1,num_label+1,1):
                local = np.where(label_lesion==id)
                disMap1[local] = 1./disMap1[local]
            print('max', disMap1.max())

            # check if label_id==0 is background or not
            local0 = np.where(label_lesion == 0)
            # disMap0 no need to inverse?????
            local = np.where(label_lesion == id)
            localMax = disMap1[local].max()
            disMap1[local] = localMax + 1 - disMap1[local]
            disMap1[local] /= localMax  # care 0!!

            disMap0[local0] = 1./disMap0[local0]# care 0!!
            print('max', disMap0.max())
            disMap = disMap1 - disMap0

            new_path = os.path.join(saved_path, prefix)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            saved_name = os.path.join(new_path, sublist[j].replace('nii', 'npy'))
            # (seg,map)is couple,so need to make sure the name is matching!!!
            print(seg_name)
            print(saved_name)
            np.save(saved_name, disMap)
'''
# sign norm inverse distance map (norm = max+1-)
def seg_SignNormInverseDistanceMap(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)
    s = generate_binary_structure(3, 3)

    for i in range(1):#131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path,prefix))
        num_sub = len(sublist)
        for j in range(num_sub):
            seg_name = os.path.join(seg_path, prefix, sublist[j])
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
            target1 = (seg_array >= label_id)*1.0#foreground
            disMap1 = distance_transform_edt(target1).astype(np.float32)  # foreground
            disMap0 = distance_transform_edt(np.logical_not(target1)).astype(np.float32)# background

            label_lesion,num_label = label(target1, structure=s, output=np.int16)#[0]
            # check if label_id==0 is background or not
            if not target1[label_lesion==0].all()==0:
                assert ValueError('label_id==0 is foreground')
            print('num_label=',num_label)
            for id in range(1,num_label+1,1):
                local = np.where(label_lesion==id)
                localMax = disMap1[local].max()
                disMap1[local] = localMax+1 - disMap1[local]
                disMap1[local] /= localMax  # care 0!!
            print('max', disMap1.max())
            # check if label_id==0 is background or not
            localMax0 = disMap0.max()
            local0 = np.where(label_lesion == 0)
            localMax00 = disMap1[local0].max()
            if localMax0 != localMax00:
                assert ValueError('localMax0 != localMax00')
            # disMap0 no need to inverse
            disMap0[local0] /= localMax0  # care 0!!
            print('max', disMap0.max())
            disMap = disMap1 - disMap0

            new_path = os.path.join(saved_path, prefix)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            saved_name = os.path.join(new_path, sublist[j].replace('nii', 'npy'))
            # (seg,map)is couple,so need to make sure the name is matching!!!
            print(seg_name)
            print(saved_name)
            np.save(saved_name, disMap)
# sign norm inverse distance map (norm = max+1-) for fg and bg
def seg_SignNormInverseDistanceMap2(label_id,seg_path,saved_path):
    # Clear saved dir
    if os.path.exists(saved_path) is True:
        shutil.rmtree(saved_path)
    os.makedirs(saved_path)
    s = generate_binary_structure(3, 3)

    for i in range(131):#131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path,prefix))
        num_sub = len(sublist)
        for j in range(num_sub):
            seg_name = os.path.join(seg_path, prefix, sublist[j])
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(seg_name, sitk.sitkFloat32))
            target1 = (seg_array >= label_id)*1.0#foreground
            disMap1 = distance_transform_edt(target1).astype(np.float32)  # foreground
            disMap0 = distance_transform_edt(np.logical_not(target1)).astype(np.float32)# background

            label_lesion,num_label = label(target1, structure=s, output=np.int16)#[0]
            # check if label_id==0 is background or not
            if not target1[label_lesion==0].all()==0:
                assert ValueError('label_id==0 is foreground')
            print('num_label=',num_label)
            for id in range(1,num_label+1,1):
                local = np.where(label_lesion==id)
                localMax = disMap1[local].max()
                disMap1[local] = localMax+1 - disMap1[local]
                disMap1[local] /= localMax  # care 0!!
            print('max', disMap1.max())
            # check if label_id==0 is background or not
            localMax0 = disMap0.max()
            local0 = np.where(label_lesion == 0)
            localMax00 = disMap1[local0].max()
            if localMax0 != localMax00:
                assert ValueError('localMax0 != localMax00')
            # disMap0 no need to inverse
            disMap0[local0] = localMax0 + 1 - disMap0[local0]
            disMap0[local0] /= localMax0  # care 0!!
            print('max', disMap0.max())
            disMap = disMap1 - disMap0

            new_path = os.path.join(saved_path, prefix)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            saved_name = os.path.join(new_path, sublist[j].replace('nii', 'npy'))
            # (seg,map)is couple,so need to make sure the name is matching!!!
            print(seg_name)
            print(saved_name)
            np.save(saved_name, disMap)
# check
def check_distanceMap(label_id,seg_path,map_path,check_path):
    # Clear saved dir
    if os.path.exists(check_path) is True:
        shutil.rmtree(check_path)
    os.makedirs(check_path)

    for i in range(1):# 131
        prefix = 'segmentation-' + str(i)
        print(prefix)
        sublist = os.listdir(os.path.join(seg_path, prefix))
        num_sub = len(sublist)
        for j in range(num_sub):
            seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(seg_path, prefix, sublist[j]), sitk.sitkFloat32))
            # target1 = (seg_array >= label_id) * 1.0  # foreground
            target1 = seg_array  # foreground
            disMap = np.load(os.path.join(map_path, prefix, sublist[j].replace('nii','npy')))
            #special check for NegtiveTumorPatchesMap
            # if (disMap>0).any():
            #     print('NegtiveTumorPatchesMap>0:',os.path.join(map_path, prefix, sublist[j]))
            new_path = os.path.join(check_path, prefix)
            if not os.path.exists(new_path):
                os.makedirs(new_path)

            z = np.any(disMap, axis=(1, 2))
            if z.any():
                start_slice, end_slice = np.where(z)[0][[0, -1]]
                for k in range(start_slice,end_slice,1):
                    plt.figure()
                    plt.subplot(121);plt.imshow(target1[0]);plt.axis('off');plt.title('target');plt.colorbar()
                    plt.subplot(122);plt.imshow(disMap[0]);plt.axis('off');plt.title('map');plt.colorbar()
                    # below for bi DM
                    # plt.subplot(132);plt.imshow(disMap[0,0]);plt.axis('off');plt.title('map0');plt.colorbar()
                    # plt.subplot(133);plt.imshow(disMap[1,0]);plt.axis('off');plt.title('map1');plt.colorbar()
                    # print((disMap[0,0]-disMap[1,0]).sum())
                    plt.savefig(os.path.join(new_path, sublist[j][:-4] +'-'+str(k) + '.png'))
                    plt.close()

if __name__ == '__main__':
    start_time = time.time()
    sys.stdout = Logger('./printLog_map')  # see utils.py
    seg_path = "/data/lihuiyu/LiTS/Preprocessed_S3_W20040_48/seg/"
    label_id = 1

    print('####seg_SignNormDistanceMap')
    map_path = "/data/lihuiyu/LiTS/Preprocessed_S3_W20040_48/SignNormInverse2_liver/"
    seg_SignNormInverseDistanceMap2(label_id, seg_path, map_path)
    # check_path = "/data/lihuiyu/LiTS/Valid_results/DecideMap/SignNormDistanceMap/"
    # check_distanceMap(label_id, seg_path, map_path, check_path)


    print('Time {:.3f} min'.format((time.time() - start_time) / 60))
    print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))
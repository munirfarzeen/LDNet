import os

import torch
import numpy as np
import matplotlib.pyplot as plt


def listFiles(rootdir='.', suffix='png'):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched as PNG or JPG
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]





def get_lane_labels():
    return np.array([
         #[  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153]
        ])





def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
      print()
    elif dataset == 'cityscapes':
        n_classes = 2
        label_colours = get_lane_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    
    r[label_mask == 255] = 0
    g[label_mask == 255] = 0
    b[label_mask == 255] =0
    
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
   # rgb = np.zeros((label_mask.shape[1], label_mask.shape[2], 3))
    #replace blue with red as opencv uses bgr
    rgb[:, :, 0] = r /255.0     
    rgb[:, :, 1] = g /255.0
    rgb[:, :, 2] = b /255.0
#    
#    rgb = np.zeros((label_mask.shape[1], label_mask.shape[2], 3))
#    #replace blue with red as opencv uses bgr
#    rgb[:, :, 0] = b #/255.0     
#    rgb[:, :, 1] = g #/255.0
#    rgb[:, :, 2] = r #/255.0
#    
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
def decode_segmap_cv(label_mask, dataset, plot=True):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
      print()
    elif dataset == 'lane':
        n_classes = 5
        label_colours = get_lane_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    
    r[label_mask == 255] = 0
    g[label_mask == 255] = 0
    b[label_mask == 255] =0
    
#    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
#   # rgb = np.zeros((label_mask.shape[1], label_mask.shape[2], 3))
#    #replace blue with red as opencv uses bgr
#    rgb[:, :, 0] = r /255.0     
#    rgb[:, :, 1] = g /255.0
#    rgb[:, :, 2] = b /255.0
#    
    rgb = np.zeros((label_mask.shape[1], label_mask.shape[2], 3))
    #replace blue with red as opencv uses bgr
    rgb[:, :, 0] = b #/255.0     
    rgb[:, :, 1] = g #/255.0
    rgb[:, :, 2] = r #/255.0
#    
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


    
from torchvision import transforms 

if __name__ == '__main__':
    print()
    ar=np.array([[0,7,10],[7,3,6]])
#     z=convertTrainIdToClassId(ar)
# #    img3= transforms.ToPILImage()(torch.from_numpy(ou).type(torch.FloatTensor))#.detach().cpu()
# #    img3.save(oupath)
#     print(z)
    
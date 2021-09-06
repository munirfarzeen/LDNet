# This code filters raw accumulated images with median filter to remove noise.
import cv2
from pathlib import Path
import natsort

#original_images_folder could be train/val/test
IMG_Path = Path("/media/farzeen/sabrent/sabent/raw_images_and_multiclass_labels/test/images/")
IMG_File = natsort.natsorted(list(IMG_Path.glob("*.bmp")), alg=natsort.PATH)
IMG_Str = []
for i in IMG_File:
    IMG_Str.append(str(i))

#make sure the out_prefix ends with '/'(linux) or '\'(windows)
out_prefix="/media/farzeen/sabrent/sabent/raw_images_and_multiclass_labels/test/filter_images/"
for k in range(len(IMG_Str)):
    src=cv2.imread(IMG_Str[k],1)
    # 3 is the median filter kernal size
    after_filter=cv2.medianBlur(src,3)
    out_path=out_prefix+Path(IMG_Str[k]).name
    cv2.imwrite(out_path,after_filter)
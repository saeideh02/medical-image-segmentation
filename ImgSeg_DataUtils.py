import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype("float32")  # original is uint16
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val)  # scale image to [0, 1]
    img = img * 255.0  # scale image to [0, 255]
    img = img.astype("uint8")
    return img


def rle_decode(mask_rle, shape):
    if not pd.isna(mask_rle):
        s = np.asarray(mask_rle.split(), dtype=int)
        starts = s[0::2] - 1
        lengths = s[1::2]
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)  # Needed to align to RLE direction
    else:
        return np.zeros(shape)

        
def id2mask(df):
    wh = df[["height", "width"]].iloc[0]
    shape = (int(wh.width), int(wh.height), 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(["large_bowel", "small_bowel", "stomach"]):
        cdf = df[df["class"] == class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask

    
def show_img(img, mask=None, apply_clahe=True):
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    plt.imshow(img, cmap="bone")

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [
            Rectangle((0, 0), 1, 1, color=_c)
            for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis("off")    

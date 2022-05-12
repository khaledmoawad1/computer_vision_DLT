"""
Assignment 1 : Image Stitching and Perspective Correction
Part I: DLT Algorithm
"""
import os
import numpy as np
import argparse
import cv2
from DLT import *
from norm_DLT import *
#%%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--srcdir", help = "path to the images directory")
    ap.add_argument("-n", "--gtdir", help = "path to npy groundtruth directory")
    ap.add_argument("--norm", action="store_true")
    args = ap.parse_args()

    error = 0
    files = os.listdir(args.srcdir)
    for image_path in files:
        image = cv2.imread(os.path.join(args.srcdir, image_path))
        npy_file = os.path.join(args.gtdir, image_path.split('/')[-1].replace('png', 'npy'))

        gt = np.load(npy_file, allow_pickle=True).item()
        points = gt['points']
        homography_gt = gt['homography']

        # TODO: Call your DLT methods according to norm flag
        homography, maxWidth, maxHeight = norm_DLT(points)
        im_out = cv2.warpPerspective(image, homography, (maxWidth,maxHeight))
        cv2.imshow("Warped Image", im_out)
        cv2.waitKey(0)


        error += np.linalg.norm(homography.flatten() - homography_gt.flatten())**2

    print('Total Mean Squared Error: ', error/len(files))
   
    cv2.destroyAllWindows()
    


    

if __name__ == "__main__":
    main()

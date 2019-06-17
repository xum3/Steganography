# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt


class Steganography():

    def __init__(self, cover_path, secret_path):
        self.img1 = cv2.imread(cover_path)
        self.img2 = cv2.imread(secret_path)
        if self.img1.shape != self.img2.shape:
            print("The shapes of the two pictures are different")
            exit()

    def __int2bin(self, bgr):
        b, g, r = bgr
        return ['{0:08b}'.format(b),
                '{0:08b}'.format(g),
                '{0:08b}'.format(r)]

    def __bin2int(self, bgr):
        b, g, r = bgr
        return [int(b, 2),
                int(g, 2),
                int(r, 2)]

    def __merge_rgb(self, bgr1, bgr2):
        b1, g1, r1 = bgr1
        b2, g2, r2 = bgr2
        bgr = [b1[:4] + b2[:4],
               g1[:4] + g2[:4],
               r1[:4] + r2[:4]]
        return bgr

    def merge(self):
        newImg = np.zeros(self.img1.shape, dtype=np.int32)
        for i in range(self.img1.shape[0]):
            for j in range(self.img1.shape[1]):
                bgr1 = self.__int2bin(self.img1[i, j, :])
                bgr2 = self.__int2bin(self.img2[i, j, :])
                bgr = self.__merge_rgb(bgr1, bgr2)
                newImg[i, j] = self.__bin2int(bgr)
        self.mergedImage = newImg
        return newImg

    def unmerge(self):
        img = self.mergedImage
        newImg = np.zeros(img.shape, dtype=np.int32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                b, g, r = self.__int2bin(img[i, j, :])
                bgr = [b[4:] + '0000',
                       g[4:] + '0000',
                       r[4:] + '0000']
                newImg[i, j] = self.__bin2int(bgr)
        return newImg


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Incorrect input parameter format")
        exit()
    cover_path, secret_path = sys.argv[1:]
    stegIns = Steganography(cover_path, secret_path)
    mergedImage = stegIns.merge()
    cv2.imwrite("./merged_image.png", mergedImage)
    unmergeImage = stegIns.unmerge()
    cv2.imwrite("./unmerged_image.png", unmergeImage)

    img1 = cv2.imread(cover_path)
    img2 = cv2.imread(secret_path)
    img1 = img1[:,:,::-1]
    img2 = img2[:,:,::-1]
    mergedImage = mergedImage[:,:,::-1]
    unmergeImage = unmergeImage[:,:,::-1]

    plt.imshow( np.hstack( (img1,img2,mergedImage,unmergeImage) ) )
    plt.show()

    # python .\lsb_steganography.py "./1.jpg" "2.jpg"

    
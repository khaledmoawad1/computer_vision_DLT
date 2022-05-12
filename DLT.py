# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:27:11 2022

@author: Dell
"""

import numpy as np


def DLT(points):
    n = len(points)
    
   
    
   
    # our original points
    tl,tr,br,bl = points
   
    # Calculating prime points
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))  
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    primes = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
    
    
    # Calc A (point pair matrix) (2N*9)
    A = []
    for i in range(0, n):
        x = points[i][0]
        y = points[i][1]
        x_prime = primes[i][0]
        y_prime = primes[i][1]
        
        
        
        A.append([-x, -y, -1, 0, 0, 0, x_prime*x, x_prime*y, x_prime])
        A.append([0, 0, 0, -x, -y, -1, y_prime*x, y_prime*y, y_prime])
    A = np.asarray(A)
    
    # Using SVD to take the last column as our solution
    U, S, V = np.linalg.svd(A)
    
    L = V[-1,:] / V[-1,-1]
    
    
    H = L.reshape(3, 3)
        
    
    return H, maxWidth, maxHeight




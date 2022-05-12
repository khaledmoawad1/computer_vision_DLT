# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:15:43 2022

@author: Dell
"""
import numpy as np

def norm_DLT(points):
     n = len(points)
     
     
     # our original points
     tl,tr,br,bl = points
     
     # Normalize original points
     m , s = np.mean(points, 0), np.std(points)
     #print(m,s)
     T = np.array([[np.sqrt(2)/s, 0, -m[0]/s], [0, np.sqrt(2)/s, -m[1]/s], [0, 0, 1]])
     # insted we can write
     # np.array([[s/np.sqrt(2), 0, m[0]], [0, s/np.sqrt(2), m[1]], [0, 0, 1]])
     # Then inverse it  T = np.linalg.inv(T) 
     # It will directly do the same 
     
     
     points = np.dot( T, np.concatenate( (points.T, np.ones((1,points.shape[0]))) ) )
     points = points[0:2].T
     

     
   
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
     
     
     # Normalize prime points
     m2, s2 = np.mean(primes, 0), np.std(primes)
     Tr = np.array([[np.sqrt(2)/s2, 0, -m2[0]/s2], [0, np.sqrt(2)/s2, -m2[1]/s2], [0, 0, 1]])
     #Tr = np.linalg.inv(Tr)
     primes = np.dot( Tr, np.concatenate( (primes.T, np.ones((1,primes.shape[0]))) ) )
     primes = primes[0:2].T
     
     
     
     # Calc A matrix (point pair matrix) (2N*9)
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
     
     #Denormalization to get final H
     H = np.dot( np.dot( np.linalg.pinv(Tr), H ), T )
     #print(H)
     H = H / H[-1, -1]
     # print(H)
     
     
     
     return H , maxWidth, maxHeight
      


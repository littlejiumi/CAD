# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:05:05 2019

@author: zxh
""" 

import cv2
import numpy as np
from tqdm import trange

#计算图像能量图（energy function为el(I),使用Scharr滤波）   
def calc_energy(img):
       b, g, r = cv2.split(img)
       b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
       g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
       r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
       return b_energy + g_energy + r_energy


#动态规划，寻找每个像素上可见的能量最小值，用backtrack记录路径  
def minimum_seam(img):
    r, c, x = img.shape 
    energy_map = calc_energy(img)
    M = energy_map.copy() #创建2D数组M,存储该像素上的最小能量值
    backtrack = np.zeros_like(M, dtype=np.int)
    for i in range(1, r):  #从第二行开始 
        for j in range(0, c):# 从第一列开始，且处理图像的左侧边缘，确保不会索引-1 
            # 线条上的每个像素必须在边缘或拐角处彼此相连，论文证明了不连续的情况效果很差 
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2]) # 找该像素点上面两个相邻点的最小值 
                backtrack[i, j] = idx + j      # backtrack记录该像素点之前的能量小点纵坐标，之后回溯 
                min_energy = M[i-1, idx + j]   # 得到最小能量点存入M,以空间换时间 
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2]) #找该像素点上面三个相邻点的最小值
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
            M[i, j] += min_energy  #存储该像素上可见的最小能量值 
    return M, backtrack

#移除或增加能量最小线 
def carve_seam(img):
    r, c, x = img.shape
    M, backtrack = minimum_seam(img)  
    output = np.zeros((r, c - 1, 3))
    j = np.argmin(M[-1])   # 从最后一行开始回溯，j为最后一行最小值纵坐标 
    for i in reversed(range(r)): # 从最后一行向前
        output[i, : ,0] = np.delete(img[i, : , 0], [j]) # 三个通道  
        output[i, : ,1] = np.delete(img[i, : , 1], [j])
        output[i, : ,2] = np.delete(img[i, : , 2], [j])
        j = backtrack[i, j]
    return output 

def add_seam(img):
    r, c, x = img.shape
    M, backtrack = minimum_seam(img)  
    output = np.zeros((r, c + 1, 3))
    j = np.argmin(M[-1])   # 从最后一行开始回溯，j为最后一行最小值纵坐标   
    for i in reversed(range(r)): # 从最后一行向前
        for ch in range(3):        #三个通道
            output[i, : j, ch] = img[i, : j, ch]    #其左边j-1个像素不变
            output[i, j + 1:, ch] = img[i, j:, ch]  # 其右部分依次右移一个像素点 
        j = backtrack[i, j]
    return output 
   
#纵向变化
def change_c(img, scale_c):   #scale_c为缩减后的比例 
    r, c, x = img.shape 
    new_c = int(scale_c * c)  #新的宽度 
    if scale_c < 1: 
        for i in trange(c - new_c): #通过循环每次缩减一个像素直到到新的宽度，显示进度条 
            img = carve_seam(img)
        return img
    elif scale_c >= 1:
        for i in trange(new_c - c): #通过循环每次增加一个像素直到到新的宽度，显示进度条  
            img = add_seam(img)
        ou_filename = "./htc.png"    #标注seam的图 
        cv2.imwrite(ou_filename,img)  # 保存改变后的图片
        r, c, x = img.shape 
        for i in range(r):
            for j in range(c):
                if (img[i][j]==[0,0,0]).all():  #三个通道均为黑色即为seam
                    img[i][j] = img[i][j-1]     #补全该像素 
        return img

#横向变化 
def change_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1)) # 旋转90度
    img = change_c(img, scale_r)   # 用crop_c方法
    img = np.rot90(img, 3, (0, 1))  # 旋转270度回原图 
    return img

def main():
    scale_r = float(0.95)          # 行变化比例系数
    scale_c = float(1.05)           # 列变化比例系数
    in_filename ="./ht.png"            #original
    out_filename = "./htb.png"        #result
    eng_filename = "./hte.png"       #energy_map
    img = cv2.imread(in_filename)   # 读入图片
    out = change_r(img, scale_r)    # 改变行
    out = change_c(out, scale_c)    # 改变列
    energy_map = calc_energy(out)   #保存能量图
    cv2.imwrite(out_filename, out)  # 保存改变后的图片
    cv2.imwrite(eng_filename, energy_map) # 保存能量图
    print(img.shape)                  #原始图片形状
    print(out.shape)                #改变后图片形状
    
if __name__ == '__main__':
    main()
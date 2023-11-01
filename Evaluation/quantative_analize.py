from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import normalized_root_mse
from PIL import Image
import numpy as np
import os

ori_img_path = '/media/amax/My Passport/test_result/GT_png'
compare_img_path = '/media/amax/My Passport/test_result/cyclegan/test_latest/fakeimage'
ori_img_list = os.listdir(ori_img_path)
compare_img_list = os.listdir(compare_img_path)
ori_img_list.sort()
compare_img_list.sort()
ssim_list = []
psnr_list = []
nmse_list = []
for i in range(len(ori_img_list)):
    ori_img = Image.open(os.path.join(ori_img_path, ori_img_list[i])).convert('L')
    compare_img = Image.open(os.path.join(compare_img_path, compare_img_list[i])).convert('L')
    ori_img_array = np.array(ori_img)
    compare_img_array = np.array(compare_img)
    ssim = structural_similarity(ori_img_array, compare_img_array)
    psnr = peak_signal_noise_ratio(ori_img_array, compare_img_array)
    nmse = normalized_root_mse(ori_img_array, compare_img_array)
    ssim_list.append(ssim)
    psnr_list.append(psnr)
    nmse_list.append(nmse)
    #calculate average ssim
    ssim_average = sum(ssim_list)/len(ssim_list)
    #save ssim list as txt
    np.savetxt('*.txt', ssim_list)
    np.savetxt('*.txt', psnr_list)
    np.savetxt('*.txt', nmse_list)
#load txt file and draw boxplot of ssim


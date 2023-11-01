import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图片
image_paths = ["/media/amax/My Passport/test_result/profile/CycleGAN2D.png", "/media/amax/My Passport/test_result/profile/Unet.png", "/media/amax/My Passport/test_result/profile/reDWI.png", "/media/amax/My Passport/test_result/profile/DualGAN.png", "/media/amax/My Passport/test_result/profile/CycleGAN3D.png", "/media/amax/My Passport/test_result/profile/Ournet.png"]
images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

# 定义直线
x1, y1 = 55, 210
x2, y2 = 55, 127
line_length = int(np.hypot(x2-x1, y2-y1))
xs = np.linspace(x1, x2, line_length)
ys = np.linspace(y1, y2, line_length)

# 从每张图片上提取直线上的像素值
profiles = []
for image in images:
    profile = [image[int(y)][int(x)] for x, y in zip(xs, ys)]
    profiles.append(profile)

# 颜色列表
colors = ['r', 'g', 'b', 'c', 'm', 'y']



# 绘制profile curve
for index, (profile, color) in enumerate(zip(profiles, colors)):
    plt.plot(profile, label=f"Image {index+1}", color=color)

plt.legend()
plt.title("Profile curve")
plt.xlabel("Distance along the line")
plt.ylabel("Pixel value")

# 保存profile curve
plt.savefig("profile_curve.png")

# 显示profile curve
plt.show()

with open("profile.txt", "w") as file:
    zipped_profiles = zip(*profiles)
    for values in zipped_profiles:
        file.write("\t".join([str(value) for value in values]) + "\n")
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# 图像文件路径列表
image_paths = [
    "path/to/your/image1.mha",
    "path/to/your/image2.mha",
    "path/to/your/image3.mha",
    # ...添加更多图像路径
]

# 初始化列表以存储平坦化后的图像
flattened_images = []


for path in image_paths:
    image = sitk.ReadImage(path)
    image_array = sitk.GetArrayFromImage(image)
    image_flatten = image_array.flatten()
    flattened_images.append(image_flatten)


flattened_images_array = np.array(flattened_images)
flattened_images_array = flattened_images_array.T
# Bland-Altman Plot Analysis
original_dwi = flattened_images_array[:, 0]
other_columns = [original_dwi] + [flattened_images_array[:, i] for i in range(1, flattened_images_array.shape[1])]
cmap = plt.get_cmap('Blues')

for i, column in enumerate(other_columns):
    diff = original_dwi - column
    mean = (original_dwi + column) / 2

    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    plt.figure()
    plt.scatter(mean, diff, alpha=0.5, c=mean, cmap=cmap)

    plt.axhline(mean_diff, color='black', linestyle='--')
    plt.axhline(upper_limit, color='black', linestyle='--')
    plt.axhline(lower_limit, color='black', linestyle='--')

    if i != 0:
        plt.text(min(mean), mean_diff, f'Mean={mean_diff:.2f}', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=12)
        plt.text(min(mean), upper_limit, f'+1.96 SD={upper_limit:.2f}', verticalalignment='bottom', horizontalalignment='left', color='black', fontsize=12)
        plt.text(min(mean), lower_limit, f'-1.96 SD={lower_limit:.2f}', verticalalignment='top', horizontalalignment='left', color='black', fontsize=12)

    plt.xlabel('Mean of original and comparison')
    plt.ylabel('Difference')

    if i == 0:
        plt.title('Bland-Altman Plot: Original vs. Original')
    else:
        plt.title(f'Bland-Altman Plot: Original vs. Column {i}')

    # 保存图片
    plt.savefig(f'Bland-Altman_Plot_Column_{i}.png')

    plt.show()

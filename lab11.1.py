import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, util
from skimage import filters
from skimage.morphology import binary_erosion, binary_dilation, convex_hull_image, skeletonize
from skimage.morphology import reconstruction  # 用于地质(或地形)重建函数
from scipy import ndimage as ndi


def generate_grid_image(rows=16, cols=16, block_size=4):
    """
    生成一个带有方格结构并部分填充的二值图像示例。
    rows, cols: 整张图像的行列大小
    block_size: 每个小方格的宽度/高度(正方形)
    生成规则示例:
      - 在一些方格填充为1，其余为0
      - 保证有一些连续区域和一些孤立小块，便于后续演示各种形态学操作
    """
    image = np.zeros((rows, cols), dtype=np.uint8)

    # 我们人为设置一些方块填充为1
    # 这里的做法是，按照 block_size 为单位，设置一些区域是1，一些区域是0
    for r in range(0, rows, block_size):
        for c in range(0, cols, block_size):
            # 人为定义一个简单规则：如果 (r + c) // block_size 是偶数，就填充这个方块
            if ((r + c) // block_size) % 2 == 0:
                image[r:r + block_size, c:c + block_size] = 1

    # 此外再“挖”掉一部分，用于演示空洞填充等
    # 例如在图中心附近挖一个小洞
    center_r, center_c = rows // 2, cols // 2
    hole_size = 2
    image[center_r - hole_size:center_r + hole_size, center_c - hole_size:center_c + hole_size] = 0

    # 让图像再多一个额外小方块用于后续凸包、骨架等观察
    image[2:4, 12:14] = 1

    return image


def show_image(img, title="Image", cmap='gray'):
    """简单封装的可视化函数"""
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')


def lab_11_1_boundary_extraction(binary_img):
    """
    11.1 边界提取 (Boundary Extraction)
    公式: Boundary(A) = A - Erode(A)
    """
    # 执行腐蚀操作
    eroded = binary_erosion(binary_img)
    boundary = binary_img.astype(np.uint8) - eroded.astype(np.uint8)

    fig = plt.figure(figsize=(10, 4))
    fig.suptitle("Lab 11.1 Boundary Extraction")

    plt.subplot(1, 3, 1)
    show_image(binary_img, "Original")

    plt.subplot(1, 3, 2)
    show_image(eroded, "Eroded")

    plt.subplot(1, 3, 3)
    show_image(boundary, "Boundary = A - Eroded(A)")

    plt.tight_layout()
    plt.show()


def lab_11_2_region_filling(binary_img):
    """
    11.2 区域填充 (Region Filling)
    思路：
      使用 scipy.ndimage 或 skimage 中的填洞函数来实现。
      这里演示先用 ndimage.binary_fill_holes 自动填充空洞。
    """
    filled = ndi.binary_fill_holes(binary_img)

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle("Lab 11.2 Region Filling")

    plt.subplot(1, 2, 1)
    show_image(binary_img, "Original")

    plt.subplot(1, 2, 2)
    show_image(filled, "Filled (binary_fill_holes)")

    plt.tight_layout()
    plt.show()


def lab_11_3_convex_hull(binary_img):
    """
    11.3 凸包 (Convex Hull)
    使用 skimage.morphology.convex_hull_image
    其返回一个与输入相同大小的布尔类型凸包图像。
    """
    chull = convex_hull_image(binary_img)

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle("Lab 11.3 Convex Hull")

    plt.subplot(1, 2, 1)
    show_image(binary_img, "Original")

    plt.subplot(1, 2, 2)
    show_image(chull, "Convex Hull")

    plt.tight_layout()
    plt.show()


def lab_11_4_skeleton(binary_img):
    """
    11.4 骨架 (Skeleton)
    使用 skimage.morphology.skeletonize
    skeletonize 会返回一个与原图相同大小的布尔图像，表示骨架所在位置为True。
    """
    # skeletonize 要求输入是布尔类型
    bool_img = binary_img.astype(bool)
    ske = skeletonize(bool_img)

    fig = plt.figure(figsize=(10, 4))
    fig.suptitle("Lab 11.4 Skeleton")

    plt.subplot(1, 3, 1)
    show_image(binary_img, "Original")

    plt.subplot(1, 3, 2)
    show_image(ske, "Skeleton")

    # 骨架通常很细，我们也可以用 boundary 来对比一下
    boundary = bool_img.astype(np.uint8) - binary_erosion(bool_img).astype(np.uint8)
    plt.subplot(1, 3, 3)
    show_image(boundary, "Boundary (for reference)")

    plt.tight_layout()
    plt.show()


def lab_11_5_geodesic_dilation_erosion(binary_img):
    """
    11.5 地质(或测地)膨胀 / 侵蚀 (Geodesic Dilation / Erosion)
    使用 skimage.morphology.reconstruction

    - 测地膨胀: reconstruction(marker, mask, method='dilation')
      其中 marker <= mask
    - 测地侵蚀: reconstruction(marker, mask, method='erosion')
      其中 marker >= mask
    为了演示，我们先人为定义一个marker图与mask图。

    示例如下:
      mask = 原图
      marker = 在原图的基础上进行一次腐蚀/膨胀(具体看我们想演示啥)，然后再通过重建
    """

    # 先做一个“marker < mask”的情况：先对图腐蚀做marker，再用原图作为mask，演示测地膨胀
    eroded = binary_erosion(binary_img, morphology.disk(1))
    marker = eroded
    mask = binary_img
    # 测地膨胀
    geo_dilation = reconstruction(marker.astype(np.uint8),
                                  mask.astype(np.uint8),
                                  method='dilation')

    # 另一个“marker > mask”的情况：先对图膨胀做marker，mask仍然是原图
    # 然后做测地侵蚀
    dilated = binary_dilation(binary_img, morphology.disk(1))
    marker2 = dilated
    mask2 = binary_img
    geo_erosion = reconstruction(marker2.astype(np.uint8),
                                 mask2.astype(np.uint8),
                                 method='erosion')

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Lab 11.5 Geodesic Dilation and Erosion")

    # --- 原图 ---
    plt.subplot(2, 3, 1)
    show_image(binary_img, "Original(mask)")

    # --- 测地膨胀部分 ---
    plt.subplot(2, 3, 2)
    show_image(eroded, "Marker for Dilation\n(Eroded original)")
    plt.subplot(2, 3, 3)
    show_image(geo_dilation, "Geodesic Dilation")

    # --- 测地侵蚀部分 ---
    plt.subplot(2, 3, 4)
    show_image(binary_img, "Original(mask) again")
    plt.subplot(2, 3, 5)
    show_image(dilated, "Marker for Erosion\n(Dilated original)")
    plt.subplot(2, 3, 6)
    show_image(geo_erosion, "Geodesic Erosion")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. 生成演示用二值图像
    original_img = generate_grid_image(rows=16, cols=16, block_size=4)

    # 2. Boundary Extraction
    lab_11_1_boundary_extraction(original_img)

    # 3. Region Filling
    lab_11_2_region_filling(original_img)

    # 4. Convex Hull
    lab_11_3_convex_hull(original_img)

    # 5. Skeleton
    lab_11_4_skeleton(original_img)

    # 6. Geodesic Dilation/Erosion
    lab_11_5_geodesic_dilation_erosion(original_img)

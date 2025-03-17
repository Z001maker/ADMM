import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.sparse import diags, kron, eye, vstack
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


# 1. 读取灰度图像
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("图片读取失败，请检查路径是否正确")
    return image.astype(np.float32) / 255.0


# 2. 创建梯度算子（稀疏矩阵），按块计算
def create_gradient_operator(n_rows, n_cols):
    T_v = diags([-1, 1], [0, 1], shape=(n_rows - 1, n_rows), format='csc')
    D_v = kron(eye(n_cols, format='csc'), T_v)

    T_h = diags([-1, 1], [0, 1], shape=(n_cols - 1, n_cols), format='csc')
    D_h = kron(T_h, eye(n_rows, format='csc'))

    return vstack([D_v, D_h], format='csc')


# 3. L1 近端算子（软阈值）
def prox_l1(x, lambda_factor):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_factor, 0)


def admm_denoise_block(noisy_block, lambda_param, rho=1.0, num_iter=50, original_block=None):
    n_rows, n_cols = noisy_block.shape
    D = create_gradient_operator(n_rows, n_cols)

    # 变量初始化
    vX = noisy_block.flatten()
    vZ = np.zeros(D.shape[0])  # 与 D 的行数相同
    vU = np.zeros(D.shape[0])  # 与 D 的行数相同

    I = eye(vX.shape[0], format='csc')
    C = I + rho * (D.T @ D)

    # 用于保存每次迭代的 PSNR 和 SSIM
    psnr_values = []
    ssim_values = []

    for iter in range(num_iter):
        vX = spsolve(C, (noisy_block.flatten() + rho * D.T @ (vZ - vU)).ravel())

        vZ = prox_l1(D @ vX + vU, lambda_param / rho)
        vU += D @ vX - vZ

        # 计算当前去噪结果的 PSNR 和 SSIM
        if original_block is not None:
            denoised_block = np.clip(vX.reshape(n_rows, n_cols), 0, 1)
            psnr_values.append(psnr(original_block, denoised_block, data_range=1.0))
            ssim_values.append(ssim(original_block, denoised_block, data_range=1.0))

    return np.clip(vX.reshape(n_rows, n_cols), 0, 1), psnr_values, ssim_values


# 4. **分块去噪（逐块计算减少内存占用）**
def blockwise_denoising(noisy_image, block_size=64, overlap=16, lambda_param=0.1, rho=1.0, num_iter=50, original_image=None):
    h, w = noisy_image.shape
    denoised_image = np.zeros_like(noisy_image)
    weight_mask = np.zeros_like(noisy_image)

    # 用于保存所有块的 PSNR 和 SSIM
    all_psnr_values = []
    all_ssim_values = []

    step = block_size - overlap
    for i in range(0, h, step):  # 修改为从 0 到 h，步长为 step
        for j in range(0, w, step):  # 修改为从 0 到 w，步长为 step
            # 计算块的起始和结束位置
            row_start = i
            row_end = min(i + block_size, h)  # 确保不超过图像边界
            col_start = j
            col_end = min(j + block_size, w)  # 确保不超过图像边界

            # 提取当前块
            block = noisy_image[row_start:row_end, col_start:col_end]

            # 如果块的大小小于 block_size，填充到 block_size
            if block.shape[0] < block_size or block.shape[1] < block_size:
                padded_block = np.zeros((block_size, block_size), dtype=noisy_image.dtype)
                padded_block[:block.shape[0], :block.shape[1]] = block
                block = padded_block

            # 提取对应的原始图像块
            original_block = None
            if original_image is not None:
                original_block = original_image[row_start:row_end, col_start:col_end]
                if original_block.shape[0] < block_size or original_block.shape[1] < block_size:
                    padded_original_block = np.zeros((block_size, block_size), dtype=original_image.dtype)
                    padded_original_block[:original_block.shape[0], :original_block.shape[1]] = original_block
                    original_block = padded_original_block

            # 对当前块进行去噪
            denoised_block, psnr_values, ssim_values = admm_denoise_block(block, lambda_param, rho, num_iter, original_block)

            # 将去噪后的块放回原图（只放回有效区域）
            denoised_image[row_start:row_end, col_start:col_end] += denoised_block[:row_end-row_start, :col_end-col_start]
            weight_mask[row_start:row_end, col_start:col_end] += 1

            # 保存当前块的 PSNR 和 SSIM
            if original_block is not None:
                all_psnr_values.append(psnr_values)
                all_ssim_values.append(ssim_values)

    # 对重叠区域进行加权平均
    denoised_image = np.divide(denoised_image, weight_mask, where=(weight_mask > 0))

    # 计算所有块的平均 PSNR 和 SSIM
    if original_image is not None:
        avg_psnr_values = np.mean(np.array(all_psnr_values), axis=0)
        avg_ssim_values = np.mean(np.array(all_ssim_values), axis=0)
        return denoised_image, avg_psnr_values, avg_ssim_values
    else:
        return denoised_image, None, None


# 5. 保存图像
def save_image(image, save_path):
    cv2.imwrite(save_path, (image * 255).astype(np.uint8))


# 6. **主程序**
if __name__ == "__main__":
    image_path = "zebra1.png"  # 输入含噪声的图像路径
    original_image_path = "zebra.png"  # 原始无噪声图像路径

    noisy_image = load_image(image_path)  # 直接加载含噪声的图像
    original_image = load_image(original_image_path)  # 加载原始无噪声图像

    # **分块去噪**
    lambda_param = 0.037  # 正则化参数
    rho = 1.0  # ADMM 参数
    num_iter = 50 # 迭代次数
    block_size = 64  # 分块大小
    overlap = 16  # 块之间的重叠区域

    start_time = time.time()
    denoised_image, psnr_values, ssim_values = blockwise_denoising(
        noisy_image, block_size, overlap, lambda_param, rho, num_iter, original_image
    )
    run_time = time.time() - start_time

    # 保存去噪图像
    save_image(denoised_image, "denoised_image.jpg")

    # 输出 PSNR 和 SSIM 随迭代次数的变化
    if psnr_values is not None and ssim_values is not None:
        plt.figure(figsize=(12, 6))

        # 绘制 PSNR 变化曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_iter + 1), psnr_values, label="PSNR", color="blue")
        plt.xlabel("Iteration")
        plt.ylabel("PSNR")
        plt.title("PSNR vs Iteration")
        plt.legend()

        # 绘制 SSIM 变化曲线
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_iter + 1), ssim_values, label="SSIM", color="red")
        plt.xlabel("Iteration")
        plt.ylabel("SSIM")
        plt.title("SSIM vs Iteration")
        plt.legend()

        plt.tight_layout()
        plt.savefig("psnr_ssim_vs_iteration.png")  # 保存图像
        plt.show()

    print(f"运行时间: {run_time:.2f} 秒")
    if psnr_values is not None:
        print(f"最终 PSNR: {psnr_values[-1]:.2f}")
    if ssim_values is not None:
        print(f"最终 SSIM: {ssim_values[-1]:.4f}")

    # 显示图像
    cv2.imshow("Noisy Image", noisy_image)
    cv2.imshow("Denoised Image", denoised_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
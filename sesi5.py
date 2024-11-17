import cv2
import numpy as np
import matplotlib.pyplot as plt

I_gray = cv2.imread('C:\\PerfLogs\\PCD\\tugas 5\\gambar.jpg', cv2.IMREAD_GRAYSCALE)

lpf = np.ones((3, 3), np.float32) / 9
J_gray_lpf = cv2.filter2D(I_gray, -1, lpf)

hpf = np.array([[-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]])
J_gray_hpf = cv2.filter2D(I_gray, -1, hpf)

I_color = cv2.imread("C:\\PerfLogs\\PCD\\tugas 5\\gambar.jpg")

J_color_lpf = cv2.filter2D(I_color, -1, lpf)

J_color_hpf = cv2.filter2D(I_color, -1, hpf)

plt.figure(figsize=(12, 8))

# Citra Grayscale
plt.subplot(2, 3, 1)
plt.imshow(I_gray, cmap='gray')
plt.title('Citra Grayscale Asli')

plt.subplot(2, 3, 2)
plt.imshow(J_gray_lpf, cmap='gray')
plt.title('Citra Grayscale setelah Low-pass Filter')

plt.subplot(2, 3, 3)
plt.imshow(J_gray_hpf, cmap='gray')
plt.title('Citra Grayscale setelah High-pass Filter')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(I_color, cv2.COLOR_BGR2RGB))
plt.title('Citra Berwarna Asli')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(J_color_lpf, cv2.COLOR_BGR2RGB))
plt.title('Citra Berwarna setelah Low-pass Filter')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(J_color_hpf, cv2.COLOR_BGR2RGB))
plt.title('Citra Berwarna setelah High-pass Filter')

plt.tight_layout()
plt.show()
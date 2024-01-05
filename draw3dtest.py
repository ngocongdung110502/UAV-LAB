from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu
x = np.random.standard_normal(100)
y = np.random.standard_normal(100)
z = np.random.standard_normal(100)

# Tạo đối tượng Figure
fig = plt.figure()

# Tạo đối tượng Axes3D
ax = fig.add_subplot(111, projection='3d')

# Vẽ biểu đồ
ax.scatter(x, y, z)

# Hiển thị biểu đồ
plt.show()

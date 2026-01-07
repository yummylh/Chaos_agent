import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 设置 Matplotlib 风格，防止中文乱码 (可选，根据你的系统环境)
plt.rcParams['axes.unicode_minus'] = False 
# 如果图表中中文显示方块，可以尝试解开下面这行的注释并设置合适的字体
# plt.rcParams['font.sans-serif'] = ['SimHei'] 

def simulate_logistic_map(r: float, steps: int = 100, x0: float = 0.5):
    """
    计算 Logistic 映射并返回：(状态描述文本, 图像对象)
    """
    # 1. 数值计算
    data = []
    x = x0
    t_vals = list(range(steps))
    
    for _ in range(steps):
        x = r * x * (1 - x)
        data.append(x)
    
    # 2. 状态分析
    # 取最后 20 个点来判断是否稳定
    final_vals = [round(v, 4) for v in data[-20:]]
    unique_vals = len(set(final_vals))
    
    if unique_vals == 1: 
        status = "稳定定点 (Fixed Point)"
    elif unique_vals == 2: 
        status = "2周期振荡 (Period-2)"
    elif unique_vals == 4: 
        status = "4周期振荡 (Period-4)"
    else: 
        status = "混沌状态 (Chaos)"
        
    result_text = f"✅ **计算完成**\n\n检测到参数 $r={r}$，系统处于 **{status}**。\n(分析基于最后20次迭代的数值特征)"

    # 3. 核心修改：生成图像对象
    # 使用面向对象方式绘图，避免多线程冲突
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # 绘制时序图
    ax.plot(t_vals, data, 'b.-', linewidth=1, markersize=8, alpha=0.7)
    
    # 设置标题和标签
    ax.set_title(f"Logistic Map Time Series (r={r})")
    ax.set_xlabel("Iteration (t)")
    ax.set_ylabel("Value (x)")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 设定Y轴范围，让图更好看
    ax.set_ylim(-0.05, 1.05)

    # 4. 返回：(文本结果, 图片对象)
    return result_text, fig

def simulate_lorenz(sigma=10.0, rho=28.0, beta=2.667, duration=40.0):
    """
    计算洛伦兹吸引子并返回：(状态描述文本, 图像对象)
    """
    # 1. 定义方程
    def lorenz_deriv(state, t):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    # 2. 数值积分
    t = np.linspace(0, duration, int(duration * 100))
    states = odeint(lorenz_deriv, [1.0, 1.0, 1.0], t)
    
    result_text = f"🦋 **洛伦兹吸引子生成完毕**\n\n参数设置：$\\sigma={sigma}, \\rho={rho}, \\beta={beta}$"
    # 3. 核心修改：生成 3D 图像对象
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    ax.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.8, color='purple', alpha=0.8)
    
    # 设置标签
    ax.set_title("Lorenz Attractor Trajectory")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    
    return result_text, fig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

mode = int(input("模式：1：2065木星引力弹弓 2：2058太阳系-逐月计划 3：2058地月系-逐月计划："))

au, G, RE, ME = 1.495978707e11, 6.6742e-11, 1.48e11, 5.97219e24

# m为天体质量数据（kg），xyz为坐标三维数据（km），uvw为速度三维数据（km）
# 每个天体的数据请竖向填写，注意顺序，顺序将作为天体编号（从0开始）
if mode == 1:
    m = np.array([1.989E+30, 5.965E+24, 1.898E+27]) * G
    xyz = 1e3 * np.array([[0, -2.248743774648339E+07, -7.334095837456553E+08],
                          [0, 1.451246839993123E+08, -3.541902929589149E+08],
                          [0, -3.745791706017405E+04, 1.787699187295392E+07]])
    uvw = 1e3 * np.array([[0, -2.989597388673741E+01, 5.511867818998292E+00],
                          [0, -4.843622138354837E+00, -1.115255652020616E+01],
                          [0, 1.590193658562056E-03, -7.649503383327172E-02]])
elif mode == 2:
    m = np.array([1.989E+30, 5.965E+24, 7.342E+22]) * G
    xyz = 1e3 * np.array([[0, -2.652958721664486E+07, -2.615064615412870E+07],
                          [0, 1.445816428381707E+08, 1.445853462488825E+08],
                          [0, 1.874232232806087E+04, -1.599796096482128E+04]])
    uvw = 1e3 * np.array([[0, -2.981975746515421E+01, -2.975980436433774E+01],
                          [0, -5.311385200743143E+00, -4.284156377234281E+00],
                          [0, 1.092907798056375E-03, 6.499539621851680E-03]])
elif mode == 3:
    m = np.array([5.965E+24, 7.342E+22]) * G
    xyz = 1e3 * np.array([[-4.604355361147710E+03, 3.743367071550058E+05],
                          [-4.499860440644611E+01, 3.658412107418428E+03],
                          [4.221147440852379E+02, -3.431816854880171E+04]])
    uvw = 1e3 * np.array([[-7.284652112620815E-04, 5.922463560520024E-02],
                          [-1.248143051387276E-02, 1.014747392994989E+00],
                          [-6.569373627219720E-05, 5.340938087522895E-03]])
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.grid()
ax.set_xlim3d([-8 * RE, 8 * RE])
ax.set_ylim3d([-8 * RE, 8 * RE])
ax.set_zlim3d([-8 * RE, 8 * RE])

# 设置视角（可选，默认斜视图）
# ax.view_init(elev=0, azim=0)

# 设置坐标系可见范围
if mode == 2:
    ax.set_xlim3d([-2e11, 2e11])
    ax.set_ylim3d([-2e11, 2e11])
    ax.set_zlim3d([-2e11, 2e11])
elif mode == 3:
    ax.set_xlim3d([-1.5e9, 1.5e9])
    ax.set_ylim3d([-1.5e9, 1.5e9])
    ax.set_zlim3d([-1.5e9, 1.5e9])

traces = [ax.plot([], [], [], '-', lw=0.5)[0] for _ in range(len(m))]
pts = [ax.plot([], [], [], marker='o')[0] for _ in range(len(m))]

# 设置模拟时间，dt为每次计算位移之间间隔的秒数，N为总计算次数
# 例：N=24;dt=3600 间隔1小时计算一次位置，模拟24次（一天）
if mode == 1:
    N = 24 * 365 * 15
    dt = 3600
elif mode == 2:
    N = 24 * 365 * 7
    dt = 3600
elif mode == 3:
    N = 24 * 365 * 7
    dt = 3600

ts = np.arange(0, N * dt, dt)
xyzs, xyzas = [], []
time = 0

# 若模拟次数过多，播放动画以最大速度仍然缓慢，此时修改skip的数值，即为播放动画时跳过的计算次数数量，不影响结果。最小值为1
skip = 24

for _ in ts:
    xyz_ij = (xyz.reshape(3, 1, len(m)) - xyz.reshape(3, len(m), 1))
    r_ij = np.sqrt(np.sum(xyz_ij ** 2, 0))

    for j in range(len(m)):
        for i in range(len(m)):
            if i != j:
                uvw[:, i] += m[j] * xyz_ij[:, i, j] * dt / r_ij[i, j] ** 3

                # 上方代码为计算正常引力，尽量不要修改
                # 若要给天体设置加速度等信息，请在下方使用，i为天体编号，可通过if语句，筛选指定天体数据
                # 通过np.linalg.norm(向量)计算模长
                # 通过uvw[:, i] / np.linalg.norm(uvw[:, i], keepdims=True)获得天体速度的单位向量，乘以加速度即可使天体的速度变化量，记得加到uvw[:, i]里
                # 通过uvw[:, a] - uvw[:, b]获得相对速度，其中a为所求天体编号，b为参考系基准天体编号

                if mode == 1:

                    if i == 1 and _ >= 3600 * (24 * 245 + 17):
                        uvw[:, i] += 2.549E-5 * 3600 * uvw[:, i] / np.linalg.norm(uvw[:, i], keepdims=True)
                        if _ in [371458800 - (60 * 24 * 3600), 371458800 + (60 * 24 * 3600)]:
                            print(np.linalg.norm(uvw[:, i]))
                elif mode == 2:
                    if i == 2:
                        if np.linalg.norm(xyz[:, 1] - xyz[:, 2]) <= 2e9:
                            dv1 = uvw[:, 2] - uvw[:, 1]
                            uvw[:, i] += 2.65e-6 * 3600 / 3.5 * dv1 / np.linalg.norm(dv1, keepdims=True)
                        else:
                            uvw[:, i] += 2.65e-6 * 3600 / 3.5 * uvw[:, i] / np.linalg.norm(uvw[:, i], keepdims=True)
                elif mode == 3:
                    if i == 1:
                        uvw[:, i] += 2.65e-6 * 3600 * uvw[:, i] / np.linalg.norm(uvw[:, i], keepdims=True)

    time += 1
    xyz += uvw * dt
    if time >= skip:
        xyzs.append(xyz.tolist())
        time = 0

xyzs = np.array(xyzs).transpose(2, 1, 0)


def animate(n):
    for i in range(len(xyzs)):
        xyz = xyzs[i]
        traces[i].set_data(xyz[0, :n], xyz[1, :n])
        traces[i].set_3d_properties(xyz[2, :n])
        pts[i].set_data(xyz[0, n], xyz[1, n])
        pts[i].set_3d_properties(xyz[2, n])

        # 若要跟踪某天体，请将data设置为天体编号，将size设置为可视范围
        # data = xyzs[2, :, n]
        # size=1e10
        # ax.set_xlim([data[0] - size, data[0] + size])
        # ax.set_ylim([data[1] - size, data[1] + size])
        # ax.set_zlim([data[2] - size, data[2] + size])
    return traces + pts


ani = animation.FuncAnimation(fig, animate, range(int(N / skip)), interval=1, blit=True)

# 若希望导出视频请修改下方代码，将ffmpeg路径设置为安装路径，并修改视频属性设置
# plt.rcParams['animation.ffmpeg_path'] = r".\ffmpeg\bin\ffmpeg.exe"
# ani.save("animation.mp4", writer=animation.FFMpegWriter(fps=60, metadata=dict(artist="Ianzb"), extra_args=['-vcodec', 'libx264']))
plt.show()

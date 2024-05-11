from os import cpu_count
import math
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib import animation
import sys

mode = int(input("模式：1：2065木星引力弹弓 2：2058太阳系-逐月计划 3：2058地月系-逐月计划："))

au, G, RE, ME = 1.495978707e11, 6.6742e-11, 1.48e11, 5.97219e24

# m = np.array([3.32e5, 0.055, 0.815, 1, 0.107, 317.8]) * ME * G
# r = np.array([0, 0.387, 0.723, 1, 1.524, 5.203]) * RE
# v = np.array([0, 47.89, 35.03, 29.79, 24.13, 13.06]) * 1000
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
# 位置和速度竖着写
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.grid()
ax.set_xlim3d([-8 * RE, 8 * RE])
ax.set_ylim3d([-8 * RE, 8 * RE])
ax.set_zlim3d([-8 * RE, 8 * RE])
if mode ==2:
    ax.set_xlim3d([-2e11, 2e11])
    ax.set_ylim3d([-2e11,2e11])
    ax.set_zlim3d([-2e11, 2e11])
elif mode == 3:
    ax.set_xlim3d([-1.5e9, 1.5e9])
    ax.set_ylim3d([-1.5e9, 1.5e9])
    ax.set_zlim3d([-1.5e9, 1.5e9])
traces = [ax.plot([], [], [], '-', lw=0.5)[0] for _ in range(len(m))]
pts = [ax.plot([], [], [], marker='o')[0] for _ in range(len(m))]
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
skip = 24
time = 0
distance = 1e100
# ax.view_init(elev=0, azim=0)
for _ in ts:
    xyz_ij = (xyz.reshape(3, 1, len(m)) - xyz.reshape(3, len(m), 1))
    r_ij = np.sqrt(np.sum(xyz_ij ** 2, 0))

    for j in range(len(m)):
        for i in range(len(m)):
            if i != j:
                uvw[:, i] += m[j] * xyz_ij[:, i, j] * dt / r_ij[i, j] ** 3
                if mode == 1:
                    # print(np.linalg.norm(uvw[:, i]))

                    if i == 1 and _ >= 3600 * (24 * 245 + 17):  # 2065年9月3日下午5时左右
                        # if -7500 < uvw[:, i][0] <= 100000 and 20000 < uvw[:, i][1] <= 200000:
                        uvw[:, i] += 2.549E-5 * 3600 * uvw[:, i] / np.linalg.norm(uvw[:, i], keepdims=True)
                        # if np.linalg.norm(xyz[:, 1] - xyz[:, 2]) < distance:
                        #     distance = np.linalg.norm(xyz[:, 1] - xyz[:, 2])
                        if _ in [371458800 - (60 * 24 * 3600), 371458800 + (60 * 24 * 3600)]:
                            print(np.linalg.norm(uvw[:, i]))
                elif mode == 2:
                    if i == 2:
                        if np.linalg.norm(xyz[:, 1] - xyz[:, 2]) <=2e9:
                            dv1=uvw[:, 2]-uvw[:, 1]
                            uvw[:, i] += 2.65e-6 * 3600/3.5 * dv1 / np.linalg.norm(dv1, keepdims=True)
                        else:
                            uvw[:, i] += 2.65e-6 * 3600/3.5 * uvw[:, i] / np.linalg.norm(uvw[:, i], keepdims=True)
                        # print(np.linalg.norm(xyz[:, 1] - xyz[:, 2]))
                elif mode == 3:
                    if i == 1:
                        uvw[:, i] += 2.65e-6 * 3600 * uvw[:, i] / np.linalg.norm(uvw[:, i], keepdims=True)
                        # print(np.linalg.norm(xyz[:, 0] - xyz[:, 1]))

    time += 1
    xyz += uvw * dt
    if time >= skip:
        xyzs.append(xyz.tolist())
        time = 0

xyzs = np.array(xyzs).transpose(2, 1, 0)


def animate(n):
    for i in range(len(xyzs)):
        xyz = xyzs[i]
        # data = xyzs[2, :, n]
        traces[i].set_data(xyz[0, :n], xyz[1, :n])
        traces[i].set_3d_properties(xyz[2, :n])
        pts[i].set_data(xyz[0, n], xyz[1, n])
        pts[i].set_3d_properties(xyz[2, n])
        # ax.set_xlim([data[0] - 1e10, data[0] + 1e10])
        # ax.set_ylim([data[1] - 1e10, data[1] + 1e10])
        # ax.set_zlim([data[2] - 1e10, data[2] + 1e10])
    return traces + pts


ani = animation.FuncAnimation(fig, animate, range(int(N / skip)), interval=1, blit=True)
plt.rcParams['animation.ffmpeg_path'] = r".\ffmpeg\bin\ffmpeg.exe"
# ani.save("animation.mp4", writer=animation.FFMpegWriter(fps=60, metadata=dict(artist="Ianzb"), extra_args=['-vcodec', 'libx264']))
plt.show()

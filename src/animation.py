import numpy as np
import pandas as pd
import matplotlib.animation

from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

# Referenced from https://github.com/SISDevelop/SwarmPackagePy
def animation3D(snapshots, function, lb, ub, sr=False):
    # Setup graph
    side = np.linspace(lb, ub, 45)
    X, Y = np.meshgrid(side, side)
    zs = np.array([function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    fig = plt.figure()
    ax = Axes3D(fig, elev=0)
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        cmap="gist_ncar",
        linewidth=0,
        antialiased=True,
    )
    ax.set_xlim(lb, ub)
    ax.set_ylim(lb, ub)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Iterate agent positions
    plots = []
    for idx, snapshot in enumerate(snapshots):
        plots.append([])
        for agent in snapshot:
            res = function(agent)
            plots[idx].append([*agent, res])
    plots = np.asarray(plots)

    def update_graph(num):
        snapshot = plots[num]
        sc._offsets3d = (snapshot[:, 0], snapshot[:, 1], snapshot[:, 2])
        title.set_text(f"iteration: {num}")

    title = ax.set_title("iteration: 0")
    snapshot = plots[0]
    sc = ax.scatter(
        snapshot[:, 0],
        snapshot[:, 1],
        snapshot[:, 2],
        c="black",
        zorder=-1,
    )
    ani = matplotlib.animation.FuncAnimation(
        fig,
        update_graph,
        len(snapshots),
        interval=200,
        blit=False,
    )

    if sr:
        ani.save("result.mp4")

    plt.show()

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


# Color lines taken from:
# https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(
    ax,
    x,
    y,
    z=None,
    cmap="copper",
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0,
):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )

    ax.add_collection(lc)

    return lc


def plot3d(ax, times, dists, satisfieds):
    ax.plot(times, dists, satisfieds, alpha=0.8, linewidth=1)
    ax.set_xlabel("time (number of flips)")
    ax.set_ylabel("$L_1$ distance from final assignment")
    ax.set_zlabel("fraction of clauses satisfied")


def plot_time_vs_dist(ax, times, dists):
    ax.plot(times, dists, alpha=0.8, linewidth=1)
    ax.set_xlabel("time (number of flips)")
    ax.set_ylabel("$L_1$ distance from final assignment")


def plot_time_vs_dist_color(fig, ax, times, dists, satisfieds):
    ls = colorline(
        ax, times, dists, satisfieds, norm=plt.Normalize(), cmap="viridis", linewidth=1
    )
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(min(dists), max(dists))
    ax.set_xlabel("time (number of flips)")
    ax.set_ylabel("$L_1$ distance from final assignment")
    fig.colorbar(ls, label="fraction of clauses satisfied")


def plot_pca_3d(fig, ax, states, satisfieds=None):
    pca = PCA(n_components=3)
    states_r = pca.fit(states).transform(states)
    s = ax.scatter(
        states_r[:, 0], states_r[:, 1], states_r[:, 2], c=satisfieds, alpha=0.6
    )
    if satisfieds is not None:
        fig.colorbar(s, label="fraction of clauses satisfied")


def plot_pca_2d(fig, ax, states, satisfieds=None, desc="(time, clause)"):
    pca = PCA(n_components=2)
    states_r = pca.fit(states).transform(states)
    s = ax.scatter(states_r[:, 0], states_r[:, 1], c=satisfieds, alpha=0.6)
    # if satisfieds is not None:
    #     fig.colorbar(s, label="fraction of clauses satisfied")
    ax.set_title("PCA (top 2) of " + desc)


DATA_KINDS = ["TRACE"]


if __name__ == "__main__":
    input_file = "trace.txt"
    with open(input_file, "r") as f:
        data = {
            kind: {"satisfieds": [], "states": [], "clauses": []} for kind in DATA_KINDS
        }
        satisfieds = []
        states = []
        clauses = []
        print("parsing data...")
        for line in f:
            if len(line) < 5 or line[:5] != "TRACE":
                continue
            kind, satisfied, state_str, clause_str = line.strip().split(";")
            data[kind]["satisfieds"].append(float(satisfied))
            temp = []
            for c in state_str:
                temp.append(int(c))
            data[kind]["states"].append(np.array(temp, dtype="int32"))
            temp = []
            for c in clause_str:
                temp.append(int(c))
            data[kind]["clauses"].append(np.array(temp, dtype="int32"))
        for kind in DATA_KINDS:
            data[kind]["satisfieds"] = np.array(data[kind]["satisfieds"])
            data[kind]["states"] = np.array(data[kind]["states"])
            data[kind]["clauses"] = np.array(data[kind]["clauses"])
            data[kind]["dists"] = np.sum(
                np.abs(data[kind]["states"] - data[kind]["states"][-1]), axis=1
            )
        print("  done.")

    fig = plt.figure()
    states = data["TRACE"]["states"]
    plot_pca_2d(
        fig,
        fig.add_subplot(121),
        states.T,
        ["green" if v else "red" for v in states[-1]],
        desc="(assignment, time)",
    )
    plot_time_vs_dist_color(
        fig,
        fig.add_subplot(122),
        list(range(len(states))),
        data["TRACE"]["dists"],
        data["TRACE"]["satisfieds"],
    )
    plt.show()

    # plot_time_vs_dist_color(fig, ax, times, dists, satisfieds)

    # plt.savefig("analysis.png", dpi=300)
    # plt.show()

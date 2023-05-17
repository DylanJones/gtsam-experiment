import pickle

import gtsam
import numpy as np
import json
from matplotlib import pyplot as plt
from apriltag_baseline import set_axes_equal


# Simple test: see if we can get positions.pkl to
# make more sense


def main():
    with open("positions.pkl", "rb") as f:
        pts = pickle.load(f)
    mean_pts = np.load("mean_pts.npy")

    # Create graph
    graph = gtsam.NonlinearFactorGraph()
    # For the first timestep, our prior is just the origin, as
    # the sensors all calibrate to 0, 0, 0 at startup.
    tag_noise_model = gtsam.noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
    # prev_pos = gtsam.Point3(0, 0, 0)
    # graph.add(gtsam.PriorFactorPoint3(0, prev_pos, tag_noise_model))
    initial_guesses = gtsam.Values()
    for i, timestep in enumerate(zip(*pts.values())):
        # One factor per measurement of camera position.
        for j, measurement in enumerate(timestep):
            graph.addPriorPoint3(i, measurement, tag_noise_model)
            if j == 0:
                initial_guesses.insert(i, measurement)

    print(graph)
    # Solve?
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_guesses, params)
    result = optimizer.optimize()

    # Plot 1: Compare raw data to GTSAM
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.title.set_text("Original")
    for tag_id, poses in pts.items():
        arr = np.array(poses)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], label=tag_id)
    set_axes_equal(ax)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.title.set_text("Simple GTSAM")
    result_pts = []
    for i in range(len(pts[0])):
        result_pts.append(result.atPoint3(i))
    result_pts = np.array(result_pts)
    ax2.scatter(result_pts[:, 0], result_pts[:, 1], result_pts[:, 2])
    set_axes_equal(ax2)

    # Plot 2: Compare mean vs GTSAM
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.title.set_text("Mean")
    ax.scatter(mean_pts[:, 0], mean_pts[:, 1], mean_pts[:, 2])
    set_axes_equal(ax)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.title.set_text("Simple GTSAM")
    ax2.scatter(result_pts[:, 0], result_pts[:, 1], result_pts[:, 2])
    set_axes_equal(ax2)

    plt.show()


if __name__ == "__main__":
    main()

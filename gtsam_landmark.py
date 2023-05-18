"""
This script expands upon our previous GTSAM scripts; this time, using GTSAM's landmark
factors and having GTSAM solve for the camera position, instead of using dt_apriltag's
positional estimates to calculate our position.
"""
import gtsam
import gtsam.utils.plot as gtplot
import numpy as np
import json
from matplotlib import pyplot as plt
import cv2
from apriltag_baseline import set_axes_equal, fx, fy, cx, cy
from dt_apriltags import Detector
from collections import defaultdict


def main():
    cap = cv2.VideoCapture("videos/tagtest_4tag_fig8.mp4")
    detector = Detector(families='tagStandard41h12',
                        nthreads=16)
    graph = gtsam.NonlinearFactorGraph()
    # Symbols for tags
    tag_sym = [gtsam.symbol('l', i) for i in range(4)]
    initial_guesses = gtsam.Values()
    for i in range(len(tag_sym)):
        pos = gtsam.Point3(0, 1 if i in (1, 2) else 0, 1 if i in (2, 3) else 0)
        initial_guesses.insert(tag_sym[i], pos)
        # Also add a large initial prior for each tag's position
        large_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([.01] * 3))
        graph.addPriorPoint3(tag_sym[i], pos, large_prior_noise)

    # First camera position is at 0,0,0
    # graph.addPriorPose3(gtsam.symbol('x', 0), gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([0] * 6)))

    pts = defaultdict(list)

    timestep_num = 0
    while cap.isOpened():
        frame = cap.read()[1]
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray, estimate_tag_pose=True, tag_size=0.192,
                                     camera_params=(fx, fy, cx, cy))
        if len(detections) != 4:
            continue

        # Symbol for current camera pos
        cam_sym = gtsam.symbol('x', timestep_num)
        # Prior to keep camera near previous position
        if timestep_num > 0:
            graph.add(gtsam.BetweenFactorPose3(cam_sym, gtsam.symbol('x', timestep_num - 1),
                                               gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1] * 6))))

        # Add each detected tag's pose as a landmark factor to the graph:
        for det in detections:
            # Convert apriltag pose to camera pose
            camera_pose = det.pose_R.T @ -det.pose_t
            camera_pose = camera_pose.flatten()  # "world coordinates" of camera
            pts[det.tag_id].append(camera_pose)

            # Make up a noise model based on reported tag error
            # (this is a pretty bad way to do it, but it's a start)
            err_scale = 1e4
            det_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([det.pose_err * err_scale] * 3))
            print(det.pose_err * err_scale)
            # Add BearingRangeFactor to graph
            graph.add(gtsam.BearingRangeFactor3D(cam_sym, tag_sym[det.tag_id],
                                                 gtsam.Unit3((-det.pose_t / np.linalg.norm(det.pose_t)).flatten()),
                                                 np.linalg.norm(det.pose_t), det_noise))

            # draw tag detections
            cv2.polylines(frame, [np.int32(det.corners)], True, (0, 255, 0), 2)

        # Add initial estimate for camera pose - just use the pose of the last tag
        initial_guesses.insert(cam_sym, gtsam.Pose3(gtsam.Rot3(det.pose_R.T), camera_pose))

        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        timestep_num += 1

    # Solve graph
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_guesses, params)
    # params = gtsam.GaussNewtonParams()
    # params.setVerbosity("SUMMARY")
    # optimizer = gtsam.GaussNewtonOptimizer(graph, initial_guesses, params)
    result = optimizer.optimize()
    marginals = gtsam.Marginals(graph, result)

    # Plot 1: Compare raw data to GTSAM; plot in 2D,
    # with confidence circles
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.title.set_text("Original")
    for tag_id, poses in pts.items():
        arr = np.array(poses)
        ax.scatter(arr[:, 0], arr[:, 1], label=tag_id)
        # error circles
        # for pose in poses:
        #     ax.add_artist(plt.Circle(pose[:2], 0.1, color='r', fill=False))
    # make axes same scale
    ax.set_aspect('equal', adjustable='box')
    ax2 = fig.add_subplot(122)
    ax2.title.set_text("GTSAM landmarks")
    result_pts = []
    for i in range(timestep_num - 1):
        result_pts.append(result.atPose3(gtsam.symbol('x', i)).translation())
    result_pts = np.array(result_pts)
    ax2.scatter(result_pts[:, 0], result_pts[:, 1])
    # error ellipses
    for i, pose in enumerate(result_pts):
        gtplot.plot_covariance_ellipse_2d(ax2, pose[:2], marginals.marginalCovariance(gtsam.symbol('x', i))[:2, :2])

    # make axes same scale
    ax2.set_aspect('equal', adjustable='box')

    # Compare mean of raw data vs. GTSAM result
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.title.set_text("Mean of raw data")
    mean_pts = np.mean(np.array(list(pts.values())), axis=0)
    ax.scatter(mean_pts[:, 0], mean_pts[:, 1])
    ax.set_aspect('equal', adjustable='box')
    ax2 = fig.add_subplot(122)
    ax2.title.set_text("GTSAM landmarks")
    ax2.scatter(result_pts[:, 0], result_pts[:, 1])
    # # error circles
    # for i, pose in enumerate(result_pts):
    #     confidence = np.sqrt(marginals.marginalCovariance(gtsam.symbol('x', i))[0, 0])
    #     ax2.add_artist(plt.Circle(pose[:2], confidence, color='r', fill=False))
    ax2.set_aspect('equal', adjustable='box')

    # Plot 3: compare second and third derivatives of mean pos vs. GTSAM pos
    mean_vel = np.diff(mean_pts, axis=0)
    gtsam_vel = np.diff(result_pts, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.title.set_text("mean d^2/dt^2")
    mean_accel = np.diff(mean_vel, axis=0)
    ax.plot(np.linalg.norm(mean_accel, axis=1))
    ax2 = fig.add_subplot(222)
    ax2.title.set_text("GTSAM d^2/dt^2")
    gtsam_accel = np.diff(gtsam_vel, axis=0)
    ax2.plot(np.linalg.norm(gtsam_accel, axis=1))
    ax3 = fig.add_subplot(223)
    ax3.title.set_text("mean d^3/dt^3")
    mean_jerk = np.diff(mean_accel, axis=0)
    ax3.plot(np.linalg.norm(mean_jerk, axis=1))
    ax4 = fig.add_subplot(224)
    ax4.title.set_text("GTSAM d^3/dt^3")
    gtsam_jerk = np.diff(gtsam_accel, axis=0)
    ax4.plot(np.linalg.norm(gtsam_jerk, axis=1))

    print("mean average speed: ", np.mean(np.abs(mean_vel)))
    print("GTSAM average speed: ", np.mean(np.abs(gtsam_vel)))
    print("mean average absolute acceleration: ", np.mean(np.abs(mean_accel)))
    print("GTSAM average absolute acceleration: ", np.mean(np.abs(gtsam_accel)))
    print("mean average absolute jerk: ", np.mean(np.abs(mean_jerk)))
    print("GTSAM average absolute jerk: ", np.mean(np.abs(gtsam_jerk)))

    plt.show()


if __name__ == "__main__":
    main()

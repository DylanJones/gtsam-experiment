"""
This script expands upon our previous GTSAM scripts; this time, using GTSAM's landmark
factors and having GTSAM solve for the camera position, instead of using dt_apriltag's
positional estimates to calculate our position.
"""
import gtsam
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
    initial_guesses = gtsam.Values()
    tag_locations = {}
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

        # One variable for the current camera pos:
        cam_sym = gtsam.symbol('x', timestep_num)
        # Add factors for each tag's position
        for det in detections:
            # Convert apriltag pose to camera pose
            camera_pose = det.pose_R.T @ -det.pose_t
            camera_pose = camera_pose.flatten()  # "world coordinates" of camera
            if det.tag_id not in tag_locations:
                tag_locations[det.tag_id] = camera_pose

            camera_pose = camera_pose - tag_locations[det.tag_id]
            pts[det.tag_id].append(camera_pose)


            # Make up a noise model based on reported tag error
            # (this is a pretty bad way to do it, but it's a start)
            err_scale = 1e5
            det_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([det.pose_err * err_scale] * 3))
            print(det.pose_err * err_scale)
            # Add the factor
            graph.add(gtsam.PriorFactorPoint3(cam_sym, gtsam.Point3(camera_pose), det_noise))
            # draw tag detections
            cv2.polylines(frame, [np.int32(det.corners)], True, (0, 255, 0), 2)

        # Add initial estimate for camera pose - just use the pose of the last tag
        initial_guesses.insert(cam_sym, gtsam.Point3(camera_pose))

        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        timestep_num += 1


    # Solve graph
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_guesses, params)
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
    ax2.title.set_text("GTSAM, basic independent variables")
    result_pts = []
    for i in range(timestep_num - 1):
        result_pts.append(result.atPoint3(gtsam.symbol('x', i)))
    result_pts = np.array(result_pts)
    ax2.scatter(result_pts[:, 0], result_pts[:, 1])
    # error circles
    for i, pose in enumerate(result_pts):
        confidence = np.sqrt(marginals.marginalCovariance(gtsam.symbol('x', i))[0, 0])
        ax2.add_artist(plt.Circle(pose[:2], confidence, color='r', fill=False))
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
    ax2.set_aspect('equal', adjustable='box')
    plt.show()


if __name__ == "__main__":
    main()

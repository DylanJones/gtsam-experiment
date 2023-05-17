import json
from collections import defaultdict
import pickle
from dt_apriltags import Detector
import numpy as np
from matplotlib import pyplot as plt
import argparse
import cv2

# Camera parameters
# TODO: better way to specify
fx, fy, cx, cy = 950.3144831668717, 957.9449124560755, 956.4186920683519, 537.6959870338106
mtx = np.array([[950.31448317, 0., 956.41869207],
                [0., 957.94491246, 537.69598703],
                [0., 0., 1.]])
dist = np.array([[0.01404704, -0.21364745, 0.0061348, -0.00254836, 0.45306021]])
newcameramatrix = np.array([[1.11676392e+03, 0.00000000e+00, 9.58039639e+02],
                            [0.00000000e+00, 9.54482300e+02, 5.42791171e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
roi = (290, 226, 1330, 628)


# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    ax.set_box_aspect((1, 1, 1))
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, default="videos/tagtest_4tag_fig8.mp4", help="Path to video file")
    parser.add_argument('-s', '--size', type=float, default=0.192, help="Size of tag in meters")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    detector = Detector(families='tagStandard41h12',
                        nthreads=16)
    positions = defaultdict(list)
    raw_positions = defaultdict(list)
    raw_rotations = defaultdict(list)
    # World-space locations of each tag
    # tag_locations = {
    #     0: np.array([0, 0, 0]),
    #     1: np.array([0, 1, 0]),
    #     2: np.array([0, 1, 1]),
    #     3: np.array([0, 0, 1]),
    # }
    tag_locations = {}
    # 3D Plot positions
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    n = 0
    while cap.isOpened():
        n += 1
        # Read image
        frame = cap.read()[1]
        if frame is None: break
        # Undistort
        # frame = cv2.undistort(frame, mtx, dist, None, newcameramatrix)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect any tags
        detections = detector.detect(gray, estimate_tag_pose=True, tag_size=args.size,
                                     camera_params=(fx, fy, cx, cy))
        if len(detections) != 4:
            continue
        # Draw detections
        for det in detections:
            # convert to integer coordinates
            pts = np.array(det.corners, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            # Convert apriltag pose to camera pose
            camera_pose = det.pose_R.T @ -det.pose_t
            camera_pose = camera_pose.flatten()
            if det.tag_id not in tag_locations:
                tag_locations[det.tag_id] = camera_pose

            camera_pose = camera_pose - tag_locations[det.tag_id]

            # Add to position list
            positions[det.tag_id].append(camera_pose)
            raw_positions[det.tag_id].append(det.pose_t)
            raw_rotations[det.tag_id].append(det.pose_R)
        # clear and replot
        plt.cla()
        for tag_id, poses in positions.items():
            arr = np.array(poses)
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], label=tag_id)
        ax.set_box_aspect((1, 1, 1))
        set_axes_equal(ax)
        plt.pause(0.001)

        # Display image
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    plt.cla()
    for tag_id, poses in positions.items():
        arr = np.array(poses)
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], label=tag_id)
    ax.set_box_aspect((1, 1, 1))
    set_axes_equal(ax)
    ax2 = fig.add_subplot(122, projection='3d')
    mean_pts = np.mean(np.array(list(positions.values())), axis=0)
    ax2.scatter(mean_pts[:, 0], mean_pts[:, 1], mean_pts[:, 2])
    ax2.set_box_aspect((1, 1, 1))
    set_axes_equal(ax2)
    plt.show()
    np.save("mean_pts.npy", mean_pts)
    with open("positions.pkl", "wb") as f:
        pickle.dump(positions, f)
    with open("raw_positions.pkl", "wb") as f:
        pickle.dump(raw_positions, f)
    with open("raw_rotations.pkl", "wb") as f:
        pickle.dump(raw_rotations, f)


if __name__ == "__main__":
    main()

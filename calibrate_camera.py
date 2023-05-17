import cv2
import argparse
import numpy as np


# Use chessboard picture to generate camera calibration parameters
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, default="videos/chessboard_6_slow.mp4", help="Path to video file")
    parser.add_argument('-s', '--size', type=float, default=0.134, help="Size of chessboard, in meters")
    args = parser.parse_args()

    cb_size = (5, 7)

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Object points for internal corners of chessboard
    square_size = args.size / 8
    objp = np.zeros((cb_size[0] * cb_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:cb_size[0], 0:cb_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    cap = cv2.VideoCapture(args.video)
    i = 0
    while cap.isOpened():
        # Read image
        frame = cap.read()[1]
        i += 1
        if frame is None: break
        if i % 10 != 0: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect any tags
        ret, corners = cv2.findChessboardCorners(gray, cb_size, None)
        if ret:
            # refine
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, cb_size, corners, ret)
            # Add to object and image points
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            print("No corners found")

        # Display image
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calibrate camera
    print(f'Found {len(objpoints)} images with chessboard corners')
    print("Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, gray.shape[::-1], 1, gray.shape[::-1])
    print("Calibration complete")

    # Extract and print fx, fy, cx, cy
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]

    # Print results
    print(f"{fx=}, {fy=}, {cx=}, {cy=}")
    print(f"{mtx=}, {dist=}")
    print(f"{newcameramatrix=}, {roi=}")


if __name__ == "__main__":
    main()

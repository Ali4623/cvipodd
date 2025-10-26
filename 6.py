# exp6_camshift_tracking.py
import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # use 0 or provide video file path
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot open camera/video")

# Select ROI manually (first frame)
r = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
x, y, w, h = [int(v) for v in r]
track_window = (x, y, w, h)
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Mask and histogram
mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup termination criteria: 10 iterations or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    # apply CamShift
    ret2, track_window = cv2.CamShift(back_proj, track_window, term_crit)
    pts = cv2.boxPoints(ret2)
    pts = np.int0(pts)
    tracked = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    cv2.imshow("CamShift Tracking", tracked)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC to break
        break

cap.release()
cv2.destroyAllWindows()

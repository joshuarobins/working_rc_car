import config
import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;0|fflags;nobuffer|flags;low_delay"

rtsp_url = "rtsp://"+config.JOSHPI_IP+":"+config.JOSHPI_PORT+"/cam"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream via FFmpeg")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("PiCam Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
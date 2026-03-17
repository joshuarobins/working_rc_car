import config
import cv2

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
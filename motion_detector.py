import cv2

start_frame = None
video = cv2.VideoCapture(0)
frame_count = 0
while True:
    # read video frame
    check, frame = video.read()

    # convert to gray image for better processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # obtain first frame to establish background
    if frame_count < 5:
        start_frame = gray_frame
        frame_count = frame_count + 1
        continue

    delta_frame = cv2.absdiff(start_frame, gray_frame)

    threshold_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

    (contours, _) = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # check findContours

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # display video
    cv2.imshow("Gray Frame", gray_frame)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", threshold_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    print(frame)

    # quit program
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

def main():
    # Load the video
    video_file = 'D:\human detection\people.mp4'
    cap = cv2.VideoCapture(video_file) #video_file

    # Initialize the HOG descriptor and SVM classifier
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Detect humans in the frame
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # Draw bounding boxes around the detected humans
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the output
        cv2.imshow('Human Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

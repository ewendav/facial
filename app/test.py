# import cv2

# def check_camera():
#     # Set the video source (use 0 for the default camera)
#     video_source = 0

#     # Open the video capture object
#     cap = cv2.VideoCapture(video_source)

#     try:
#         # Check if the camera is opened successfully
#         if not cap.isOpened():
#             raise Exception("Could not open camera. Check if it is properly connected.")

#         # Read a frame from the camera
#         ret, frame = cap.read()

#         # Check if the frame is successfully read
#         if not ret:
#             raise Exception("Could not read a frame from the camera. Check if it is working properly.")

#         # Display the frame (optional)
#         cv2.imshow("Camera Test", frame)
#         cv2.waitKey(0)  # Wait until any key is pressed

#     except Exception as e:
#         print(f"Error: {e}")

#     finally:
#         # Release the video capture object
#         cap.release()
#         cv2.destroyAllWindows()

# # Call the function to check the camera
# check_camera()


import cv2

# Open the video capture device (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if ret:
        # Display the frame
        cv2.imshow('Video Feed', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Optionally, you can print a message if there's an issue with capturing frames
        print("Error: Unable to capture frame.")

# Release the video capture object and close the window when the loop is exited
cap.release()
cv2.destroyAllWindows()

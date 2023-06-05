import cv2
import numpy as np

class HandoverDetector():
    def init(self):
        ...

    def find_orange_shape_and_compute_optical_flow(self, image):
        # Load the image
        # image = cv2.imread(image_path)
        
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define the lower and upper bounds for the orange color in HSV
        orange_lower = np.array([5, 50, 50])
        orange_upper = np.array([15, 255, 255])
        
        # Threshold the image to get only the orange color
        mask = cv2.inRange(hsv_image, orange_lower, orange_upper)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # If no orange shape is found, compute optical flow
            
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Create a mask for drawing the optical flow vectors
            mask = np.zeros_like(image)
            
            # Create some random colors for visualizing the optical flow
            color = np.random.randint(0, 255, (100, 3))
            
            # Parameters for the Lucas-Kanade optical flow method
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Detect keypoints in the first frame
            keypoints = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
            
            # Create an array to store the previous keypoints
            prev_keypoints = keypoints
            
            # Initialize variables for calculating average flow magnitude
            total_magnitude = 0.0
            num_vectors = 0
            
            # Iterate over subsequent frames
            while True:
                # Load the next frame
                ret, frame = cv2.imread().read()
                
                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow using Lucas-Kanade method
                next_keypoints, status, error = cv2.calcOpticalFlowPyrLK(
                    prevImg=gray,
                    nextImg=gray,
                    prevPts=prev_keypoints,
                    nextPts=None,
                    **lk_params
                )
                
                # Select good keypoints and draw optical flow vectors
                good_prev = prev_keypoints[status == 1]
                good_next = next_keypoints[status == 1]
                
                for i, (prev, next) in enumerate(zip(good_prev, good_next)):
                    x1, y1 = prev.ravel()
                    x2, y2 = next.ravel()
                    
                    # Draw the optical flow vector
                    cv2.line(mask, (x1, y1), (x2, y2), color[i].tolist(), 2)
                    cv2.circle(frame, (x2, y2), 5, color[i].tolist(), -1)
                    
                    # Compute the magnitude of the optical flow vector
                    magnitude = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    
                    # Accumulate the magnitude and increment the count
                    total_magnitude += magnitude
                    num_vectors += 1
                
                # Overlay the optical flow vectors on the frame
                result = cv2.add(frame, mask)
                
                # Display the result
                cv2.imshow("Optical Flow", result)
                
                # Exit if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Update the previous keypoints
                prev_keypoints = good_next
            
            # Calculate the average flow magnitude
            average_magnitude = total_magnitude / num_vectors
            print("Average Flow Magnitude:", average_magnitude)
            
            # Release the video capture object and close windows
            cv2.destroyAllWindows()
        
        else:
            # Iterate over the contours and find the orange shape
            for contour in contours:
                # Calculate the area of the contour
                area = cv2.contourArea(contour)
                
                # Filter out small contours
                if area > 100:
                    # Draw a bounding box around the contour
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
            # Display the result
            cv2.imshow("Orange Shape Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def cameraLoop(self):  
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            # self.find_orange_shape_and_compute_optical_flow(frame)
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    HD = HandoverDetector()
    HD.cameraLoop()
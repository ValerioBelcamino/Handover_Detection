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
        # orange_lower_perimeter = np.array([  1, 40, 83])
        # orange_upper_perimeter = np.array([23, 255, 254])
        orange_lower_perimeter = np.array([  1, 25, 60])
        orange_upper_perimeter = np.array([50, 255, 254])
        

        # Threshold the image to get only the orange color
        mask_perimeter = cv2.inRange(hsv_image, orange_lower_perimeter, orange_upper_perimeter)
        
        # mask_perimeter = cv2.dilate(mask_perimeter, None, iterations=1)
        mask_perimeter= cv2.morphologyEx(mask_perimeter, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

        
        contours, _ = cv2.findContours(mask_perimeter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        destination = np.array([[0, 0], [0, 480-1], [640-1, 480-1], [640-1, 0]], dtype=np.float32)

        if len(contours) > 0:
            # approx = cv2.approxPolyDP(contours, 0.01 * cv2.arcLength(contours[0], True), True)
            # cv2.drawContours(image, contours[0], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contours[0])
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            approx = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
            # print(approx)
            corners = self.find_corners(approx)
            print(corners.shape)

            # cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            # for c in corners:
            #     cv2.circle(image, c, 5, (0, 0, 255), -1)

            tf_matrix = cv2.getPerspectiveTransform(corners, destination)
            print(tf_matrix)
            cv2.warpPerspective(image, tf_matrix, (640, 480), image, cv2.INTER_LINEAR)
            cv2.warpPerspective(mask_perimeter, tf_matrix, (640, 480), mask_perimeter, cv2.INTER_LINEAR)
            cv2.imshow("mask_perimeter", mask_perimeter)
            # perimeter = cv2.arcLength(contours[0], True)
            # print(perimeter)
            # approx = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
            # cv2.drawContours(image, approx, -1, (0, 255, 0), 2)
        return
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

    def find_corners(self, pts, image_size=(640, 480)):
        pts= pts.reshape((pts.shape[0], -1))
        # print(pts.shape)
        top_left = np.array(sorted(pts, key=lambda x: np.sqrt((x[0]-0)**2 + (x[1]-0)**2)))
        top_right = np.array(sorted(pts, key=lambda x: np.sqrt((x[0]-image_size[0])**2 + (x[1]-0)**2)))
        bottom_left = np.array(sorted(pts, key=lambda x: np.sqrt((x[0]-0)**2 + (x[1]-image_size[1])**2)))
        bottom_right = np.array(sorted(pts, key=lambda x: np.sqrt((x[0]-image_size[0])**2 + (x[1]-image_size[1])**2)))

        return np.array([top_left[0], bottom_left[0], bottom_right[0], top_right[0]], dtype=np.float32)

    def cameraLoop(self):  
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            self.find_orange_shape_and_compute_optical_flow(frame)
            cv2.imshow("Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    HD = HandoverDetector()
    HD.cameraLoop()
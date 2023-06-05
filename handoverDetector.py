import cv2
import numpy as np

class HandoverDetector():

    def __init__(self):
        self.of_mask = np.zeros((480, 640, 3), dtype=np.uint8)
        self.of_mask[..., 1] = 255
        self.of_prev_gray = None
        self.motion_threshold = 5

        # the empty box is around 0.015 so offset should be near 0.015
        self.mask_offset = 0.02
        # with a single screw is around 0.03

        self.orange_lower_perimeter = np.array([  1, 25, 60])
        self.orange_upper_perimeter = np.array([50, 255, 254])



    def find_orange_shape_and_compute_optical_flow(self, image):
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold the image to get only the orange color
        mask_perimeter = cv2.inRange(hsv_image, self.orange_lower_perimeter, self.orange_upper_perimeter)
        
        # mask_perimeter = cv2.dilate(mask_perimeter, None, iterations=1)
        mask_perimeter= cv2.morphologyEx(mask_perimeter, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))

        contours, _ = cv2.findContours(mask_perimeter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        destination = np.array([[0, 0], [0, 480-1], [640-1, 480-1], [640-1, 0]], dtype=np.float32)

        if len(contours) > 0:
            ''' UNCOMMENT TO USE THE RECTANGLE '''
            # x, y, w, h = cv2.boundingRect(contours[0])
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            perimeter = cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], 0.01 * perimeter, True)
            # print(approx)
            corners = self.find_corners(approx)
            # print(corners.shape)

            ''' UNCOMMENT TO DRAW THE CORNERS AND THE CONTOURS'''
            # cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            # for c in corners:
            #     cv2.circle(image, c, 5, (0, 0, 255), -1)

            tf_matrix = cv2.getPerspectiveTransform(corners, destination)
            # print(tf_matrix)
            warped_image = image.copy()
            cv2.warpPerspective(image, tf_matrix, (640, 480), warped_image, cv2.INTER_LINEAR)
            cv2.warpPerspective(mask_perimeter, tf_matrix, (640, 480), mask_perimeter, cv2.INTER_LINEAR)
            # cv2.imshow("mask_perimeter", mask_perimeter)
            # print(perimeter)
            filled_ratio = 1 - (cv2.countNonZero(mask_perimeter) / (640*480))

            filled_threshold = self.mask_offset
            if filled_ratio > filled_threshold:
                print(f"Filled box! {filled_ratio:.2}")
            else:
                print(f"Empty box! {filled_ratio:.2}")

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.of_prev_gray is None:
                self.of_prev_gray = gray
            
            ''' UNCOMMENT TO USE OPTICAL FLOW '''
            # print(self.of_prev_gray.shape, gray.shape)
            # optical_flow = cv2.calcOpticalFlowFarneback(self.of_prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
 
            # magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
            # self.of_mask[..., 0] = angle * 180 / np.pi / 2
            # self.of_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            # rgb_mask = cv2.cvtColor(self.of_mask, cv2.COLOR_HSV2BGR)

            diff_frame = cv2.absdiff(self.of_prev_gray, gray)
            # cv2.imshow("optical_flow", diff_frame)

            motion_idx = np.sum(diff_frame) / (640*480)
            # print(f'Motion Index: {motion_idx:.4}\n\n')
       
            if motion_idx > self.motion_threshold:
                print("Handover detected!", motion_idx)
            self.of_prev_gray = gray



    def find_corners(self, pts, image_size=(640, 480)):
        pts= pts.reshape((pts.shape[0], -1))
        # print(pts.shape)
        top_left = np.array(sorted(pts, key=lambda x: np.sqrt((x[0]-0)**2 + (x[1]-0)**2)))
        top_right = np.array(sorted(pts, key=lambda x: np.sqrt((x[0]-image_size[0])**2 + (x[1]-0)**2)))
        bottom_left = np.array(sorted(pts, key=lambda x: np.sqrt((x[0]-0)**2 + (x[1]-image_size[1])**2)))
        bottom_right = np.array(sorted(pts, key=lambda x: np.sqrt((x[0]-image_size[0])**2 + (x[1]-image_size[1])**2)))

        return np.array([top_left[0], bottom_left[0], bottom_right[0], top_right[0]], dtype=np.float32)



    def cameraLoop(self):  
        cap = cv2.VideoCapture(-1)
        i = 0
        while True:
            _, frame = cap.read()
            if i > 0:
                self.find_orange_shape_and_compute_optical_flow(frame)
            cv2.imshow("Camera", frame)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    HD = HandoverDetector()
    HD.cameraLoop()
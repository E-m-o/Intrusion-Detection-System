import cv2

def frame_capture(path='test.webm'):
    """Captures and saves the frames from the video specified at path"""
    vid_obj = cv2.VideoCapture(path)

    count = 0

    success = 1
    
    while success:
        success, image = vid_obj.read()
        if count < 10:
            cv2.imwrite("video_frames/frame_00%d.jpg" % count, image)
        elif count < 100:
            cv2.imwrite("video_frames/frame_0%d.jpg" % count, image)
        else:
            cv2.imwrite("video_frames/frame_%d.jpg" % count, image)
        print('saving frame ' + str(count))
        count += 1


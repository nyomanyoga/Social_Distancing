# imports
import cv2, time, argparse
import numpy as np

# own modules
import calc, plot

confid = 0.5
thresh = 0.5
click_mouse = []

def mouse_click(event, x, y, flags, param):
    global click_mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(click_mouse) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)        
        if len(click_mouse) >= 1 and len(click_mouse) <= 3:
            cv2.line(image, (x, y), (click_mouse[len(click_mouse)-1][0], click_mouse[len(click_mouse)-1][1]), (70, 70, 70), 2)
            if len(click_mouse) == 3:
                cv2.line(image, (x, y), (click_mouse[0][0], click_mouse[0][1]), (70, 70, 70), 2)    
        if "click_mouse" not in globals():
            click_mouse = []
        click_mouse.append((x, y))

def calculate_distancing(vid_path, net, output_dir, ln1):
    count = 0
    vs = cv2.VideoCapture(vid_path)    
    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))

    points = []
    global image
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            print('here')
            break
        (H, W) = frame.shape[:2]
        
        # first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
        if count == 0:
            while True:
                image = frame
                cv2.imshow("image", image)
                cv2.waitKey(1)
                if len(click_mouse) == 8:
                    cv2.destroyWindow("image")
                    break
            points = click_mouse      

        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]        
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

        # models
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []   
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # detecting humans in frame
                if classID == 0:
                    if confidence > confid:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
        if len(boxes1) == 0:
            count = count + 1
            continue

        person = calc.get_transformed_points(boxes1, prespective_transform)
        distances_mat, bxs_mat = calc.get_distances(boxes1, person, distance_w, distance_h)
        risk_count = calc.get_count(distances_mat)
        frame1 = np.copy(frame)  
        img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)

        # write image fps
        if count != 0:
            cv2.imwrite(output_dir+"frame%d.jpg" % count, img)

        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;
    vs.release()
    cv2.destroyAllWindows() 

if __name__== "__main__":
    # Receives arguements specified by user
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', action='store', dest='video_path', default='./data/ex.avi' ,
                    help='Path for input video')
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/' ,
                    help='Path for Output images')
    parser.add_argument('-m', '--model', action='store', dest='model', default='./models/',
                    help='Path for models directory')

    values = parser.parse_args()   
    model_path = values.model
    if model_path[len(model_path) - 1] != '/':
        model_path = model_path + '/'
    output_dir = values.output_dir
    if output_dir[len(output_dir) - 1] != '/':
        output_dir = output_dir + '/'
        
    # load Yolov3 weights
    weightsPath = model_path + "yolov3.weights"
    configPath = model_path + "yolov3.cfg"

    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

    # set mouse callback 
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_click)
    np.random.seed(42)
    
    calculate_distancing(values.video_path, net_yl, output_dir, ln1)
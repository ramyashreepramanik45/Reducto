# Codes are from the actual Reducto repo 
# We've modified it a bit as per our 
# application. 

# from cProfile import label
# import enum
# from genericpath import exists
from math import sqrt
import cv2
from pathlib import Path
# from matplotlib.pyplot import axis
import numpy as np
from sklearn.cluster import KMeans
import time
# import skvideo.io
import os
import imutils
from PIL import Image
from sklearn.metrics import auc
import gc
import json
# import joblib

################################################
threshes = [ i / 10000 for i in range(0, 10001)]
test_weights = [0 for _ in range(29)]
final_frames = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
################################################





#############################################################################
# methods for differences
def frame_edge_diff(edge, prev_edge):
    total_pixels = edge.shape[0] * edge.shape[1]
    frame_diff = cv2.absdiff(edge, prev_edge)
    frame_diff = cv2.threshold(frame_diff, 21, 255, cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(frame_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed

def get_frame_edge(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edge = cv2.Canny(blur, 101, 255)
    return edge


def frame_pixel_diff(frame, prev_frame):
    total_pixels = frame.shape[0] * frame.shape[1]
    frame_diff = cv2.absdiff(frame, prev_frame)
    frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)
    frame_diff = cv2.threshold(frame_diff, 21, 255, cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(frame_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed

def area_frame_diff(frame, prev_frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_pixels = frame.shape[0] * frame.shape[1]
    frame_delta = cv2.absdiff(frame, prev_frame)
    thresh = cv2.threshold(frame_delta, 21, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if not contours:
        return 0.0
    return max([cv2.contourArea(c) / total_pixels for c in contours])

#############################################################################



#############################################################################
def video2img(video_path, frame_root, extension='bmp', scale=1):
    orig_width = 1920
    orig_height = 1080
    scale_str = f'{orig_width // scale}:{orig_height // scale}'
    frame_root.mkdir(parents=True, exist_ok=True)
    # ffmpeg -r 1 -i segment000.mp4 -r 1 "/tmp/frames/%05d.bmp"
    command = f'ffmpeg -hide_banner -loglevel quiet -r 1 -i {video_path} -r 1 -vf scale={scale_str} "{frame_root}/%05d.{extension}"'
    os.system(command)
    frames = [f for f in sorted(frame_root.iterdir()) if f.match(f'?????.{extension}')]
    return len(frames)


def img2video(frame_root, output_path, selected_frames=None, frame_pattern='?????', extension='bmp'):
    # if output_path.exists():
    #     return
    frame_root = Path(frame_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_frames is None:
        selected_frames = [f.stem for f in sorted(frame_root.iterdir()) if f.match(f'{frame_pattern}.{extension}')]
    # print(f'img2video {frame_root} ({len(selected_frames)}) ... ', end='')
    frame_list = [f'{frame_root}/{int(i):05d}.{extension}' for i in selected_frames]
    frame_str = ' '.join(frame_list)
    command = f'cat {frame_str} | ' \
              f'ffmpeg -hide_banner -loglevel panic ' \
              f'-f image2pipe -framerate 30 -i - {output_path}'
    os.system(command)
    # print('done')
#############################################################################

#############################################################################
#Search for the best thresh during caliberation
def get_max_acc_vector(frames, all_objects, accPassed):
    ctr = 0
    if(len(all_objects) <= frames[0][1] + 29): return  0,1, [0 for _ in range (30)], 1
    # object_vec = all_objects[frames[0][1] - 1: frames[0][1] + 29]

    if(frames[0][1]+30 not in all_objects.keys()): return 0,1,[0 for _ in range(30)],1
    
    object_vec = [all_objects[frame] for frame in range(frames[0][1] - 1,frames[0][1] + 29)]
    
    if(len(object_vec) != 30): return 0, 1, [0 for _ in range (30)], 1
    l = 0
    r = len(threshes) - 1
    max_poss_thresh = 0
    acc = 0
    lost = 0
    while(l <= r):
        m = (l + r) // 2
        lost_objects = 0
        mAP = []
        IOU_retained = 0
        last_selected = 0
        for i in range(0, 29): 
            edg1 = get_frame_edge(frames[i][0])
            edg2 = get_frame_edge(frames[i + 1][0])
            if(i >= len(frames)): break
            if(m >= len(threshes)): break
            if(frame_edge_diff(edg1, edg2) <= threshes[m]):
                # print(i, len(object_vec), last_selected)
                if(i + 1 >= len(object_vec) or last_selected >= len(object_vec)): break
                map  = get_mAP(object_vec[last_selected], object_vec[i + 1])
                mAP.append(map)
                # IOU_retained += get_IOU_retain(object_vec[last_selected], object_vec[i + 1])
            else:
                last_selected = i + 1

        avg_mAP = np.mean(mAP) if mAP else 0
        acc_for_thresh = avg_mAP 
        # acc_for_thresh = IOU_retained/29
        
        if(acc_for_thresh >= accPassed) :
            if(max_poss_thresh < threshes[m]):
                max_poss_thresh = threshes[m]
                acc = acc_for_thresh
            l = m + 1
        else:
            r = m - 1

    frames_sent = 1
    # reqd_frames = []
    diffs = []
    # reqd_frames.append(frames[0][0])
    for i in range(0, 29):
        edg1 = get_frame_edge(frames[i][0])
        edg2 = get_frame_edge(frames[i + 1][0])
        # frame_edge_diff(edg1, edg2)
        area_dif = frame_edge_diff(edg1, edg2)
        # area_dif = area_frame_diff(frames[i + 1][0], frames[i][0])
        diffs.append(area_dif)
        if(area_dif > max_poss_thresh):
            # reqd_frames.append(frames[i + 1][0])
            frames_sent += 1
    

    # print(max_poss_thresh, acc, frames_sent)
    return frames_sent, max_poss_thresh, diffs, acc
#############################################################################



def get_dist(centre, point):
    dist = 0
    for i in range(28):
        dist += ((point[i] - centre[i]) * (point[i] - centre[i]))

    dist = sqrt(dist)
    return dist

def get_radius(centre, points):
    dist = -1
    for p in points:
        t_dist = get_dist(centre, p)
        dist = max(dist, t_dist)
    return dist


#############################################################################
#calibration
def calibrate(P, vidObj, frames_done, fps, width, height, accPassed, resPath, saveSegments ,seg_no):
    frames_done_cal = 0
    frame_list = []
    diff_segment_vector = []
    segment_threshes = []
    acc = 0
    sets = 0
    used_frames = 0
    
    print("\nCalibrating...........")
    print()
    
    while frames_done_cal < 120 * fps:
        for i in range(30):
            success, img = vidObj.read()
            if not success: break
            frames_done_cal += 1
            frame_list.append((img, frames_done + frames_done_cal))

        if len(frame_list) != 30:
            break
        
        f, thresh_for_vect, diffs_for_vect, acc_here  = get_max_acc_vector(frame_list, P, accPassed)
        
        # Pad diffs_for_vect to ensure consistent length
        if len(diffs_for_vect) < 29:
            diffs_for_vect.extend([0] * (29 - len(diffs_for_vect)))
        
        used_frames += f
        # final_frames.extend(reqd_frames[:])
        diff_segment_vector.append(diffs_for_vect[:29])  # Ensure exactly 29 elements
        segment_threshes.append(thresh_for_vect)
        frame_list.clear()
        acc += acc_here
        sets += 1
        seg_no+=1
       
        print("segment number",seg_no)
        

    #print(diff_segment_vector)
    X = np.array(diff_segment_vector)
    
    if(len(X) < 5): return -1, -1, -1, -1, -1,-1
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    size_of_cluster = [0 for i in range(5)]
    thresh_of_cluster = [0 for i in range(5)]
    
    # weights = [0 for i in range(29)]
    # for i,vec in enumerate(diff_segment_vector):
    #     res = kmeans.fit(vec)
    #     cluster = res[0]
    #     size_of_cluster[cluster] += 1
    #     thresh_of_cluster[cluster] += segment_threshes[i]
    
    index_0 = np.where(kmeans.labels_ == 0)[0]
    vec_0 = np.array(diff_segment_vector)[index_0]
    index_1 = np.where(kmeans.labels_ == 1)[0]
    vec_1 = np.array(diff_segment_vector)[index_1]
    index_2 = np.where(kmeans.labels_ == 2)[0]
    vec_2 = np.array(diff_segment_vector)[index_2]
    index_3 = np.where(kmeans.labels_ == 3)[0]
    vec_3 = np.array(diff_segment_vector)[index_3]
    index_4 = np.where(kmeans.labels_ == 4)[0]
    vec_4 = np.array(diff_segment_vector)[index_4]

    l = [index_0, index_1, index_2, index_3, index_4]
    for i in range(5):
        # thresh_of_cluster[i] = np.median(np.array(segment_threshes)[l[i]])
        thresh_of_cluster[i] = np.array(segment_threshes)[l[i]].mean()


    thresh_of_cluster = np.array(thresh_of_cluster)
    
    segment_vars = np.array([get_radius(kmeans.cluster_centers_[0], vec_0), 
                             get_radius(kmeans.cluster_centers_[1], vec_1), 
                             get_radius(kmeans.cluster_centers_[2], vec_2), 
                             get_radius(kmeans.cluster_centers_[3], vec_3), 
                             get_radius(kmeans.cluster_centers_[4], vec_4)])

    return kmeans, thresh_of_cluster, segment_vars, acc , len(X) ,seg_no



    
#############################################################################



#############################################################################
#Given the kmeans, variance and diffs returns the best fit cluster, if -1 then recaliberate
def get_cluster(kmeans, var, diffs):
    dist = 100000
    res = -1
    for i in range(5):
        centroid = kmeans.cluster_centers_[i]
        dist_here = get_dist(centroid, diffs)
        if dist_here <= var[i] and dist_here < dist:
            dist = dist_here
            res = i

    return res
#############################################################################



#############################################################################
# def get_lost_objects_r(last_selected, current):
#     change = 0
#     tot_last = 0
#     for i, cnt in enumerate(last_selected):
#         tot_last += cnt
#         change += abs(cnt - current[i])

#     if tot_last == 0: return 1
#     return change / tot_last



def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
        inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        return intersection / (w1 * h1 + w2 * h2 - intersection)

def get_mAP(frame1_objects, frame2_objects, iou_threshold=0.5):
    """
    Computes mAP for a single pair of consecutive frames, considering multiple object classes.

    Parameters:
    - frame1_objects: List of objects in frame 1 [(frame_num, ID, class, bbox)]
    - frame2_objects: List of objects in frame 2 [(frame_num, ID, class, bbox)]
    - iou_threshold: Minimum IoU to consider a detection as TP

    Returns:
    - mAP (Mean Average Precision) for the given frame pair
    """
    classwise_ap = {}

    # Group objects by class
    frame1_by_class = {}
    frame2_by_class = {}

    for obj in frame1_objects:
        class_name = obj[2]
        if class_name not in frame1_by_class:
            frame1_by_class[class_name] = []
        frame1_by_class[class_name].append(obj)

    for obj in frame2_objects:
        class_name = obj[2]
        if class_name not in frame2_by_class:
            frame2_by_class[class_name] = []
        frame2_by_class[class_name].append(obj)

    # Get all unique classes
    all_classes = set(frame1_by_class.keys()).union(set(frame2_by_class.keys()))

    # Compute AP for each class
    for class_name in all_classes:
        tp = 0
        fn = 0
        fp = 0

        # Get objects of the current class
        frame1_class_objects = frame1_by_class.get(class_name, [])
        frame2_class_objects = frame2_by_class.get(class_name, [])

        # Map object ID to bounding box
        frame1_dict = {obj[1]: obj[3] for obj in frame1_class_objects}  # {ID: bbox}
        frame2_dict = {obj[1]: obj[3] for obj in frame2_class_objects}  # {ID: bbox}

        # Compute TP and FN
        for obj_id, bbox1 in frame1_dict.items():
            if obj_id in frame2_dict:
                bbox2 = frame2_dict[obj_id]
                iou = compute_iou(bbox1, bbox2)
                if iou >= iou_threshold:
                    tp += 1  # True positive
                else:
                    fn += 1  # False negative (object exists but IoU is low)
            else:
                fn += 1  # False negative (object is missing)

        # Compute FP (objects in frame2 but not in frame1)
        fp = len(set(frame2_dict.keys()) - set(frame1_dict.keys()))

        # Compute Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Generate Precision-Recall curve
        recalls = np.linspace(0, 1, 11)
        precisions = [precision if recall >= r else 0 for r in recalls]

        # Compute AP (Average Precision)
        ap = auc(recalls, precisions)
        classwise_ap[class_name] = ap

    # Compute Mean Average Precision (mAP) across all classes
    map_score = np.mean(list(classwise_ap.values())) if classwise_ap else 0

    return map_score




#############################################################################

#############################################################################
#Returns the accuracy and list of sent frames indices for a given thresh
def test_for_thresh(frames, thresh, all_objects, sets, pathForRes):
    
    if(frames[0][1]+30 not in all_objects.keys()): 1, [], 1 
    
    # object_vec = [all_objects[frame] for frame in range(frames[0][1] - 1,frames[0][1] + 29)]
    # if(len(object_vec) != 30): 1, [], 1 
    try:
        object_vec = [all_objects[frame] for frame in range(frames[0][1] - 1, frames[0][1] + 29)]
        if len(object_vec) != 30: 
            return 1, [], 1
    except KeyError:
        # Handle case where a frame in the range doesn't exist
        return 1, [], 1
    frames_sent = 1
    lost_objects = 0
    IOU_retained = 0
    reqd_frames = []
    reqd_indices = []
    mAP = []
    map = 0
    reqd_frames.append(frames[0][0])
    reqd_indices.append(frames[0][1])
    last_selected = 0
    for i in range(0, 29): 
        if i + 1 >= len(frames): 
            break
        edg1 = get_frame_edge(frames[i][0])
        edg2 = get_frame_edge(frames[i + 1][0])
        if(i >= len(frames)): break
        if(len(frames[i]) == 0): break
        # if(area_frame_diff(frames[i + 1][0], frames[i][0]) <= thresh):
        
        if(frame_edge_diff(edg1, edg2) <= thresh):
            # get_IOU_retain
            map  = get_mAP(object_vec[last_selected], object_vec[i + 1])
            mAP.append(map)
            # IOU_retained += get_IOU_retain(object_vec[last_selected], object_vec[i + 1])
        else:
            frames_sent += 1
            last_selected = i + 1
            reqd_frames.append(frames[i + 1][0])
            reqd_indices.append(frames[i + 1][1])

    avg_mAP = np.mean(mAP) if mAP else 0
    acc = avg_mAP
    # acc = IOU_retained/29
    # print("Frames Selected:", frames_sent)
    path_for_segs = str(pathForRes + '/frames/segment_' + str(sets).zfill(5))
    path_for_vid_segs = str(pathForRes + '/seg')

    Path(path_for_segs).mkdir(parents=True, exist_ok=True)
    Path(path_for_vid_segs).mkdir(parents=True, exist_ok=True)

    # for i, img in enumerate(reqd_frames):
    #     # name = 'frame_' + i÷/
    #     im1 = Image.fromarray(reqd_frames[i])
    #     im1.save(path_for_segs + '/' + f"frame_{str(i).zfill(5)}.bmp")

    # # print(">>>>>>>>>>>>> Generating Segments")
    # path_for_vid = f'{path_for_segs}' + '/frame_%05d.bmp'
    # command = f'ffmpeg -hide_banner -loglevel panic -framerate 30 -i {path_for_vid} -c:v libx264 {path_for_vid_segs}/segment_{str(sets).zfill(5)}'
    # os.system(command)
    
    final_frames.extend(reqd_indices[:])
    frames_done =len(reqd_frames)
    del frames,reqd_frames
    return acc, reqd_indices ,frames_done
#############################################################################



#########################################################################
#main driver code
def driver(pathVid, pathYolo, accPassed,  pathForRes, saveSegements,saveFile):

    objects = {}
    with open(pathYolo) as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0].isnumeric():
                frame_num = int(parts[0])
                object_id = int(parts[1])
                class_name = parts[2]
                bbox = [float(x) for x in parts[3:]]
                objects.setdefault(frame_num, []).append([frame_num,object_id,class_name,bbox])

    print(len(objects))


    vidObj = cv2.VideoCapture(pathVid)
    if not vidObj.isOpened():
        # print("Video not readable")
        return
    seg_no = 0
    fps = int(vidObj.get(cv2.CAP_PROP_FPS))
    width  = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_done = 0
    kmeans, thresh_of_cluster, segment_vars, acc_cali, segments,seg_no = calibrate(objects, vidObj, frames_done, fps, width, height, accPassed, pathForRes, saveSegements,seg_no)
    print("\nFiltering...............")
    
    frames_done += segments * fps
    calis = 1
    acc = 0
    sets = segments
    non_cali_segs = 0
    total_elapsed_time = 0
    final_filtered_frames = []
    acc_list = []
    
    while True:
        # vidObj = cv2.VideoCapture(pathVid)
        if(sets >= len(objects)): break
        frame_list = []
        for i in range(30):
            success, img1 = vidObj.read()
            if not success:
                break
            frames_done += 1
            frame_list.append((img1, frames_done))

        if(len(frame_list) != 30): break
        diffs = []
        
        for i in range(0, 29):
            edg1 = get_frame_edge(frame_list[i][0])
            edg2 = get_frame_edge(frame_list[i + 1][0])
            # if(frame_edge_diff(edg1, edg2) <= thresh):
            diffs.append(frame_edge_diff(edg1, edg2))
            # diffs.append(area_frame_diff(frame_list[i + 1][0], frame_list[i][0]))

        cluster = get_cluster(kmeans, segment_vars, diffs)
        if(cluster == -1):
            if(sets + 120 >= len(objects)): break
            res, thresh_of_cluster, segment_vars, acc_cali, segments, seg_no = calibrate(objects, vidObj, frames_done, fps, width, height, accPassed, pathForRes, saveSegements,seg_no)
            frames_done += segments * fps
            sets += segments
            calis += 1
            if(res != -1): kmeans = res
            # print(kmeans.labels_)
            continue
        
        t1 = time.time()
        sets += 1
        seg_no += 1
        print("segment number",seg_no)
        non_cali_segs += 1
        thresh = thresh_of_cluster[cluster]
        if len(frame_list) != 30:
            break
        acc_for_set,reqd_frames,f = test_for_thresh(frame_list, thresh, objects, sets, pathForRes)
        acc += acc_for_set
        print("ACC Now:", acc / non_cali_segs)
        acc_list.append(acc / non_cali_segs)
        frames_done += f
        final_frames.extend(reqd_frames[:])
        final_filtered_frames.extend(reqd_frames[:])
        # if not saveSegements: continue
        # video_path = pathForRes + '/filtered_segs/segment_' + str(sets) + '.avi'
        # writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        # print("thresh for cluster:", thresh)
        # print("Acc for set:", acc_for_set)
        # for img in reqd_frames:
        #     writer.write(img)

        t2 = time.time()
        elapsed_time = t2 - t1
        total_elapsed_time += elapsed_time


    # for i, img in enumerate(final_filtered_frames):
    #     # name = 'frame_' + i÷/
    #     im1 = Image.fromarray(final_filtered_frames[i])``
    #     im1.save(pathForRes + '/frames/' + f"frame_{str(i).zfill(4)}.bmp")

    acc = acc / non_cali_segs
    total_frames = sets * fps

    # video_path = '/Users/pr0hum/Desktop/ReductoVids/compressedVids/Entire' + '.avi'
    # writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    # for img in enumerate(final_frames):
    #     writer.write(img)


    


    res = { "acc": acc,
            "No. final frames": len(final_filtered_frames),
            "non calibration segments": non_cali_segs,
            "no. calibrations": calis,
            "Total Elapsed Time": total_elapsed_time
        }

    acc_list.append(acc)
    # csv = acc_list.copy()
    # csv.append(calis)
    # with open("accuracy_stats.csv", 'a') as file:
    #     file.write(pathVid.split('/')[-1].split('.')[0] + ",")  # Write video name
    #     file.write(",".join(map(str, csv)) + "\n")
    # joblib.dump(res, "AITr5S3C10.pkl")

    data = {}
    data['name'] = pathVid
    data['segments'] = acc_list
    data['final_acc'] = acc
    data['final_frames'] = final_filtered_frames
    data['non_cali_segs'] = non_cali_segs
    data['calibrations'] = calis
    data['total_segs'] = sets
    
    filePath = f'{saveFile}.json'
    
    
    if os.path.exists(filePath):
        # Load existing data
        with open(filePath, "r") as file:
            try:
                jsonData = json.load(file)
            except json.JSONDecodeError:
                jsonData = []  # If file is empty or corrupted, start fresh
    else:
        jsonData = []  # If file doesn't exist, start fresh

    # Ensure jsonData is a list before appending
    if not isinstance(jsonData, list):
        raise ValueError("JSON file must contain a list at the top level")

    # Append new object
    jsonData.append(data)

    # Save updated jsonData
    with open(filePath, "w") as file:
        json.dump(jsonData, file, indent=4,separators=(",",":"))
    acc_list.append(acc)


    print("acc:", acc)
    print("No. final frames", len(final_filtered_frames))
    print("non calibration segments", non_cali_segs)
    print("no. calibrations:", calis)
    print("Total Elapsed Time:", total_elapsed_time)
#############################################################################
 
# call driver with the arguments given below
# video path, yolo label path, acc for thresh, path to result folder, saveSegment
driver('videos/1.mp4', 'mp4_deepsort_txt/1.txt', 0.9, 'mp4_normal_res_bbox', True ,'results/result')




# ############################################################################################################################

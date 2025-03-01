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
import json
# import joblib

################################################
threshes = [ i / 100000 for i in range(0, 10001)]
test_weights = [0 for _ in range(29)]
final_frames = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
################################################

INPUT_VIDEO = "videos/1.mp4"
INPUT_TXT = "mp4_yolo_txt/1.txt "
OUTPUT_PATH = "mp4_normal_res_count"
OUTPUT_JSON = "results/count_result"

#############################################################################
# methods for differences
def frame_edge_diff(edge, prev_edge):
    """Calculate edge-based frame difference."""
    total_pixels = edge.shape[0] * edge.shape[1]
    frame_diff = cv2.absdiff(edge, prev_edge)
    frame_diff = cv2.threshold(frame_diff, 21, 255, cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(frame_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed

def get_frame_edge(frame):
    """Extract edges from a given frame using Canny edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edge = cv2.Canny(blur, 101, 255)
    return edge


def frame_pixel_diff(frame, prev_frame):
    """Calculate pixel-wise difference between two frames."""
    total_pixels = frame.shape[0] * frame.shape[1]
    frame_diff = cv2.absdiff(frame, prev_frame)
    frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)
    frame_diff = cv2.threshold(frame_diff, 21, 255, cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(frame_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed

def area_frame_diff(frame, prev_frame):
    """Calculate difference in area between two frames using contours."""
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
    """Extract frames from a video and save them as images."""
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
    """Convert a sequence of images into a video."""
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
    
#############################################################################

#############################################################################
#Search for the best thresh during caliberation
def get_max_acc_vector(frames, all_objects, accPassed):
    
    ctr = 0
    if(len(all_objects) <= frames[0][1] + 30): 0, 1, [], 1  # value of frames[0][1] is 1.
    object_vec = all_objects[frames[0][1] - 1: frames[0][1] + 29] # [[8 classes][][]....[30th frame]] of all object for frames 0 to 30
    if(len(object_vec) < 30): return 0,1,[],1
    l = 0
    r = len(threshes) - 1
    max_poss_thresh = 0
    acc = 0
    lost = 0
    while(l <= r):
        m = (l + r) // 2
        lost_objects = 0
        last_selected = 0
        for i in range(0, 29): 
            if(i >= len(frames)): break
            if(m >= len(threshes)): break
            if(area_frame_diff(frames[i + 1][0], frames[i][0]) <= threshes[m]):
                if(i + 1 >= len(object_vec) or last_selected >= len(object_vec)): break
                lost_objects += get_lost_objects_r(object_vec[last_selected], object_vec[i + 1])
            else:
                last_selected = i + 1

        acc_for_thresh = 1 - lost_objects / 29 
        if(acc_for_thresh >= accPassed) :
            if(max_poss_thresh < threshes[m]):
                max_poss_thresh = threshes[m]
                acc = acc_for_thresh
            l = m + 1
        else:
            r = m - 1

    frames_sent = 1
    diffs = []
   
    for i in range(0, 29):
        area_dif = area_frame_diff(frames[i + 1][0], frames[i][0])
        diffs.append(area_dif)
        if(area_dif > max_poss_thresh):
            frames_sent += 1

    # print(max_poss_thresh, acc, frames_sent)
    return frames_sent,max_poss_thresh, diffs, acc
#############################################################################


def get_dist(centre, point):
    """Calculate Euclidean distance between two points."""
    dist = 0
    for i in range(29):
        dist += ((point[i] - centre[i]) * (point[i] - centre[i]))

    dist = sqrt(dist)
    return dist

def get_radius(centre, points):
    """Calculate maximum distance from the centroid to any point in the cluster."""
    dist = -1
    for p in points:
        t_dist = get_dist(centre, p)
        dist = max(dist, t_dist)
    return dist


#############################################################################
#calibration
def calibrate(P, vidObj, frames_done, fps, width, height, accPassed, resPath, saveSegments):
    """Perform calibration to determine optimal thresholds."""
    print("calibrating.....")
    frames_done_cal = 0
    frame_list = [] # contains tuple of img and frames_done + frames_done_cal
    diff_segment_vector = []
    segment_threshes = []
    acc = 0
    sets = 0
    used_frames = 0
    while frames_done_cal < 120 * fps: #offline profiling for 2 min 
        for i in range(30):
            success, img = vidObj.read()
            if not success: break
            frames_done_cal += 1
            frame_list.append((img, frames_done + frames_done_cal))

        if len(frame_list) != 30: 
            break
        f, thresh_for_vect, diffs_for_vect, acc_here = get_max_acc_vector(frame_list, P, accPassed)
        used_frames += f
        
        
        if len(diffs_for_vect) < 29:
            diffs_for_vect.extend([0] * (29 - len(diffs_for_vect)))
        diff_segment_vector.append(diffs_for_vect[:])
        segment_threshes.append(thresh_for_vect)
        frame_list.clear()
        acc += acc_here
        sets += 1
        # if not saveSegments: continue
        # video_path = resPath + '/cali_segs/segment_' + str(sets) + '.avi'
        # print("number of frames:", len(reqd_frames))
        # writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        # for img in reqd_frames:
        #     writer.write(img)

    X = np.array(diff_segment_vector)
    if(len(X) < 5): return -1, -1, -1, -1, -1
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    size_of_cluster = [0 for i in range(5)] #Stores the number of segments in each cluster.
    thresh_of_cluster = [0 for i in range(5)]#Stores the average threshold for each cluster.
    
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

    return kmeans, thresh_of_cluster, segment_vars, acc , len(X)
    
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
def get_lost_objects_r(last_selected, current):
    """Calculate number of objects lost between two frames."""
    change = 0
    tot_last = 0
    for i, cnt in enumerate(last_selected):
        tot_last += cnt
        change += abs(cnt - current[i])

    if tot_last == 0: return 1
    # return change / tot_last
    return change

def get_objects_r(current):
    total_obj = 0
    for i in current:
        total_obj += i
    
    if total_obj == 0: return 1
    return total_obj
#############################################################################



#############################################################################
#Returns the accuracy and list of sent frames for a given thresh
def test_for_thresh(frames, thresh, all_objects, sets, pathForRes):
    object_vec = all_objects[frames[0][1] - 1: frames[0][1] + 29]
    frames_sent = 1
    lost_objects = 0
    reqd_frames = []
    reqd_indices =[]
    reqd_frames.append(frames[0][0])
    reqd_indices.append(frames[0][1])
    last_selected = 0
    last_selected_count = 0
    act_lost_objects = 0
    last_selected_frame = 0
    changed_objects = 0
    total_object_count = 0
     
    for i in range(0, 29): 
        if(i >= len(frames)): break
        if(len(frames[i]) == 0): break
        
        if(len(object_vec) < 30): 
            print(len(object_vec))
            return 0,[]
        
        total_object_count += get_objects_r(object_vec[i])
        
        if(area_frame_diff(frames[i + 1][0], frames[i][0]) <= thresh):
            
            changed_objects += abs(get_objects_r(object_vec[last_selected]) -  get_objects_r(object_vec[i+1]))
            
        else:
            frames_sent += 1
            last_selected = i + 1
            last_selected_count = 0
            reqd_frames.append(frames[i + 1][0])
            reqd_indices.append(frames[i + 1][1])
        
    if( total_object_count == 0):
        acc = 0
    else:
        acc = 1 - changed_objects/total_object_count
    # print("Frames Selected:", frames_sent)
    path_for_segs = str(pathForRes + '/frames/segment_' + str(sets).zfill(5))
    path_for_vid_segs = str(pathForRes + '/seg')

    Path(path_for_segs).mkdir(parents=True, exist_ok=True)
    Path(path_for_vid_segs).mkdir(parents=True, exist_ok=True)


    # uncomment to save the segments 
    # for i, img in enumerate(reqd_frames):
    #     # name = 'frame_' + i÷/
    #     im1 = Image.fromarray(reqd_frames[i])
    #     im1.save(path_for_segs + '/' + f"frame_{str(i).zfill(5)}.bmp")

    # print(">>>>>>>>>>>>> Generating Segments")
    # path_for_vid = f'{path_for_segs}' + '/frame_%05d.bmp'
    # command = f'ffmpeg -hide_banner -loglevel panic -framerate 30 -i {path_for_vid} -c:v libx264 {path_for_vid_segs}/segment_{str(sets).zfill(5)}'
    # os.system(command)
    
    final_frames.extend(reqd_frames[:])
    del frames,reqd_frames
    return acc, reqd_indices
#############################################################################



#############################################################################
#main driver code
def driver(pathVid, pathYolo, accPassed,  pathForRes, saveSegements,saveFile):
  
    acc_list = []
    objects = []
    with open(pathYolo) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            freq = [0 for i in range(8)]
            objects_in_frame = 0
            line = line.split(' ')
            for j in line:
                if(j.isnumeric()):
                    freq[int(j)] += 1
            objects.append(freq[:])



    vidObj = cv2.VideoCapture(pathVid)
    if not vidObj.isOpened():
        # print("Video not readable")
        return
    
    fps = int(vidObj.get(cv2.CAP_PROP_FPS))
    width  = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_done = 0
    kmeans, thresh_of_cluster, segment_vars, acc_cali, segments = calibrate(objects, vidObj, frames_done, fps, width, height, accPassed, pathForRes, saveSegements)
    frames_done += segments * fps
    calis = 1
    acc = 0
    sets = segments
    cal_sets = segments
    non_cali_segs = 0
    total_elapsed_time = 0
    final_filtered_frames = []

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
        
            diffs.append(area_frame_diff(frame_list[i + 1][0], frame_list[i][0]))

        cluster = get_cluster(kmeans, segment_vars, diffs)
        if(cluster == -1):
            if(sets + 120 >= len(objects)): break
            res, thresh_of_cluster, segment_vars, acc_cali, segments = calibrate(objects, vidObj, frames_done, fps, width, height, accPassed, pathForRes, saveSegements)
            frames_done += segments * fps
            sets += segments
            cal_sets +=segments
            calis += 1
            if(res != -1): kmeans = res
            # print(kmeans.labels_)
            continue
        
        t1 = time.time()
        sets += 1
        non_cali_segs += 1
        thresh = thresh_of_cluster[cluster]
        if len(frame_list) != 30:
            break
        acc_for_set, reqd_frames = test_for_thresh(frame_list, thresh, objects, sets, pathForRes)
        acc += acc_for_set
        print("ACC NOW",acc_for_set)
        acc_list.append(acc_for_set)
        frames_done += len(reqd_frames)
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

    data = {}
    data['name'] = pathVid
    data['segments'] = acc_list
    data['final_acc'] = acc
    data['final_frames'] = final_filtered_frames
    data['non_cali_segs'] = non_cali_segs
    data['calibrations'] = calis
    data['calibrated_segs'] = cal_sets
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
        json.dump(jsonData, file, indent=4,separators=(",", ":"))
    acc_list.append(acc)

    # with open(saveFile, 'a') as file:
    #     file.write(pathVid.split('/')[-1].split('.')[0] + ",")  # Write video name
    #     file.write(",".join(map(str, acc_list)) + "\n")

    # joblib.dump(res, "AITr5S3C10.pkl")

    print("acc:", acc)
    print("No. final frames", len(final_filtered_frames))
    print("non calibration segments", non_cali_segs)
    print("no. calibrations:", calis)
    print("Total Elapsed Time:", total_elapsed_time)
    print("Total segments",sets)
    print("Total calibrated segments",cal_sets)
    
############################################################################################
 
# call driver with the arguments given below

if __name__ == "__main__":
    encodedVideoFileName = 'encoded_videos/AINormal_mp4'
    txtFileName = 'yolo_txt_files/AINormal_mp4/AITr5S1C4__q.txt'
    segmentFileName = 'segments_files/AINormal_mp4/AITr5S1C4__q/'
    csvFileName = 'counting_AlNormal.csv'

    driver(INPUT_VIDEO,INPUT_TXT, 0.9, OUTPUT_PATH, True, OUTPUT_JSON)


# ############################################################################################################################

"""
This README file is intended to point the reader to the use of the notebook `piano_detection.ipynb` to test out the system.
The following code is not intended to be run in this file, but to guide the reader to run the same code in the notebook, with documented
explanations. 

Please note that each cell of the notebook also contains step-by-step executions of the system along with the implementations 
of the used routines. At the end of the notebook, the system is also summarized to enable running on multiple examples.
"""
# -------------------------------------------------------------------------------------------------


# Firstly, the frames of the video used to test the system must be extracted.
# The test video "clair_de_lune.mp4" should be in the folder "videos".
VIDEOS_DIR_PATH = "videos/"
FRAMES_DIR_PATH = "frames/"
TEST_VIDEO = "clair_de_lune.mp4"
extractFrames(VIDEOS_DIR_PATH + TEST_VIDEO, FRAMES_DIR_PATH, 10)


# -------------------------------------------------------------------------------------------------


# We will test the system on one example frame. Firstly, preprocess the frame.
# The following two variables can be tuned, with the binary threshold being very important
# depending on the brightness and lighting in the image. The current values work for this example.
TEST_FRAME = "frame3.jpg"
downsize_scale = 10
binary_threshold_for_keyboard_area = 75
frame_preprocessed = preProcessFrame(FRAMES_DIR_PATH + TEST_FRAME)
frame_thresholded = binarizeFrame(downsizeFrame(frame_preprocessed, downsize_factor=downsize_scale), binary_threshold=binary_threshold_for_keyboard_area)


# -------------------------------------------------------------------------------------------------


# Next, the keyboard area is detected
# Apply Scharr operator to find horizontal edges
frame_scharr = applyScharr(frame_thresholded, is_horiz=True)

# Apply Hough transform to find horizontal lines
hough_lines = findHoughLines(frame_scharr)

# Take average of lines to get 3 defining lines
# Importantly, depending on the quality of the binarization result, if there are other lines or
# line groups in the image, this may cause this routine to fail to find the 3 intended groups.
defining_lines = averageLines(hough_lines, frame_scharr)

# Rotate frame if need be
frame_keyboard_area_final, defining_lines = rotateFrameMaybe(frame_scharr, defining_lines)


# -------------------------------------------------------------------------------------------------


# Then, the keys are detected.
# The binary threshold here is again important, and the chosen value works for this example.
binary_threshold_for_keys = 80
_, frame_binarized = cv2.threshold(frame_preprocessed, binary_threshold_for_keys, 255, cv2.THRESH_BINARY)

# Scale up defining lines. Get potential edges
upscale_defining_lines = [(x1, y1 * downsize_scale, x2, y2 * downsize_scale) for x1, y1, x2, y2 in defining_lines]
potential_edges = detectKeyEdges(frame_binarized, upscale_defining_lines)

# Label keys
keys = labelKeys(potential_edges)

# Iterpolate keys
real_keys, real_edges = interpolateWhiteKeys(keys)

# Apply Scharr to find vertical edges
sectioned_frame_binarized = frame_binarized[top[1]:bottom[1], :]
frame_scharr_vert = applyScharr(sectioned_frame_binarized, is_horiz=False)

# Apply Hough transform to find vertical lines
# A small theta space is searched to reduce runtime, knowing that lines should be vertical.
# Empirically, finding the vertical lines are not as problematic as horizontal ones. This routine
# should find 1 to a few lines associated with each edge
hough_lines_vert = findHoughLines(frame_scharr_vert, theta_min=-np.pi/24, theta_max=np.pi/24, theta_step=np.pi/90)
hough_lines_vert = [(x1, 0, x2, frame_scharr_vert.shape[1]) for x1, _, x2, _ in hough_lines_vert]

# Match edges to lines
edge_line_pairs = pairEdgeLines(real_edges, hough_lines_vert)

# Get the keys with boundaries attached
keys_with_boundaries = findKeyBoundaries(real_keys, edge_line_pairs, upscale_defining_lines)

# Get final named, bounded keys
keys_with_names_and_boundaries = nameKeys(keys_with_boundaries)

# Visualize final result
frame_showcase = cv2.cvtColor(cv2.imread(FRAMES_DIR_PATH + TEST_FRAME), cv2.COLOR_BGR2RGB)
for name, key, boundaries in keys_with_names_and_boundaries:
    # Draw bounding rect
    color = (255, 0, 0) if key[0] == "B" else (0, 255, 0)
    top_left, bottom_right = boundaries
    cv2.rectangle(frame_showcase, top_left, bottom_right, color, thickness=2)

    # Add key name
    text_position = (top_left[0], bottom_right[1] + 20) if key[0] == "W" else (top_left[0], top_left[1] - 10)
    cv2.putText(frame_showcase, name, text_position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)

plt.figure(figsize=(10, 8))
plt.imshow(frame_showcase)
plt.title("Keyboard & Keys Detection Final Result")
plt.show()
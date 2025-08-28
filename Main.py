import cv2
import numpy as np
import pytesseract
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from timetable_to_ics import timetable_to_ics

min_conf = 0.20
dist_thresh = 10
min_size = 2
img_path = "D:/major/test_timetable3.jpg"

def table_extraction(image_path):
    img = cv2.imread(image_path)

    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # Step 3: Detect horizontal lines
    horizontal = thresh.copy()
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size,1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # Step 4: Detect vertical lines
    vertical = thresh.copy()
    rows = vertical.shape[0]
    vertical_size = rows // 30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    # Step 5: Combine lines
    mask = horizontal + vertical

    # Step 6: Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Assume largest contour is table
    areas = [cv2.contourArea(c) for c in contours]
    max_idx = np.argmax(areas)
    cnt = contours[max_idx]
    x, y, w, h = cv2.boundingRect(cnt)

    # Step 8: Crop the table
    table_img = img[y:y+h, x:x+w]
    return table_img


def extract_cells(data):
    coords = []
    ocr_text = []
    for i, txt in enumerate(data['text']):
        if txt.strip():
            txt = txt.strip().rstrip(".")
            if len(txt) == 1 and txt != "-":
                 continue
            conf = data['conf'][i]
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            

            if conf > min_conf:
                ocr_text.append(txt if txt != "-" else "None")
                coords.append((x,y,w,h))
    return ocr_text, coords


table_img = table_extraction(img_path)
table_img_gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.adaptiveThreshold(~table_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
results = pytesseract.image_to_data(cv2.cvtColor(thresh_img, cv2.COLOR_BGR2RGB),output_type=pytesseract.Output.DICT)
ocr_text, coords = extract_cells(results)

xCoords = [(c[0]+c[2]/2, 0) for c in coords]
clustering = AgglomerativeClustering(
	n_clusters=None,
	metric="manhattan",
	linkage="complete",
	distance_threshold=dist_thresh,)

clustering.fit(xCoords)
sortedClusters = []

# loop over all clusters
for l in np.unique(clustering.labels_):
	# extract the indexes for the coordinates belonging to the current cluster
	idxs = np.where(clustering.labels_ == l)[0]
	# verify that the cluster is sufficiently large
	if len(idxs) > min_size:
		# compute the average x-coordinate value of the cluster and update our clusters list with the current label and the average x-coordinate
		avg = np.average([coords[i][0] for i in idxs])
		sortedClusters.append((l, avg))
# sort the clusters by their average x-coordinate and initialize our
# data frame
sortedClusters.sort(key=lambda x: x[1])

df = pd.DataFrame()

# loop over the clusters again, this time in sorted order
for col_idx, (l, _) in enumerate(sortedClusters):
    # extract the indexes for the coordinates belonging to the current cluster
    idxs = np.where(clustering.labels_ == l)[0]
    # extract the y-coordinates from the elements in the current cluster, then sort them from top-to-bottom
    yCoords = [coords[i][1] for i in idxs]
    sortedIdxs = idxs[np.argsort(yCoords)]

    # generate a random color for the cluster
    color = np.random.randint(0, 255, size=(3,), dtype="int")
    color = [int(c) for c in color]
    
    # draw bounding boxes
    for i in sortedIdxs:
        (x, y, w, h) = coords[i]
        cv2.rectangle(table_img, (x, y), (x + w, y + h), color, 2)

    # extract OCR'd text
    cols = [ocr_text[i].strip() for i in sortedIdxs]

    # Patch: insert a header cell if missing in the first column
    if col_idx == 0 and len(cols) == 5:
        cols = ["Day/Time"] + cols  # add header to align length

    # create DataFrame column
    currentDF = pd.DataFrame({cols[0]: cols[1:]})
    df = pd.concat([df, currentDF], axis=1)

# replace NaN values with an empty string and then show a nicely
# formatted version of our multi-column OCR'd text
df.fillna("None", inplace=True)
print(df)



# Convert to ICS file (specify your timezone)
output_file = timetable_to_ics(df, timezone='Asia/Kolkata', output_file='my_schedule.ics')

print(f"Calendar file created: {output_file}")
print("You can now import this file into Google Calendar, Apple Calendar, Outlook, etc.")
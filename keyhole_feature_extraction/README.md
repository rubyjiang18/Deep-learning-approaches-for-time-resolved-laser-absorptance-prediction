# Vapor Depression Geometric Feature Extraction from Segmented Masks

Feature extraction is straight and only requires a good segmentation mask. For example, using the following example masks, we show that it is just array manipulations. In the following mask, 0 indicate a black pixel (background), and 1 indicate a white pixel (vapor depression). 

000000 \
001100 \
001110 \
001100 \
001000 \
000000 

The features we can extracts are 
1. number white pixel (can be converted to area, the length of each pixel is 1.923 um in our experiment, so number of white pixel × 1.923 × 1.923 gives you the area of keyhole)
```
np.sum(mask == 1)
```
2. keyhole depth
We first identify all row index where there is at least one white pixel 
```
num_white_pixel_row = np.sum(mask == 1, 1) 
nonzero_index = [i for i, e in enumerate(num_white_pixel_row) if e != 0]
depth = len(nonzero_index)
```
3. Keyhole width at various depth
```
width_half = num_white_pixel_row[nonzero_index[2]]
```
4. Keyhole aspect ratio (width/depth)
5. Keyhole perimete
```
contours,hierarchy = cv.findContours(mask, 1, 2)
if contours:
    cnt = contours[0]
    perimeter = cv.arcLength(cnt,True)
```
6. Front wall angle

See the file for detailed solution.

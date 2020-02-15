import cv2
import numpy as np
import statistics
from time import sleep
from os import walk
import re

def get_image():
    filename = f'images/test{n}.png'
    img = cv2.imread(filename)
    return img

def background_color(img):
    colors, count = np.unique(img.reshape(-1,img.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

def read_number(img):
    pass

def read_puzzle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # extract bounding rect
    border_mask = cv2.inRange(img, np.array([0, 245, 245]), np.array([255, 255, 255]))
    contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
    grid_rect = max(bounding_rects, key=lambda t: t[2]*t[3])
    (gx, gy, gw, gh) = grid_rect
    print(f'bounding rectangle size: ({gw}, {gh})')

    # get grid dimension (r, c in {5, 10, 15, ...})
    grid_img = img[gy:gy+gh, gx:gx+gw].copy()
    grid_background_color = background_color(grid_img)
    grid_img_mask = cv2.inRange(grid_img, grid_background_color - 20, grid_background_color + 20)
    contours, _ = cv2.findContours(grid_img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
    cws, chs = [t[2] for t in bounding_rects], [t[3] for t in bounding_rects]
    cw, ch = statistics.median_grouped(cws) + 1, statistics.median_grouped(chs) + 1
    r, c = int(gw / cw),  int(gh / ch)
    cw, ch = gw / c, gh / r
    print(f'grid dimension: {r} x {c}')

    # load digit templates
    digits = []
    filenames = []
    for (_, _, fn) in walk('images/digits'):
        filenames.extend(fn)
        break
    for fn in filenames:
        digit = int(re.match('\d+', fn).group())
        template_img = cv2.imread(f'images/digits/{fn}')
        template_img = cv2.split(cv2.cvtColor(template_img, cv2.COLOR_RGB2HSV))[2]
        digits.append((template_img, digit))
    digits = sorted(digits, key=lambda t: t[1])

    # read numbers
    clues_h = [None] * r
    for j in range(r):
        for i in range(100):
            ys, ye = int(gy + ch * j + 1), int(gy + ch * j) + int(ch) - 1 
            xs, xe = int(gx - cw * i) - int(cw), int(gx - cw * i)
            if xs < 0:
                break

            clue_img = img[ys:ye,xs:xe].copy()
            ratio, width = (ye-ys)/(xe-xs), 100
            clue_img = cv2.resize(clue_img, (width, int(width*ratio)), interpolation=cv2.INTER_LINEAR)

            col = background_color(clue_img)
            clue_back_mask = cv2.inRange(clue_img, col - 10, col +10)
            if np.sum(clue_back_mask == 255) > clue_back_mask.shape[0] * clue_back_mask.shape[1] * 0.90:
                break
      
            clue_hsv = cv2.cvtColor(clue_img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(clue_hsv)
            vl, vu = np.percentile(v, 2), np.percentile(v,98)
            v = (v-vl) * 255 / (vu-vl) 
            
            black = cv2.inRange(v, 0, 40)
            contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
            x = min(t[0] for t in bounding_rects)
            y = min(t[1] for t in bounding_rects)
            w = max(t[2] + t[0] for t in bounding_rects) - x 
            h = max(t[3] + t[1] for t in bounding_rects) - y

            v = np.uint8(np.clip(v, 0, 255))
            col = background_color(clue_hsv)
            back_mask = cv2.bitwise_not(cv2.inRange(clue_hsv, col - 10, col + 10))
            v = cv2.bitwise_and(v, v, mask=back_mask)
            v_th = 0.8
            v = np.clip(1 + (v.astype(float) / 255 - v_th) * 3, 0, 1)
            v = (v * 255).astype(np.uint8)
            v_clipped = v[y:y+h,x:x+w]
            v_clipped = cv2.resize(v_clipped, None, fx=0.5, fy=0.5)

            v = cv2.resize(v, None, fx=0.5, fy=0.5)
            scores_x = []
            scores_xx = []
            for template_img, digit in digits:
                res = cv2.matchTemplate(v, template_img, cv2.TM_CCOEFF_NORMED)
                (_, score, _, _) = cv2.minMaxLoc(res)
                if digit <= 9:
                    scores_x.append((score, digit))
                else:
                    scores_xx.append((score, digit))

            if w / h > 1.0:
                digit = max(scores_xx, key=lambda t: t[0])[1]
            else:
                digit = max(scores_x, key=lambda t: t[0])[1]

            #cv2.rectangle(clue_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.imwrite(f'output/{n}h{j}_{i}_r{digit}.png', v_clipped)

            clues_h.append((i,j))
            break
    print(len(clues_h))
    cv2.waitKey(0)

for n in [3]:
# for n in [6]:
    read_puzzle(get_image())
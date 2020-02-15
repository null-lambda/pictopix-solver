from pathlib import Path
from os import walk
from zipfile import ZipFile
import re
from statistics import median_grouped
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import nonogram


def get_image():
    filename = f'images/test{n}.png'
    img = cv2.imread(filename)
    return img


def background_color(img):
    colors, count = np.unique(
        img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def merge_bounding_rects(rectangles):
    rectangles = list(rectangles)
    x = min(t[0] for t in rectangles)
    y = min(t[1] for t in rectangles)
    w = max(t[2] + t[0] for t in rectangles) - x
    h = max(t[3] + t[1] for t in rectangles) - y
    return x, y, w, h


def read_number(clue_img, digit_templates, index_str):
    clue_w, clue_h = clue_img.shape[1], clue_img.shape[0]
    ratio, new_h = clue_w / clue_h, 100
    clue_img = cv2.resize(clue_img, (int(new_h * ratio),
                                     new_h), interpolation=cv2.INTER_LINEAR)
    clue_hsv = cv2.cvtColor(clue_img, cv2.COLOR_RGB2HSV)

    h, w, _ = clue_hsv.shape
    i, j = np.indices((h, w))
    grid_indices = np.stack((i, j))
    x_vector = grid_indices.transpose(1, 2, 0).reshape(h * w, -1)
    y_vector = clue_hsv.reshape(h * w, 3).astype(float)
    reg = LinearRegression().fit(x_vector, y_vector)
    plane = (np.einsum('aij,ka->ijk', grid_indices, reg.coef_) + reg.intercept_)

    dist_abs = np.linalg.norm(abs(clue_hsv - plane), axis=2)

    d = np.mean(dist_abs ** 2) ** .5
    if d < 12:
        return None

    _, _, v = cv2.split(clue_hsv)
    v_lower, v_upper = np.percentile(v, 2), np.percentile(v, 98)
    v = (v - v_lower) * 255 / (v_upper - v_lower)

    black = cv2.inRange(v, 0, 40)
    contours, _ = cv2.findContours(
        black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = merge_bounding_rects(
        cv2.boundingRect(cnt) for cnt in contours)

    v = np.uint8(np.clip(v, 0, 255))
    col = background_color(clue_hsv)
    background_mask = cv2.bitwise_not(
        cv2.inRange(clue_hsv, col - 10, col + 10))
    v = cv2.bitwise_and(v, v, mask=background_mask)
    v_threshold = 0.8
    v = np.clip(1 + (v.astype(float) / 255 - v_threshold) * 3, 0, 1)
    v = (v * 255).astype(np.uint8)

    v = v[y:y+h, x:x+w]
    ratio, new_h = w / h, 28
    v = cv2.resize(v, (int(new_h * ratio), new_h))
    scale_factor = new_h / h
    x, y, w, h = map(lambda a: int(a * scale_factor), (x, y, w, h))
    x_pad = (50 - w) // 2
    v_paddings = cv2.copyMakeBorder(v, 2, 2, x_pad, x_pad, cv2.BORDER_CONSTANT)

    scores_x = []
    scores_xx = []
    for template_img, digit in digit_templates:
        #print(v_paddings.shape, template_img.shape)
        res = cv2.matchTemplate(v_paddings, template_img, cv2.TM_CCOEFF_NORMED)
        (_, score, _, _) = cv2.minMaxLoc(res)
        if digit <= 9:
            scores_x.append((score, digit))
        else:
            scores_xx.append((score, digit))
    score, digit = max(scores_xx if (w / h) > 1.0 else scores_x)
    Path("output").mkdir(parents=True, exist_ok=True)
    if score < 0.9:
        cv2.imwrite(f'output/p{int(100 * score)}_r{digit}_{index_str}.png', v)
    return digit


def read_puzzle(img):
    # extract bounding rect
    border_mask = cv2.inRange(img, np.array(
        [0, 245, 245]), np.array([255, 255, 255]))
    contours, _ = cv2.findContours(
        border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
    grid_rect = max(bounding_rects, key=lambda t: t[2]*t[3])
    (gx, gy, gw, gh) = grid_rect
    print(f'bounding rectangle size: ({gw}, {gh})')

    # get grid dimension (r, c in {5, 10, 15, ...})
    grid_img = img[gy:gy+gh, gx:gx+gw].copy()
    grid_background_color = background_color(grid_img)
    grid_img_mask = cv2.inRange(
        grid_img, grid_background_color - 20, grid_background_color + 20)
    contours, _ = cv2.findContours(
        grid_img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
    cws, chs = [t[2] for t in bounding_rects], [t[3] for t in bounding_rects]
    cw, ch = median_grouped(cws) + 1, median_grouped(chs) + 1
    r, c = int(gw / cw), int(gh / ch)
    cw, ch = gw / c, gh / r
    print(f'grid dimension: {r} x {c}')

    # load template images
    digits_template = []
    filenames = []
    for (_, _, fn) in walk('images/digits'):
        filenames.extend(fn)
        break
    for fn in filenames:
        if fn.endswith('.zip'):
            zfile = ZipFile(f'images/digits/{fn}', 'r')
            for name in zfile.namelist():
                digit = int(re.match(r'\d+', name).group())
                template_img = cv2.imdecode(
                    np.frombuffer(zfile.read(name), np.uint8), 1)
                template_img = cv2.split(cv2.cvtColor(
                    template_img, cv2.COLOR_RGB2HSV))[2]
                digits_template.append((template_img, digit))
            zfile.close()
        elif fn.endswith('.png'):
            digit = int(re.match(r'\d+', fn).group())
            template_img = cv2.imread(f'images/digits/{fn}')
            template_img = cv2.split(cv2.cvtColor(
                template_img, cv2.COLOR_RGB2HSV))[2]
            digits_template.append((template_img, digit))
    digits_template = sorted(digits_template, key=lambda t: t[1])

    # read number clues
    clues_h, clues_v = [[] for _ in range(r)], [[] for _ in range(c)]
    for j in range(r):
        for i in range(100):
            ys, ye = int(gy + ch * j + 1), int(gy + ch * j) + int(ch)
            xs, xe = int(gx - cw * i) - int(cw), int(gx - cw * i)
            if xs < 0:
                break
            clue_img = img[ys:ye, xs:xe].copy()
            # cv2.rectangle(img,(xs,ys),(xe,ye),(0,0,255,2))
            # cv2.imshow(f'1',img)
            digit = read_number(clue_img, digits_template,
                                f'{n}h{j:02d}{i:02d}')
            if digit is None:
                break
            clues_h[j].append(digit)
    for i in range(c):
        for j in range(100):
            xs, xe = int(gx + cw * i + 1), int(gx + cw * i) + int(cw)
            ys, ye = int(gy - ch * j) - int(ch), int(gy - ch * j)
            if ys < 0:
                break
            clue_img = img[ys:ye, xs:xe].copy()
            cv2.rectangle(img, (xs, ys), (xe, ye), (0, 0, 255, 2))
            digit = read_number(clue_img, digits_template,
                                f'{n}v{j:02d}{i:02d}')
            if digit is None:
                break
            clues_v[i].append(digit)
        cv2.waitKey(0)
    clues_h = tuple(tuple(clue[::-1]) for clue in clues_h)
    clues_v = tuple(tuple(clue[::-1]) for clue in clues_v)

    return clues_v, clues_h


if __name__ == "__main__":
    for n in [1, 2, 3, 4, 5, 6]:
        # for n in [6]:
        clues = read_puzzle(get_image())
        grid = nonogram.Nonogram(clues, callback=None).solve()
        print(clues)
        nonogram.print_grid(grid)

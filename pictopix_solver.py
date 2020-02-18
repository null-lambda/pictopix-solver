import re
import shutil
import time
from collections import deque
from os import walk
from pathlib import Path
from statistics import median_grouped
from time import sleep
from zipfile import ZipFile

import cv2
import numpy as np
import win32gui
from PIL import ImageGrab
from pynput import keyboard, mouse
from sklearn.linear_model import LinearRegression

import nonogram


def background_color(img):
    colors, count = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def merge_bounding_rects(rectangles):
    rectangles = list(rectangles)
    x = min(t[0] for t in rectangles)
    y = min(t[1] for t in rectangles)
    w = max(t[2] + t[0] for t in rectangles) - x
    h = max(t[3] + t[1] for t in rectangles) - y
    return x, y, w, h


def crop_or_pad_center(img, formatted_size):
    h, w = img.shape[0:2]
    fw, fh = formatted_size
    dw, dh = abs(fw - w), abs(fh - h)
    left_pad, right_pad = dw // 2, (dw + 1) // 2
    top_pad, bottom_pad = dh // 2, (dh + 1) // 2
    if w > fw:
        img_paddings = img[:, left_pad:(-right_pad)]
        left_pad, right_pad = 0, 0
    if h > fh:
        img_paddings = img[top_pad:(-bottom_pad), :]
        top_pad, bottom_pad = 0, 0
    img_paddings = cv2.copyMakeBorder(
        img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT
    )
    return img_paddings


def format_number_image(clue_img):
    clue_w, clue_h = clue_img.shape[1], clue_img.shape[0]
    ratio, new_h = clue_w / clue_h, 100
    clue_img = cv2.resize(
        clue_img, (int(new_h * ratio), new_h), interpolation=cv2.INTER_LINEAR
    )
    clue_hsv = cv2.cvtColor(clue_img, cv2.COLOR_RGB2HSV)

    ch, cw, _ = clue_hsv.shape
    i, j = np.indices((ch, cw))
    grid_indices = np.stack((i, j))
    x_vector = grid_indices.transpose(1, 2, 0).reshape(ch * cw, -1)
    y_vector = clue_hsv.reshape(ch * cw, 3).astype(float)
    reg = LinearRegression().fit(x_vector, y_vector)
    plane = np.einsum("aij,ka->ijk", grid_indices, reg.coef_) + reg.intercept_

    dist_abs = np.linalg.norm(abs(clue_hsv - plane), axis=2)

    d = np.mean(dist_abs ** 2) ** 0.5
    if d < 12:
        return None

    _, _, v = cv2.split(clue_hsv)
    v_lower, v_upper = np.percentile(v, 2), np.percentile(v, 98)
    v = (v - v_lower) * 255 / (v_upper - v_lower)

    black = cv2.inRange(v, 0, 40)
    contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = merge_bounding_rects(cv2.boundingRect(cnt) for cnt in contours)
    y, h = (y + h / 2 - ch * 0.35), ch * 0.7
    x, y, w, h = map(int, (x, y, w, h))

    v = np.uint8(np.clip(v, 0, 255))
    col = background_color(clue_hsv)
    background_mask = cv2.bitwise_not(cv2.inRange(clue_hsv, col - 10, col + 10))
    v = cv2.bitwise_and(v, v, mask=background_mask)
    v = v[y : y + h, x : x + w]

    v_threshold = 0.8
    v = np.clip(1 + (v.astype(float) / 255 - v_threshold) * 3, 0, 1)
    v = (v * 255).astype(np.uint8)

    v_blur = cv2.blur(v, (20, 20))
    v_blur = cv2.inRange(v, 100, 255)
    contours, _ = cv2.findContours(v_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = merge_bounding_rects(cv2.boundingRect(cnt) for cnt in contours)
    v = v[:, x : x + w]

    h, w = v.shape
    ratio, new_h = w / h, 28
    v = cv2.resize(v, (int(new_h * ratio), new_h))
    h, w = v.shape
    ratio = w / h

    return v


def match_number(clue_img, digit_imgs):
    v = format_number_image(clue_img)
    if v is None:
        return None
    v_paddings = crop_or_pad_center(v, (32 + 4, 28 + 4))

    img_out = v_paddings.copy()

    scores_x = []
    scores_xx = []
    for d_img, digit in digit_imgs:
        # print(v_paddings.shape, d_img.shape)
        res = cv2.matchTemplate(v_paddings, d_img, cv2.TM_CCOEFF_NORMED)
        (_, score, _, _) = cv2.minMaxLoc(res)
        if digit <= 9:
            scores_x.append((score, digit))
        else:
            scores_xx.append((score, digit))
    s1, d1 = max(scores_x)
    s2, d2 = max(scores_xx)
    digit = max([(s1, d1), (s2, d2)])[1]

    # if score < 0.98:
    cv2.imwrite(f"images/output/r{digit:02d}_p{score*100:02.2f}.png", img_out)
    return digit


def read_puzzle(img):
    # extract grid rect
    border_mask = cv2.inRange(img, np.array([0, 245, 245]), np.array([255, 255, 255]))
    contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
    gx, gy, gw, gh = max(bounding_rects, key=lambda t: t[2] * t[3])
    gx, gy, gw, gh = gx + 1, gy + 1, gw - 2, gh - 2
    grid_rect = gx, gy, gw, gh
    print(f"bounding rectangle size: ({gw}, {gh})")

    # get grid dimension (r, c in {5, 10, 15, ...})
    grid_img = img[gy : gy + gh, gx : gx + gw].copy()
    grid_background_color = background_color(grid_img)
    grid_img_mask = cv2.inRange(
        grid_img, grid_background_color - 20, grid_background_color + 20
    )
    contours, _ = cv2.findContours(grid_img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bounding_rects = [cv2.boundingRect(cnt) for cnt in contours]
    cws, chs = [t[2] for t in bounding_rects], [t[3] for t in bounding_rects]
    cw, ch = median_grouped(cws) + 1, median_grouped(chs) + 1
    c, r = int(gw / cw), int(gh / ch)
    # print(r, c)
    r, c = int(r / 5 + 0.5) * 5, int(c / 5 + 0.5) * 5
    cw, ch = gw / c, gh / r
    print(f"grid dimension: {r} x {c}")

    # load template images
    digit_imgs = []
    filenames = []
    for (_, _, fn) in walk("images/digits"):
        filenames.extend(fn)
        break
    for fn in filenames:
        if fn.endswith(".zip"):
            zfile = ZipFile(f"images/digits/{fn}", "r")
            for name in zfile.namelist():
                digit = int(re.match(r"\d+", name).group())
                d_img = cv2.imdecode(np.frombuffer(zfile.read(name), np.uint8), 1)
                digit_imgs.append((d_img, digit))
            zfile.close()
        elif fn.endswith(".png"):
            digit = int(re.match(r"\d+", fn).group())
            d_img = cv2.imread(f"images/digits/{fn}")
            digit_imgs.append((d_img, digit))
    for i, (d_img, digit) in enumerate(digit_imgs):
        d_img = cv2.split(cv2.cvtColor(d_img, cv2.COLOR_RGB2HSV))[2]
        d_img = crop_or_pad_center(d_img, (32, 28))
        digit_imgs[i] = d_img, digit

    digit_imgs = sorted(digit_imgs, key=lambda t: t[1])

    # read number clues
    shutil.rmtree("images/output", ignore_errors=True)
    Path("images/output").mkdir(parents=True, exist_ok=True)
    img_debug = img.copy()
    clues_h, clues_v = [[] for _ in range(r)], [[] for _ in range(c)]
    margin = 1
    colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 125, 0], [0, 125, 255]]
    )
    for j in range(r):
        for i in range(20):
            ys = int(gy + ch * j + margin)
            ye = int(gy + ch * j) + int(ch) - margin
            xs = int(gx - cw * i) - int(cw) + margin - 1
            xe = int(gx - cw * i) - margin - 1
            if xs < 0:
                break
            clue_img = img[ys:ye, xs:xe].copy()
            digit = match_number(clue_img, digit_imgs)
            if digit is None:
                break
            col = np.array(colors[digit % 5]) * (0.3 if digit >= 10 else 1.0)
            cv2.rectangle(img_debug, (xs, ys), (xe, ye), col, 2)
            clues_h[j].append(digit)
    for i in range(c):
        for j in range(20):
            xs = int(gx + cw * i + margin)
            xe = int(gx + cw * i) + int(cw) - margin
            ys = int(gy - ch * j) - int(ch) + margin - 1
            ye = int(gy - ch * j) - margin - 1
            if ys < 0:
                break
            clue_img = img[ys:ye, xs:xe].copy()
            digit = match_number(clue_img, digit_imgs)
            if digit is None:
                break
            col = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 125, 0], [0, 125, 255]][
                digit % 5
            ]
            col = np.array(col) * (0.4 if digit >= 10 else 1.0)
            cv2.rectangle(img_debug, (xs, ys), (xe, ye), col, 2)
            clues_v[i].append(digit)
    clues_h = tuple(tuple(clue[::-1]) for clue in clues_h)
    clues_v = tuple(tuple(clue[::-1]) for clue in clues_v)
    clues = (clues_v, clues_h)

    for s, cs in zip(("clues-h: ", "clues-v: "), (clues_h, clues_v)):
        print(s + " / ".join(" ".join(map(str, t)) for t in cs))

    cv2.imwrite(f"images/output/clue_debug{r,c,sum(map(len, clues_h))}.png", img_debug)
    return grid_rect, clues


def test():
    def get_image(n):
        filename = f"images/test/{n}.png"
        img = cv2.imread(filename)
        return img

    for n in [1, "1x", 2, 3, 4, 5, 6, 7]:
        _, clues = read_puzzle(get_image(n))
        grid = nonogram.Nonogram(clues, callback=None).solve()
        nonogram.print_grid(grid)


def test_interactive():
    def get_image():
        try:
            hwnd = win32gui.FindWindow(None, "Pictopix")
            if not hwnd:
                raise Exception("pictopix not running")
            x, y, x1, y1 = win32gui.GetClientRect(hwnd)
            x, y = win32gui.ClientToScreen(hwnd, (x, y))
            x1, y1 = win32gui.ClientToScreen(hwnd, (x1, y1))
            bbox = (x, y, x1, y1)

            mouse_controller = mouse.Controller()
            pos_backup = mouse_controller.position
            mouse_controller.position = (x + 10, y + 10)
            sleep(0.2)
            img = ImageGrab.grab(bbox)
            mouse_controller.position = pos_backup

            return img
        except Exception as e:
            print(e)
            return None

    def capture_solve_task():
        try:
            img = get_image()
            time_start = time.perf_counter()
            img = np.array(img)
            grid_rect, clues = read_puzzle(img)
            time_end = time.perf_counter()
            print(f"puzzle read ({time_end - time_start:.3f}s)")
            time_start = time.perf_counter()
            grid = nonogram.Nonogram(clues, callback=None).solve()
            if grid is None:
                return None, None
            time_end = time.perf_counter()
            print(f"nonogram solved ({time_end - time_start:.3f}s)")
            nonogram.print_grid(grid)
            return grid_rect, grid
        except Exception as e:
            print(e)
            return None, None

    def fill_task(grid_rect, grid):
        try:
            hwnd = win32gui.FindWindow(None, "Pictopix")
            if not hwnd:
                raise Exception("pictopix not running")
            mouse_controller = mouse.Controller()
            keyboard_controller = keyboard.Controller()

            pos_backup = mouse_controller.position
            gx, gy, gw, gh = grid_rect
            r, c = len(grid), len(grid[0])
            grid_pos = {
                (i, j): (
                    int(gx + gw * (2 * i + 1) / (2 * c)),
                    int(gy + gh * (2 * j + 1) / (2 * r)),
                )
                for j in range(r)
                for i in range(c)
            }

            indices = grid_pos.keys()
            indices_0 = [(i, j) for (i, j) in indices if grid[j][i] == 0]
            indices_1 = [(i, j) for (i, j) in indices if grid[j][i] == 1]
            indices = indices_1 + indices_0
            # indices = indices_1
            for (i, j) in indices:
                pos = win32gui.ClientToScreen(hwnd, grid_pos[(i, j)])
                mouse_controller.position = pos

                if grid[j][i] == 0:
                    key = keyboard.KeyCode(char="x")
                else:
                    key = keyboard.Key.space
                keyboard_controller.press(key)
                keyboard_controller.release(key)
                yield (i, j)
                sleep(0.05)
            mouse_controller.position = pos_backup
        except Exception as e:
            print(e)

    def capture_solve_fill_task():
        grid_rect, grid = capture_solve_task()
        yield 0
        if grid_rect is None:
            return
        for x in fill_task(grid_rect, grid):
            yield x

    cmd_queue = deque()
    input_blocked = False

    def command(cmd_text, *, block_other_input=False, ignore_block=False):
        def f():
            nonlocal cmd_queue, input_blocked
            if not block_other_input:
                if not input_blocked or ignore_block:
                    cmd_queue.appendleft((cmd_text, block_other_input))
                return
            else:
                input_blocked = True
                cmd_queue.appendleft((cmd_text, block_other_input))

        return f

    task_it = None
    key_listener = keyboard.GlobalHotKeys(
        {
            "<ctrl>+z": command("quit", ignore_block=True),
            "q": command("interrupt", ignore_block=True),
            "<ctrl>+r": command("capture_solve_fill", block_other_input=True),
        }
    )
    key_listener.start()
    while True:
        if cmd_queue:
            cmd, block_other_input = cmd_queue.pop()
            print(f"command: {cmd}")
            if cmd == "quit":
                break
            elif cmd == "interrupt":
                if task_it is not None:
                    task_it = None
            elif cmd == "capture_solve_fill":
                task_it = capture_solve_fill_task()
        if task_it:
            try:
                task_it.__next__()
            except StopIteration:
                task_it = None
                if block_other_input:
                    input_blocked = False
                print("finished\n")
    key_listener.stop()


if __name__ == "__main__":
    test_interactive()

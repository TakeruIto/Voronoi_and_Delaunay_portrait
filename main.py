import sys
import argparse
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser(
        description="voronoi and delaunay portrait")
    parser.add_argument("--type", required=True,
                        help="v:voronoi or d:delaunay")
    parser.add_argument("--path", required=True, help="path to image")
    parser.add_argument("--color", required=False,
                        help="when choose voronoi, result is colored if '--color c'")
    args = parser.parse_args()
    path = args.path
    if args.type == "v":
        img = make_voronoi(path, args.color)
    elif args.type == "d":
        img = make_delaunay(path)

    cv2.imwrite("result.png", img)
    cv2.imshow("result", img)
    cv2.waitKey(0)


def make_subdiv(img):
    n_point = 10000

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    result = np.zeros((img.shape[0], img.shape[1]))
    rand_xy = []
    cnt = 0
    while True:
        x = np.random.randint(1, img.shape[1])
        y = np.random.randint(1, img.shape[0])
        if th[y][x] == 0:
            rand_xy.append([x, y])
            result[y][x] = 1
            cnt += 1
        if cnt >= n_point:
            break

    rect = (0, 0, img.shape[1], img.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in rand_xy:
        subdiv.insert((p[0], p[1]))

    return subdiv


def make_delaunay(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    subdiv = make_subdiv(img_gray)

    triangles = subdiv.getTriangleList()
    pols = triangles.reshape(-1, 3, 2)
    img_draw = np.zeros((img.shape[0], img.shape[1]))
    cv2.polylines(img_draw, pols.astype(int), True, 1, thickness=1)
    img_draw = post_pro(img_draw)
    return img_draw


def make_voronoi(path, color):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    subdiv = make_subdiv(img_gray)
    facets, _ = subdiv.getVoronoiFacetList([])
    img_draw = np.zeros((img.shape[0], img.shape[1], 3))
    if color is None:
        cv2.polylines(img_draw, [f.astype(int)
                                 for f in facets], True, (1, 1, 1), thickness=1)
    elif color == "c":
        for p in (f.astype(int) for f in facets):
            c = img[p[0][1], p[0][0]
                    ] if (0 <= p[0][0] < img_draw.shape[1] and 0 <= p[0][1] < img_draw.shape[0]) else [0, 0, 0]
            c = np.array(c)
            c = c/255
            cv2.fillPoly(img_draw, [p], c)
    img_draw = post_pro(img_draw)
    return img_draw


def post_pro(img):
    img = (img*255).astype(np.uint8)
    # img = 255-img
    return img


if __name__ == "__main__":
    main()

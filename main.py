import cv2
from matplotlib import pyplot as plt
import os
import argparse
import random
import numpy as np
from pyeasyga import pyeasyga

seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)


def creat_folder(path):
    """Create required folder."""
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f'File exist: {path}')

def fitness(individual, data):
    """."""
    matches_selected = []
    for selected, match_selected in zip(individual, data):
        if selected == 1:
            matches_selected.append(match_selected)

    src_pts_right_selected = np.float32([kps_right[m.queryIdx].pt for m in matches_selected]).reshape(-1, 1, 2)
    dst_pts_left_selected = np.float32([kps_left[m.trainIdx].pt for m in matches_selected]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(
        src_pts_right_selected,
        dst_pts_left_selected,
        cv2.RANSAC,
        ransacReprojThreshold=r_th_g,
        maxIters=r_iter_g,
        confidence=conf_g
    )
    m_destortion = M[:2, :2]
    det_ = np.linalg.det(m_destortion)
    rez = (1 - det_) ** 2
    return rez


def create_individual(data):
    """."""
    return [random.randint(0, 1) for _ in
            range(len(data))]  # individual[random.randint(0, 1) for _ in range(len(data))]


def features_matching(
        img_right,
        img_left,
        lowe_rate,
        conf,
        match_take,
        flag_ga,
        r_iter,
        r_th,
        save
):
    """Function for Feature Matching + Perspective Transformation."""
    sift = cv2.SIFT_create()
    global kps_right
    global kps_left
    global r_iter_g
    global r_th_g
    global conf_g

    r_iter_g = r_iter
    r_th_g = r_th
    conf_g = conf

    kps_right, des_right = sift.detectAndCompute(img_right, None)
    kps_left, des_left = sift.detectAndCompute(img_left, None)

    matcher = cv2.BFMatcher()

    matches_all = matcher.knnMatch(des_right, des_left, k=match_take)

    g_math_lowes = []
    for match in matches_all:
        if match[0].distance < lowe_rate * match[1].distance:
            g_math_lowes.append(match[0])

    if len(g_math_lowes) > 4:
        if flag_ga:
            data = g_math_lowes.copy()
            ga = pyeasyga.GeneticAlgorithm(
                data,
                population_size=50,
                generations=20,
                crossover_probability=0.70,
                mutation_probability=0.3,
                elitism=False,
                maximise_fitness=False
            )

            ga.create_individual = create_individual
            ga.fitness_function = fitness
            ga.run()
            # print(ga.best_individual())
            # print(calculate_bic(len(data), ga.best_individual()[0], sum(ga.best_individual()[1])))
            selected_matches_final = [match for selected, match in zip(ga.best_individual()[1], data) if selected == 1]
            g_match = selected_matches_final.copy()
        else:
            g_match = g_math_lowes

        ##### solve ##############
        src_pts_right_final0 = np.float32([kps_right[m.queryIdx].pt for m in g_match]).reshape(-1, 1, 2)
        dst_pts_left_final0 = np.float32([kps_left[m.trainIdx].pt for m in g_match]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(
            src_pts_right_final0,
            dst_pts_left_final0,
            cv2.RANSAC,
            ransacReprojThreshold=r_th,
            maxIters=r_iter,
            confidence=conf
        )

        matchesMask = mask.ravel().tolist()

        m_destortion = M[:2, :2]
        det_0 = np.linalg.det(m_destortion)
        rez0 = (1 - det_0) ** 2
        print("result", rez0)

        xsr = []
        ysr = []
        xsl = []
        ysl = []
        xsr_g = []
        ysr_g = []
        xsl_g = []
        ysl_g = []

        for right_c, left_c, flag in zip(src_pts_right_final0, dst_pts_left_final0, matchesMask):
            xr, yr = right_c[0]
            xl, yl = left_c[0]
            if flag == 1:
                xsr_g.append(xr)
                ysr_g.append(yr)
                xsl_g.append(xl)
                ysl_g.append(yl)
            xsr.append(xr)
            ysr.append(yr)
            xsl.append(xl)
            ysl.append(yl)
        plt.scatter(xsr, xsl, c='b', s=2, alpha=0.3)
        plt.scatter(ysr, ysl, c='r', s=5, alpha=0.3)
        plt.scatter(xsr_g, xsl_g, c='r', s=2, alpha=0.3)
        plt.scatter(ysr_g, ysl_g, c='g', s=2, alpha=0.3)
        plt.grid(True)
        plt.title("0")
        plt.savefig(f"{save}/img1.jpg")
        plt.close('all')

        plt.scatter(ysr_g, ysr_g, c='g', s=2, alpha=0.3)
        plt.scatter(ysr_g, ysl_g, c='r', s=2, alpha=0.3)

        plt.scatter(xsr_g, xsr_g, c='b', s=2, alpha=0.3)
        plt.scatter(xsr_g, xsl_g, c='c', s=2, alpha=0.3)

        plt.grid(True)
        plt.savefig(f"{save}/img_rl.jpg")
        plt.close('all')

    else:
        print("Not enough matches have been found! - %d/%d" % (len(g_math_lowes)))
        matchesMask = None

    # if flag_matchesMask:
    draw_params = dict(
        matchesMask=matchesMask,
        matchColor=(0, 255, 0),
        # singlePointColor=(255, 255, 0),  # only inliers
        flags=0
    )
    # else:
    draw_no_params = dict(
        matchesMask=None,
        matchColor=(0, 255, 0),
        # singlePointColor=(255, 255, 0),  # only inliers
        flags=0
    )

    img_with_matches = cv2.drawMatches(img_right, kps_right, img_left, kps_left, g_match, None, **draw_params)
    img_no_with_matches = cv2.drawMatches(img_right, kps_right, img_left, kps_left, g_match, None, **draw_no_params)
    plt.close('all')

    return (img_with_matches, img_no_with_matches, img_right, img_left, M)


def do_stitching(img_1, img_2, H):
    """Perform stitching."""
    dst = cv2.warpPerspective(img_1, H, ((img_1.shape[1] + img_2.shape[1]), img_2.shape[0]))
    wraped_image = dst.copy()
    dst[0:img_2.shape[0], 0:img_2.shape[1]] = img_2
    return wraped_image, dst


def main(
        path_img_left,
        path_img_right,
        save,
        lowe_rate,
        conf,
        r_iter,
        r_th,
        match_take,
        flag_ga
):
    """Run main code."""
    creat_folder(save)

    img_right_init = cv2.imread(path_img_right, 1)
    img_left_init = cv2.imread(path_img_left, 1)

    img_matches_mask, img_no_matches_mask, img_r, img_l, H = features_matching(
        img_right_init.copy(),
        img_left_init.copy(),
        lowe_rate,
        conf,
        match_take,
        flag_ga,
        r_iter,
        r_th,
        save
    )

    warped_mask, stitched_mask = do_stitching(img_right_init, img_left_init, H)

    cv2.imwrite(f'{save}/stitched_img.jpg', stitched_mask)
    cv2.imwrite(f'{save}/warped_right_img_lowe_{lowe_rate}.jpg', warped_mask)

    cv2.imwrite(f'{save}/all_matches_lowe_{lowe_rate}.jpg', img_no_matches_mask)
    cv2.imwrite(f'{save}/selected_matches_lowe_{lowe_rate}.jpg', img_matches_mask)

    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_img_left", type=str, default='')
    parser.add_argument("--path_img_right", type=str, default='')
    parser.add_argument("--save_rez", type=str, default='')
    parser.add_argument("--lowe_rate", type=float, default=0.8)
    parser.add_argument("--conf", type=float, default=0.9)
    parser.add_argument("--r_iter", type=int, default=1000)
    parser.add_argument("--r_th", type=float, default=0.5)
    parser.add_argument("--match_take", type=int, default=2)
    parser.add_argument("--ga", action="store_true", help="activate GA")

    args = parser.parse_args()

    main(
        args.path_img_left,
        args.path_img_right,
        args.save_rez,
        args.lowe_rate,
        args.conf,
        args.r_iter,
        args.r_th,
        args.match_take,
        args.ga
    )

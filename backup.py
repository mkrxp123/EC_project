# %%
import os, time, random, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# %%
def SIFT_img(img):
    SIFT = cv2.SIFT_create()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = SIFT.detectAndCompute(img_gray, None)
    kpts = np.stack(np.vectorize(lambda kpt: kpt.pt)(data[0])).T
    feature = data[1]
    return kpts, feature


def find_matches(f1, f2, threshold=0.70):
    # generate l2-norm distance matrix similarity, similarity[i, j] = l2-norm(a1[i] - a2[j])
    similarity = np.vectorize(lambda v1, a2: np.linalg.norm(v1 - a2, axis=1), excluded=["a2"],
                              signature="(n),(m,n)->(m)")(f1, f2)
    # get two highest index and value of similarity
    two_lowest_index = np.argpartition(similarity, 2)[:, :2]
    two_lowest_value = np.vectorize(lambda s, i: s[i], signature="(n),(m)->(m)")(similarity, two_lowest_index)
    # ratio test
    good_match = two_lowest_value[:, 0] < threshold * two_lowest_value[:, 1]
    # generate matching matrix based on good_match
    kpts1 = np.arange(f1.shape[0])[good_match]
    kpts2 = two_lowest_index[good_match, 0]
    return kpts1, kpts2


def homography(m1, m2):
    A = np.zeros((8, 8))
    A[:4, :2] = m1
    A[:4, 2] = 1
    A[4:, 3:5] = m1
    A[4:, 5] = 1
    A[:4, 6] = -m1[:, 0] * m2[:, 0]
    A[:4, 7] = -m1[:, 1] * m2[:, 0]
    A[4:, 6] = -m1[:, 0] * m2[:, 1]
    A[4:, 7] = -m1[:, 1] * m2[:, 1]
    b = np.hstack([m2[:, 0], m2[:, 1]])
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)


def calculate_outlier(H, m1, m2, tolerance=4):
    m2_hat = H @ m1
    m2_hat = m2_hat / np.where(m2_hat[2] != 0, m2_hat[2], 1)
    return (np.linalg.norm(m2[:2] - m2_hat[:2], axis=0) > tolerance).sum()


def roulette_selection(population, fits, pairs):
    fits = fits / np.sum(fits)
    parents = rng.choice(population, size=(pairs, 2), replace=True, p=fits)
    return parents


# a set must not contain duplicate numbers
def reproduce(parent: np.ndarray, match_counts: int, lamb: float):
    duplicate = True
    # concatenate two chromosome
    shuffle = parent.flatten()
    while duplicate:
        # shuffle
        # cut the whole array in half
        # check whether each half contains duplicate numbers
        # mutation = np.concatenate([shuffle, np.random.randint(0, match_counts, np.random)])
        np.random.shuffle(shuffle)
        c1, c2 = sorted(shuffle[0:4]), sorted(shuffle[4:8])
        duplicate = False
        for i in range(1, 4):
            if c1[i] == c1[i-1] or c2[i] == c2[i-1]:
                duplicate = True
                break
    return c1, c2


def Evo_RANSAC(base, addition, tolerance=4, outlier_rate=0.1, max_iter=1000, population=100):
    if population % 2:  raise RuntimeError('population must be divisible by 2')
    
    match_counts = base.shape[0]
    expand = lambda points: np.vstack([points.T, np.ones((1, match_counts))])
    rand_comb = lambda n, r=4: sorted(random.sample(range(n), r))
    points_base, points_addition = expand(base), expand(addition)
    threshold = match_counts * outlier_rate
    iter, max_inlier, argmax_H = 0, 0, None
    population = np.asarray([rand_comb(match_counts) for _ in range(population)])
    rec = []
    while (match_counts - max_inlier) > threshold and iter <= max_iter:
        # print(iter)
        Hs = [homography(points_addition.T[points, :2], points_base.T[points, :2]) for points in population]
        fits = [match_counts - calculate_outlier(H, points_addition, points_base) for H in Hs]
        argmax_fit = np.argmax(fits)
        rec.append(match_counts - fits[argmax_fit])

        if fits[argmax_fit] > max_inlier:
            max_inlier = fits[argmax_fit]
            argmax_H = Hs[argmax_fit]

        parents = roulette_selection(population, fits, population.shape[0] // 2)
        
        # TODO: mutation
        children = np.empty_like(population, dtype=population.dtype)
        for i in range(parents.shape[0]):
            children[2*i], children[2*i+1] = reproduce(parents[i])
            # children[4*i+2] = rand_comb(match_counts)
            # children[4*i+3] = rand_comb(match_counts)
        population = children
        
        # children = []
        # for p1, p2 in parents:
        #     random.shuffle(p1)
        #     random.shuffle(p2)
        #     children.append(sorted(np.concatenate([p1[:2], p2[2:]])))
        #     children.append(sorted(np.concatenate([p2[:2], p1[2:]])))
        #     children.append(rand_comb(match_counts))
        #     children.append(rand_comb(match_counts))
        # population = np.asarray(children)

        iter += 1
    return argmax_H, rec


def RANSAC(base, addition, tolerance=4, outlier_rate=0.1, max_iter=50000):
    match_counts = base.shape[0]
    expand = lambda points: np.vstack([points.T, np.ones((1, match_counts))])
    rand_comb = lambda n, r=4: sorted(random.sample(range(n), r))
    points_base, points_addition = expand(base), expand(addition)
    outlier_cnt, threshold = match_counts, match_counts * outlier_rate
    iter, min_outlier, argmin_H = 0, match_counts, None
    while outlier_cnt > threshold and iter <= max_iter:
        # print(iter)
        picked = rand_comb(match_counts)
        picked_points_base, picked_points_addition = points_base.T[picked, :2], points_addition.T[picked, :2]
        temp_H = homography(picked_points_addition, picked_points_base)
        test_points = temp_H @ points_addition
        test_points = test_points / np.where(test_points[2] != 0, test_points[2], 1)
        outlier_cnt = (np.linalg.norm(points_base[:2] - test_points[:2], axis=0) > tolerance).sum()
        iter += 1
        if outlier_cnt < min_outlier:
            min_outlier = outlier_cnt
            argmin_H = temp_H
    return argmin_H


def img_mask(img_1, img_2):
    gray_1, gray_2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    mask_1, mask_2 = gray_1 != 0, gray_2 != 0
    mask_all = np.logical_or(mask_1, mask_2)
    return gray_1, gray_2, mask_1, mask_2, mask_all


def gap_blend(img_1, img_2, img_3, gray_1, gray_2, mask_all, threshold=125):
    img_sum = gray_1.astype(float) + gray_2.astype(float)
    weight_1 = np.divide(gray_1, img_sum, out=np.ones_like(gray_1, dtype=float), where=img_sum != 0)
    weight_2 = 1 - weight_1

    patch = np.zeros_like(img_1)
    for i in range(3):
        patch[:, :, i] = weight_1 * img_1[:, :, i] + weight_2 * img_2[:, :, i]

    fix = np.logical_and(cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY) < threshold, mask_all)
    img_3[fix] = patch[fix]
    return img_3


def wrap_imgs(H, addition, base):
    h1, w1, _ = addition.shape
    h2, w2, _ = base.shape
    corners = cv2.perspectiveTransform(np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]]).reshape(-1, 1, 2), H).squeeze()

    min_x, max_x = 0, w1
    min_y, max_y = 0, h1
    for img_3 in corners:
        min_x = img_3[0] if img_3[0] < min_x else min_x
        max_x = img_3[0] if img_3[0] > max_x else max_x
        min_y = img_3[1] if img_3[1] < min_y else min_y
        max_y = img_3[1] if img_3[1] > max_y else max_y

    delta_x = -(min_x if min_x < 0 else 0)
    delta_y = -(min_y if min_y < 0 else 0)

    A = np.eye(3)
    A[:2, 2] = [delta_x, delta_y]

    new_w = int(max_x - min_x) if max_x - min_x > (w2 + delta_x) else int(w2 + delta_x)
    new_h = int(max_y - min_y) if max_y - min_y > (h2 + delta_y) else int(h2 + delta_y)

    img_1 = cv2.warpPerspective(base, A, (new_w, new_h))
    img_2 = cv2.warpPerspective(addition, A @ H, (new_w, new_h))
    
    # use this to debug
    return cv2.addWeighted(img_1, 0.5, img_2, 0.5, 0)

    # use this to do image blending
    gray_1, gray_2, mask_1, mask_2, mask_all = img_mask(img_1, img_2)
    img_3 = img_1.copy()
    img_3[mask_2] = img_2[mask_2]
    img_3 = gap_blend(img_1, img_2, img_3, gray_1, gray_2, mask_all)
    return img_3


def run_RANSAC(RGB):
    base_img = RGB[0]
    for img in RGB[1:]:
        kpts_base, feature_base = SIFT_img(base_img)
        kpts_addition, feature_addition = SIFT_img(img)

        index_base, index_addition = find_matches(feature_base, feature_addition)
        H = RANSAC(kpts_base[index_base], kpts_addition[index_addition])
        # H, _ = cv2.findHomography(kpts_addition[index_addition], kpts_base[index_base], cv2.RANSAC)
        base_img = wrap_imgs(H, img, base_img)

    # cv2.imwrite("img.jpg", base_img)
    return cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)


def run_EVO_RANSAC(RGB):
    base_img = RGB[0]
    rec = []
    for img in RGB[1:]:
        kpts_base, feature_base = SIFT_img(base_img)
        kpts_addition, feature_addition = SIFT_img(img)

        index_base, index_addition = find_matches(feature_base, feature_addition)
        H, tmp = Evo_RANSAC(kpts_base[index_base], kpts_addition[index_addition])
        rec.append(tmp)
        base_img = wrap_imgs(H, img, base_img)

    # cv2.imwrite("evo_img.jpg", base_img)
    return cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB), rec


def plot(info):
    gs = gridspec.GridSpec(4, 6)

    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0:3, 0:3])
    ax1.imshow(info["RANSAC_IMG"])
    ax1.set_title(f"RANSAC: {info['RANSAC_TIME']:.3f} sec")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0:3, 3:6])
    ax2.imshow(info["EVO_IMG"])
    ax2.set_title(f"EVO: {info['EVO_TIME']:.3f} sec")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[-1, :])
    for r in info["EVO_REC"]:
        ax3.plot(r)

    plt.savefig("result.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # modify the folder path to change stitching images
    folder = "./baseline"
    RGB = np.stack([cv2.imread(f"{folder}/{img}") for img in os.listdir(folder)])

    # Since the complexity of match finding is O(N1N2), where Ni is number of key-points.
    # If the image resolution is higher, number of key-points grows too
    # In this case, just lower the resolution would be fine
    # folder = "./test"
    # RGB = np.stack([cv2.resize(cv2.imread(f"{folder}/{img}"), (720, 1080)) for img in os.listdir(folder)])
    
    info = {}
    t = time.time()
    info["RANSAC_IMG"] = run_RANSAC(RGB)
    info["RANSAC_TIME"] = time.time() - t
    # print(f"RANSAC time: {time.time() - t}")
    t = time.time()
    info["EVO_IMG"], info["EVO_REC"] = run_EVO_RANSAC(RGB)
    info["EVO_TIME"] = time.time() - t
    # print(f"EVO_RANSAC time: {time.time() - t}")

    plot(info)

# %%
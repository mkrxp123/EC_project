import os, cv2, time
import numpy as np
from abc import ABC, abstractmethod


class StitchBase(ABC):
    def __init__(self, config):
        self.rng = np.random.default_rng(config["seed"])     
        self.ratio_test = config["ratio_test"]
        self.tolerance = config["tolerance"]
        self.outlier_rate = config["outlier_rate"]
        self.max_iter = config["max_iter"]

    @staticmethod
    def read_img(folder, size):
        if size is not None:    return np.stack([cv2.resize(cv2.imread(f"{folder}/{img}"), size) for img in os.listdir(folder)])
        else:                   return np.stack([cv2.imread(f"{folder}/{img}") for img in os.listdir(folder)])

    @staticmethod
    def expand(points):
        return np.vstack([points.T, np.ones_like(points.T[-1])])
    
    def rand_comb(self, n, r=4):
        return sorted(self.rng.choice(n, r, replace=False))
    
    @staticmethod
    def SIFT_img(img):
        SIFT = cv2.SIFT_create()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data = SIFT.detectAndCompute(img_gray, None)
        kpts = np.stack(np.vectorize(lambda kpt: kpt.pt)(data[0])).T
        feature = data[1]
        return kpts, feature
    
    def find_matches(self, f1, f2):
        # generate l2-norm distance matrix similarity, similarity[i, j] = l2-norm(a1[i] - a2[j])
        similarity = np.vectorize(lambda v1, a2: np.linalg.norm(v1 - a2, axis=1), excluded=["a2"],
                                signature="(n),(m,n)->(m)")(f1, f2)
        # get two highest index and value of similarity
        two_lowest_index = np.argpartition(similarity, 2)[:, :2]
        two_lowest_value = np.take_along_axis(similarity, two_lowest_index, axis=1)
        # ratio test
        good_match = two_lowest_value[:, 0] < self.ratio_test * two_lowest_value[:, 1]
        # generate matching matrix based on good_match
        kpts1 = np.arange(f1.shape[0])[good_match]
        kpts2 = two_lowest_index[good_match, 0]
        return kpts1, kpts2
    
    @staticmethod
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
    
    def calculate_outlier(self, H, m1, m2):
        m2_hat = H @ m1
        m2_hat = m2_hat / np.where(m2_hat[2] != 0, m2_hat[2], 1)
        return (np.linalg.norm(m2[:2] - m2_hat[:2], axis=0) > self.tolerance).sum()
    
    @staticmethod
    def img_mask(img_1, img_2):
        gray_1, gray_2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        mask_1, mask_2 = gray_1 != 0, gray_2 != 0
        mask_all = np.logical_or(mask_1, mask_2)
        return gray_1, gray_2, mask_1, mask_2, mask_all
    
    @staticmethod
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
    
    @staticmethod
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
    
    @abstractmethod
    def fit(self, base, addition):
        raise NotImplementedError
    
    def run(self, folder, size=None):
        RGB = self.read_img(folder, size)
        t = time.time()
        
        base_img = RGB[0]
        recs = []
        for img in RGB[1:]:
            kpts_base, feature_base = self.SIFT_img(base_img)
            kpts_addition, feature_addition = self.SIFT_img(img)

            index_base, index_addition = self.find_matches(feature_base, feature_addition)
            H, rec = self.fit(kpts_base[index_base], kpts_addition[index_addition])
            recs.append(rec)
            base_img = self.wrap_imgs(H, img, base_img)

        # cv2.imwrite("evo_img.jpg", base_img)
        return cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB), recs, time.time() - t
    

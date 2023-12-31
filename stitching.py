import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from base import StitchBase


class RANSAC(StitchBase):
    def __init__(self, config):
        super(RANSAC, self).__init__(config)
    
    def fit(self, base, addition):
        match_counts = base.shape[0]
        points_base, points_addition = self.expand(base), self.expand(addition)
        outlier_cnt, threshold = match_counts, match_counts * self.outlier_rate
        iter, min_outlier, argmin_H = 0, match_counts, None
        rec = []
        while outlier_cnt > threshold and iter <= self.max_iter:
            # print(iter)
            picked = self.rand_comb(match_counts)
            picked_points_base, picked_points_addition = points_base.T[picked, :2], points_addition.T[picked, :2]
            temp_H = self.homography(picked_points_addition, picked_points_base)
            test_points = temp_H @ points_addition
            test_points = test_points / np.where(test_points[2] != 0, test_points[2], 1)
            outlier_cnt = (np.linalg.norm(points_base[:2] - test_points[:2], axis=0) > self.tolerance).sum()
            rec.append(outlier_cnt / match_counts)
            iter += 1
            if outlier_cnt < min_outlier:
                min_outlier = outlier_cnt
                argmin_H = temp_H
        return argmin_H, rec


class EVOSAC(StitchBase):
    def __init__(self, config):        
        super(EVOSAC, self).__init__(config)
        self.lamb = config["mutation_factor"]
        self.mutation_rate = config["mutation_rate"]
        if config["strategy"] == 'ROULETTE':
            self.selection_strategy = self._roulette_selection
        elif config["strategy"] == 'TOURNAMENT':
            self.selection_strategy = self._tournament_selection

        if config["1/5-rule"]:
            self._one_fifth_enable = True
            self.threshold = config["1/5-rule-threshold"]
            self.period = config["1/5-rule-iter"]
            self.alpha = config["1/5-rule-alpha"]
        else:
            self._one_fifth_enable = False

        if config["early_stop"]:
            self._early_stop_enable = True
            self.early_stop_threshold = config["early_stop-threshold"]
        else:
            self._early_stop_enable = False
            self.early_stop_threshold = np.inf
    
    # a set must not contain duplicate numbers
    def _reproduce(self, parent: np.ndarray, match_counts: int):
        duplicate = True
        # concatenate two chromosome
        combine = parent.flatten()
        while duplicate:
            # adding several random integers (size is based on poisson distribution) to the chromosomes
            # shuffle
            # cut the whole array in half
            # check whether each half contains duplicate numbers
            if self.rng.random() < self.mutation_rate:
                random = self.rng.choice(range(match_counts), self.rng.poisson(self.lamb, 1)[0], replace=False)
                mutation = np.concatenate([combine, random])
            else:
                mutation = combine
            self.rng.shuffle(mutation)
            c1, c2 = sorted(mutation[0:4]), sorted(mutation[4:8])
            duplicate = False
            for i in range(1, 4):
                if c1[i] == c1[i-1] or c2[i] == c2[i-1]:
                    duplicate = True
                    break
        return c1, c2
    
    def _roulette_selection(self, population, fits, pairs):
        fits = fits / np.sum(fits)
        parents = self.rng.choice(population, size=(pairs, 2), replace=True, p=fits)
        return parents

    def _tournament_selection(self, population, fits, pairs):
        sz = population.shape[0]
        def _find_one():
            indices = self.rng.choice(sz, size=2, replace=True)
            return population[indices[0]] if fits[indices[0]] > fits[indices[1]] else population[indices[1]]
        return np.array([(_find_one(), _find_one()) for _ in range(sz >> 1)], dtype=int)

    def _one_fifth(self, counter):
        if counter > self.period * self.threshold:
            self.lamb = self.lamb / self.alpha
        else:
            self.lamb = self.lamb * self.alpha
    
    def fit(self, base, addition):
        match_counts = base.shape[0]
        points_base, points_addition = self.expand(base), self.expand(addition)
        threshold = match_counts * self.outlier_rate
        eps = 0.01
        best_counter = 0
        iter, max_inlier, argmax_H = 0, 0, None
        if self._one_fifth_enable:
            counter = 0
            prev_avg_inlier = 0

        # initialize the population with reshaped random permutation of range(match_counts)
        step = match_counts // 4
        population = self.rng.choice(match_counts, size=(step, 4), replace=False)
        fits = np.empty(step, dtype=np.uint32)
        rec = []
        while (match_counts - max_inlier) > threshold and iter <= self.max_iter and best_counter < self.early_stop_threshold:
            for i in range(step):
                H = self.homography(points_addition.T[population[i], :2], points_base.T[population[i], :2])
                fits[i] = match_counts - self.calculate_outlier(H, points_addition, points_base)
                if fits[i] > max_inlier:
                    argmax_i, argmax_H = i, H
                    max_inlier = fits[i]
            rec.append((match_counts - fits[argmax_i]) / match_counts)

            parents = self.selection_strategy(population, fits, population.shape[0] // 2)
            
            # TODO: mutation
            # children = np.empty_like(population, dtype=population.dtype)
            for i in range(parents.shape[0]):
                population[2*i], population[2*i+1] = self._reproduce(parents[i], match_counts)
            # population = children
            iter += step

            if rec[-1] - max_inlier / match_counts <= eps:
                best_counter += 1
            else:
                best_counter = 0

            if self._one_fifth_enable:
                if np.average(fits) > prev_avg_inlier:
                    counter += 1
                prev_avg_inlier = np.average(fits)
                if iter % self.period == 0:
                    self._one_fifth(counter)
                    counter = 0
        return argmax_H, rec


def plot(info):
    gs = gridspec.GridSpec(4, 6)

    fig = plt.figure()
    ax1 = fig.add_subplot(gs[0:3, 0:3])
    ax1.imshow(info["RANSAC_IMG"])
    ax1.set_title(f"RANSAC: {info['RANSAC_TIME']:.3f} sec")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0:3, 3:6])
    ax2.imshow(info["EVOSAC_IMG"])
    ax2.set_title(f"EVOSAC: {info['EVOSAC_TIME']:.3f} sec")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[-1, 0:3])
    for r in info["RANSAC_REC"]:
        ax3.plot(r)
    ax3.set_xlim(0, max([len(rec) for rec in info["RANSAC_REC"]]))
    ax3.set_title(f"RANSAC outlier record")
        
    ax4 = fig.add_subplot(gs[-1, 3:6])
    for r in info["EVOSAC_REC"]:
        ax4.plot(r)
    ax4.set_xlim(0, max([len(rec) for rec in info["EVOSAC_REC"]]))
    ax4.set_title(f"EVOSAC outlier record")
        
    plt.savefig("result.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    info = {}
    # Since the complexity of match finding is O(N1N2), where Ni is number of key-points.
    # If the image resolution is higher, number of key-points grows too
    # In this case, just lower the resolution would be fine
    folder, size = "./baseline", (720, 480)   # resolution (h, w), use None to perserve the original size
    
    # True to use cv2.addWeight, False to use my own method
    blending_cv2 = False
    
    ransac_config = {
        "seed": 0,
        "ratio_test": 0.7,
        'tolerance': 4,
        "outlier_rate": 0.3,
        "max_iter": 10000,
        "blending_cv2": blending_cv2
    }
    print('Running RANSAC...', time.time())
    ransac = RANSAC(ransac_config)
    info["RANSAC_IMG"], info["RANSAC_REC"], info["RANSAC_TIME"] = ransac.run(folder, size)

    evosac_config = {
        "seed": 0,
        "ratio_test": 0.7,
        'tolerance': 4,
        "outlier_rate": 0.3,
        "max_iter": 10000,
        'mutation_factor': 0.5,
        'mutation_rate': 0.3,
        '1/5-rule': False,
        '1/5-rule-threshold': 0.1,
        '1/5-rule-iter': 50,
        '1/5-rule-alpha': 0.9,
        'early_stop': False,
        'early_stop-threshold': 10,
        'strategy': 'TOURNAMENT',
        "blending_cv2": blending_cv2
    }
    print('Running EVOSAC...', time.time())
    evosac = EVOSAC(evosac_config)
    info["EVOSAC_IMG"], info["EVOSAC_REC"], info["EVOSAC_TIME"] = evosac.run(folder, size)
    plot(info)
    print(info["EVOSAC_REC"])
    pickle.dump(info, open("info.pkl", "wb"))

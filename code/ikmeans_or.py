import time

from sklearn.cluster import KMeans
import numpy as np

from code.kmeans import CustomKMeans
class I_Kmeans_minus_plus:

    def __init__(self, data, k, init='k-means++'):
        self.k = k
        self.data = data
        self.tkmeans = t_k_means(self.data, self.k)
        self.S = Proposal(data, k)
        if not init in ['k-means++', 'useful', 'random']:
            raise Exception("Initialization is not supported, supported initializations are: k-means++, useful, random")
        self.init = init

    def fit(self):
        # Instruction 1
        if self.init == 'useful':
            self.S.useful_init()
            kmeans = CustomKMeans(n_clusters=self.k, init=self.S.centroids)
            kmeans.fit(self.data)
        start_time = time.time()
        cluster_assignment = kmeans.predict(self.data)
        distances = kmeans.transform(self.data)
        self.S.update_proposal(kmeans.centroids, cluster_assignment, distances)

        # instruction 2
        n_success = 0
        indivisible_clusters = []
        unmatchable_pairs = []  # list of pairs (s1,s2)
        irremovable_clusters = []
        while True:
            start1 = time.time()
            # instruction 3
            if len(indivisible_clusters) == self.k:
                break
            Si, gain_Si = self.S.get_max_gain(indivisible_clusters)
            # Instruction 4
            if self.S.k2_better_gain(indivisible_clusters, gain_Si):
                break
            # Instruction 5
            posible_Sjs = self.S.check_conditions(Si, gain_Si, unmatchable_pairs, irremovable_clusters)
            if len(posible_Sjs) == 0:
                break
            j, cost_Sj = self.S.get_min_cost(posible_Sjs)
            Sj = posible_Sjs[j]
            # Instruction 6
            if self.S.k2_better_cost(posible_Sjs, cost_Sj):
                indivisible_clusters.append(Si)
                continue
            # Instruction 7
            newCentroid = self.S.get_random_centroid(Si)
            end1 = time.time()
            start2 = time.time()
            # new solution
            S_ = Proposal(self.data, self.k, self.S.centroids, self.S.nearest_centers, self.S.second_nearest_centers)
            S_ = self.tkmeans.fit(S_, newCentroid, Si, Sj)
            S_.recompute_distances()
            # Instruction 8
            S_res = self.S.total_SSEDM()
            newS_res = S_.total_SSEDM()
            end2 = time.time()
            # print(Si, Sj)
            # print('prev: ' + str("{:e}".format(S_res)), 'new: ' + str("{:e}".format(newS_res)))
            # print('time1 :' + str(end1 - start1), 'time2 :' + str(end2 - start2))
            if newS_res >= S_res:
                unmatchable_pairs.append((Si, Sj))
            else:
                irremovable_clusters.append(Si)
                irremovable_clusters.append(Sj)
                prev_strong_adj = self.S.get_strong_adjacents(Sj)
                indivisible_clusters += prev_strong_adj
                self.S = S_
                curr_strong_adj = list(set(self.S.get_strong_adjacents(Si) + self.S.get_strong_adjacents(Sj)))
                indivisible_clusters += curr_strong_adj
                n_success += 1
                # print('succes!!!!!!!!')
            if n_success > self.k / 2:
                break
        end_time = time.time()
        return self.S.centroids, self.S.total_SSEDM(), self.S.max_SSEDM(), (end_time - start_time)

    def predict(self, X):
        centroid_idxs = []
        for x in X:
            dists = np.linalg.norm(x - self.S.centroids)
            centroid_idx = np.argmin(dists)
            centroid_idxs.append(centroid_idx)
        return centroid_idxs

import numpy as np

class Proposal:

    def __init__(self, data, k, centers=None, nc=None, snc=None):
        self.data = data
        self.k = k
        if centers is None:
            self.centroids = []
        else:
            self.centroids = centers.copy()
        if nc is None:
            self.nearest_centers = np.zeros(shape=self.data.shape[0], dtype=int)
        else:
            self.nearest_centers = nc.copy()
        if snc is None:
            self.second_nearest_centers = np.zeros(shape=self.data.shape[0], dtype=int)
        else:
            self.second_nearest_centers = snc.copy()

        self.distance_to_center = np.zeros(shape=(self.k, self.data.shape[0]))
        self.distance_inter_center = np.zeros(shape=(self.k, self.k))

    ##########################################################
    ##                 CUSTOM INITIALIZATION                ##
    ##########################################################

    def _useful_nearest_centers(self, P_id, c, last_UNC):
        if len(self.centroids) <= 1:
            return self.centroids
        else:
            useless = set()
            for cx in last_UNC:
                cx_id = self.centroids.index(cx)
                c_id = self.centroids.index(c)

                if cx != c and P_id != c and P_id != cx:

                    if self.distance_inter_center[c_id, cx_id] == 0:
                        Ccx_dist = np.linalg.norm(self.data[c, :] - self.data[cx, :])
                        self.distance_inter_center[cx_id, c_id] = Ccx_dist
                        self.distance_inter_center[c_id, cx_id] = Ccx_dist

                    if not (c in useless) and (
                            self.distance_to_center[cx_id, P_id] < self.distance_to_center[c_id, P_id]) and (
                            self.distance_inter_center[c_id, cx_id] < self.distance_to_center[c_id, P_id]):
                        # last center is useless
                        return last_UNC

            # last center is not useless
            for cx in last_UNC:
                cx_id = self.centroids.index(cx)
                c_id = self.centroids.index(c)

                if cx != c and P_id != c and P_id != cx:

                    if self.distance_inter_center[c_id, cx_id] == 0:
                        Ccx_dist = np.linalg.norm(self.data[c, :] - self.data[cx, :])
                        self.distance_inter_center[cx_id, c_id] = Ccx_dist
                        self.distance_inter_center[c_id, cx_id] = Ccx_dist

                    if not (cx in useless) and (
                            self.distance_to_center[c_id, P_id] < self.distance_to_center[cx_id, P_id]) and (
                            self.distance_inter_center[c_id, cx_id] < self.distance_to_center[cx_id, P_id]):
                        # cx is useless
                        useless.add(cx)

            useful = [center for center in last_UNC if not center in useless]
            if not c in useful:
                useful.append(c)

            return useful

    def _UNC_value(self, P_id, UNC):
        sum, lnsum, max = 0, 0, None
        for c in UNC:
            if (P_id != c):
                c_id = self.centroids.index(c)
                eucl = self.distance_to_center[c_id, P_id]  # DISTANCIA
                if max is None or max < eucl:
                    max = eucl
                sum += eucl
                lnsum += np.log(eucl)
        if (max is None): return 0
        avg = sum / len(UNC)
        return (avg / max) * lnsum

    def _compute_distances(self, center):
        return np.linalg.norm(self.data - center, axis=1)

    def useful_init(self):
        self.centroids.append(self.data[:, 0].argmin())
        UNCs = [[None] for _ in range(self.data.shape[0])]
        while len(self.centroids) < self.k:
            max, best_id, last_center = None, None, self.centroids[-1]
            if self.distance_to_center[len(self.centroids) - 1, 1] == 0 and self.distance_to_center[
                len(self.centroids) - 1, 2] == 0:
                self.distance_to_center[len(self.centroids) - 1, :] = self._compute_distances(self.data[last_center, :])
            for id, p in enumerate(self.data):
                if not id in self.centroids:
                    UNCs[id] = self._useful_nearest_centers(id, last_center, UNCs[id])
                    evaluation = self._UNC_value(id, UNCs[id])
                    if max == None or max < evaluation:
                        max = evaluation
                        best_id = id
            self.centroids.append(best_id)
        self.centroids = self.data[self.centroids, :]

    ##########################################################
    ##        UPDATE SOLUTION WITH KMEANS RESULTS           ##
    ##########################################################

    def _second_nearest_centers(self, transformed_data):
        sorted_distances = np.argsort(transformed_data, axis=1)
        self.second_nearest_centers = [sorted_distances[i, 1] for i in range(self.second_nearest_centers.shape[0])]

    def update_proposal(self, centers, nc, distances):
        self.centroids = centers
        self.nearest_centers = nc
        self.distance_to_center = distances.T

    ##########################################################
    ##                      K-Means-+                       ##
    ##########################################################

    def _SSEDM(self, cluster_ids, centroid_id):
        return np.sum(np.square(self.distance_to_center[centroid_id, cluster_ids]))

    def max_SSEDM(self):
        return np.max([self._SSEDM(np.where(self.nearest_centers == i)[0], i) for i in range(self.k)])

    def total_SSEDM(self):
        return np.sum([self._SSEDM(np.where(self.nearest_centers == i)[0], i) for i in range(self.k)])

    def _gain(self, cluster_ids, centroid_id):
        return self._SSEDM(cluster_ids, centroid_id) * (3 / 4)

    def get_max_gain(self, indivisibles):
        gains = np.array([self._gain(np.where(self.nearest_centers == i)[0], i) for i in range(self.k) if i not in indivisibles])
        id = gains.argmax()
        return id, gains[id]

    def k2_better_gain(self, indivisibles, gain):
        gains = [self._gain(np.where(self.nearest_centers == i)[0], i) for i in indivisibles if
                 self._gain(np.where(self.nearest_centers == i)[0], i) > gain]
        return len(gains) >= self.k / 2

    def _is_adjacent(self, cluster_id1, cluster_id2):
        p_cluster2 = np.where(self.nearest_centers == cluster_id2)[0]
        second_nearest_cluster = self.second_nearest_centers[p_cluster2]
        return np.any(second_nearest_cluster == cluster_id1)

    def _cost(self, cluster_id):
        cluster = np.where(self.nearest_centers == cluster_id)[0]
        score = self._SSEDM(cluster, cluster_id)
        second_nearest_centers = self.centroids[self.second_nearest_centers[cluster]]
        sum = 0
        for i in range(cluster.shape[0]):
            sum += np.square(np.linalg.norm(self.data[cluster[0], :] - second_nearest_centers[i]))

        return score - sum

    def check_conditions(self, Si, gain_Si, unmatchable_pairs, irremoval):
        possible_clusters = []
        for Sj in range(self.k):
            if Sj != Si and self._cost(Sj) < gain_Si and not ((Sj, Si) in unmatchable_pairs) and not (
                    (Si, Sj) in unmatchable_pairs) and not self._is_adjacent(Si, Sj) and not self._is_adjacent(Sj,Si) and not Sj in irremoval:
                possible_clusters.append(Sj)
        return possible_clusters

    def get_min_cost(self, candidates):
        costs = np.zeros(shape=len(candidates))
        for id, cluster_id in enumerate(candidates):
            costs[id] = self._cost(cluster_id)
        best_id = costs.argmin()
        return best_id, costs[best_id]

    def k2_better_cost(self, no_candidates, better):
        candidates = set(range(self.k)) - set(no_candidates)
        if len(candidates) < self.k / 2:
            return False
        sum = 0
        for cluster_id in candidates:
            if self._cost(cluster_id) < better:
                sum += 1
                if sum >= self.k / 2:
                    return True
        return False

    def get_random_centroid(self, Si):
        return self.data[np.random.choice(np.where(self.nearest_centers == Si)[0]), :]

    def get_strong_adjacents(self, Cj):
        sadj = []
        for c in range(self.k):
            if c != Cj and not c in sadj and self._is_adjacent(c, Cj) and self._is_adjacent(Cj, c):
                sadj.append(c)
        return sadj

    ##########################################################
    ##                tolerant-K-Means                      ##
    ##########################################################

    def get_adjacent_centers(self, centers):
        adj = set()
        for ci in centers:
            for id, cj in enumerate(self.centroids):
                if ci != id and self._is_adjacent(ci, id) and not id in adj:
                    adj.add(id)
        return adj

    def get_affected_points(self, centers_id):
        affected_p = np.array([], dtype=int)
        for c in centers_id:
            affected_p = np.union1d(affected_p, np.where(
                np.logical_or(self.nearest_centers == c, self.second_nearest_centers == c))[0])
        return affected_p

    def update_centroid(self, id, c):
        self.centroids[id] = c

    def update_first_second_nearest_center(self, points, centers):
        distances = np.zeros(shape=(points.shape[0], len(centers)))
        for id, c in enumerate(centers):
            distances[:, id] = np.linalg.norm(self.data[points] - self.centroids[c], axis=1)

        nearest_centers = distances.argsort(axis=1)[:, 0]
        second_nearest_centers = distances.argsort(axis=1)[:, 1]
        potencial = set()
        for i, p in enumerate(points):
            cx = self.nearest_centers[p]
            cy = centers[nearest_centers[i]]
            if cx != cy:
                potencial.add(cx)
                potencial.add(cy)
            self.nearest_centers[p] = cy
            self.second_nearest_centers[p] = centers[second_nearest_centers[i]]
        return potencial

    def update_centers(self, centers):
        for c in centers:
            cluster = np.where(self.nearest_centers == c)[0]
            if cluster.shape[0] != 0:
                self.centroids[c, :] = np.mean(self.data[cluster, :], axis=0)

    def recompute_distances(self):
        for id, c in enumerate(self.centroids):
            self.distance_to_center[id, :] = np.linalg.norm(self.data - c, axis=1)

    ##########################################################
    ##                      Results                         ##
    ##########################################################

    def results(self):
        return self.total_SSEDM(), self.max_SSEDM()




class t_k_means:

    def __init__(self, data, k):
        self.data = data
        self.k = k

    def fit(self, S: Proposal, newCj, Ci, Cj):
        AC = set([Ci, Cj])
        Ac_adj = set()
        AP = []
        # Ac_adj += S.get_adjacent_centers([Cj])
        # AP += S.get_affected_points([Cj]).tolist()
        S.update_centroid(Cj, newCj)

        while len(AC) != 0:
            potencial_AC = set()
            Ac_adj = S.get_adjacent_centers(AC)
            AP = S.get_affected_points(AC)
            aux = list(AC | Ac_adj)
            pot = S.update_first_second_nearest_center(AP, aux)
            potencial_AC = potencial_AC | pot
            S.update_centers(aux)
            AC = potencial_AC
            _ = S.update_first_second_nearest_center(AP, aux)

        return S

    
"""import pandas as pd

df = pd.read_csv('datasets_processed/grid.csv')
df_X = np.array(df[df.columns[:-1]])
df_y = np.array(df[df.columns[-1]])
I_kmeans = I_Kmeans_minus_plus(df_X, 3, init='useful')
print(I_kmeans.fit())
"""

from datetime import datetime
from queue import PriorityQueue
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


class Distribution:
    def __init__(self, name='uniform', *, low=0, high=1, mean=0, stddev=1):
        if name != 'uniform' and name != 'normal':
            raise Exception(f'unknown distribution name: {name}')

        if stddev < 0:
            raise Exception('stddev must be non-negative')

        self.name = name
        self.low = low
        self.high = high
        self.mean = mean
        self.stddev = stddev

    def gen_data(self, sample_size):
        if self.name == 'uniform':
            return np.random.default_rng().uniform(self.low, self.high, sample_size)
        elif self.name == 'normal':
            return np.random.default_rng().normal(self.mean, self.stddev, sample_size)


class Index:
    def __init__(self, dim=2, metric='euclid'):
        self.dim = dim
        self.metric = metric

        self.data = None
        self.pivots = None
        self.index = None

    def add_data(self, samples_count: int, distributions: List[Distribution] = None):
        if distributions is None:
            if self.data is None:
                self.data = np.array(
                    np.array([Distribution('uniform').gen_data(samples_count) for _ in range(self.dim)]).transpose())
            else:
                self.data = np.concatenate(
                    (self.data,
                     np.array([Distribution('uniform').gen_data(samples_count) for _ in range(self.dim)]).transpose()),
                    axis=0)
            return

        if self.data is not None and len(distributions) != self.dim:
            raise Exception('wrong dimension of data to add')

        if self.data is None:
            self.data = np.array([distribution.gen_data(samples_count) for distribution in distributions]).transpose()
        else:
            self.data = np.concatenate(
                (self.data,
                 np.array([distribution.gen_data(samples_count) for distribution in distributions]).transpose()),
                axis=0)

        return self

    def select_pivots(self, pivots_count, method='random', *, user_pivots=None):
        if method == 'random':
            rng = np.random.default_rng()
            selected = rng.choice(self.data.shape[0], size=pivots_count, replace=False)
            self.pivots = self.data[selected, :]
        elif method == 'kmeans':
            kmeans = KMeans(pivots_count, n_init='auto').fit(self.data)

            self.pivots = kmeans.cluster_centers_
        elif method == 'user':
            if not user_pivots:
                raise "user pivots not specified"

            pivot_point_distances = np.apply_along_axis(self.calc_distance_data, 1, user_pivots, self.data, self.metric,
                                                        np.identity(self.dim)).transpose()
            # select nearest point in dataset for each pivot
            self.pivots = self.data[np.argmin(pivot_point_distances, axis=0)]

    def calc_index(self, pivots_count, pivot_selection_method='random', *, user_pivots=None):
        self.select_pivots(pivots_count, pivot_selection_method, user_pivots=user_pivots)
        self.index = np.apply_along_axis(self.calc_distance_data, 1, self.pivots, self.data, self.metric,
                                         np.identity(self.dim)).transpose()

    def range_query(self, point, max_range, use_index=True, quiet=False, return_stats=False):
        if self.index is None:
            raise Exception("index is not calculated")

        point = np.array(point)

        start_time = datetime.now()
        distance_calculated_cnt = 0
        possible_points = None

        if use_index:
            distance_from_pivots = np.apply_along_axis(self.calc_distance, 1, self.pivots, point, self.metric,
                                                       np.identity(self.dim)).transpose()
            distance_calculated_cnt += self.pivots.shape[0]
            # if a point does not intersect all spheres it can be ruled out
            possible_points_booleans = np.all(self.index + max_range >= distance_from_pivots, axis=1)
            possible_points = self.data[possible_points_booleans]
        if not use_index:
            possible_points = self.data

        points_in_range = np.empty((0, self.data.shape[1]))
        if possible_points.shape[0] != 0:
            points_in_range_booleans = np.apply_along_axis(self.calc_distance, 1, possible_points,
                                                           point, self.metric,
                                                           np.identity(self.dim)).transpose() <= max_range
            distance_calculated_cnt += possible_points.shape[0]
            points_in_range = possible_points[points_in_range_booleans]
        end_time = datetime.now()

        if not quiet:
            print(f"range query (point={point}, range={max_range}, use_index={use_index})")
            print(f"time: {end_time - start_time}")
            print(
                f"distance calculated: {distance_calculated_cnt} ({distance_calculated_cnt / self.data.shape[0] * 100} % of dataset)")
            print(
                f"selected points: {points_in_range.shape[0]} ({points_in_range.shape[0] / self.data.shape[0] * 100} % of dataset)")

            self.plot(self.data.shape[1], gray=self.data, green=points_in_range, red=self.pivots,
                      blue=np.array([point]))

        if not return_stats:
            return points_in_range
        elif return_stats == 'both':
            return [points_in_range, (end_time - start_time).microseconds, distance_calculated_cnt, self.data.shape[0],
                    points_in_range.shape[0]]
        else:
            return np.array([(end_time - start_time).microseconds, distance_calculated_cnt, self.data.shape[0],
                             points_in_range.shape[0]])

    def knn_query(self, point, k: int, use_index=True, quiet=False, return_stats=False):
        """
        https://moodle-vyuka.cvut.cz/pluginfile.php/602589/mod_page/content/24/thesis.pdf page 56
        """
        if self.index is None:
            raise Exception("index is not calculated")

        point = np.array(point)

        start_time = datetime.now()
        distance_calculated_cnt = 0

        if use_index:
            distance_from_pivots = np.apply_along_axis(self.calc_distance, 1, self.pivots, point, self.metric,
                                                       np.identity(self.dim)).transpose()
            distance_calculated_cnt += self.pivots.shape[0]
            lower_bound = np.max(np.abs(distance_from_pivots - self.index), axis=1)
            lb_order = np.argsort(lower_bound)

            pq = PriorityQueue()
            for item in lb_order:
                if pq.qsize() < k:
                    point_item_distance = self.calc_distance(point, self.data[item], self.metric, np.identity(self.dim))
                    pq.put((-point_item_distance, item))  # minus to use pq as max heap
                    distance_calculated_cnt += 1
                else:
                    if -pq.queue[0][0] < lower_bound[item]:
                        break

                    point_item_distance = self.calc_distance(point, self.data[item], self.metric, np.identity(self.dim))
                    distance_calculated_cnt += 1
                    if point_item_distance < -pq.queue[0][0]:
                        pq.put((-point_item_distance, item))
                        pq.get()

            knn_points = np.array([self.data[item_idx[1]] for item_idx in pq.queue])
        else:
            point_distances = np.apply_along_axis(self.calc_distance, 1, self.data, point, self.metric,
                                                  np.identity(self.dim)).transpose()
            distance_calculated_cnt += self.data.shape[0]
            point_distances_ordered = np.argsort(point_distances)
            knn_points = self.data[point_distances_ordered[:k]]

        end_time = datetime.now()

        if not quiet:
            print(f"knn query (point={point}, k={k}, use_index={use_index})")
            print(f"time: {end_time - start_time}")
            print(
                f"distance calculated: {distance_calculated_cnt} ({distance_calculated_cnt / self.data.shape[0] * 100} % of dataset)")
            print(f"selected points: {k} ({k / self.data.shape[0] * 100} % of dataset)")

            self.plot(self.data.shape[1], gray=self.data, green=knn_points, red=self.pivots,
                      blue=np.array([point]))

        if not return_stats:
            return knn_points
        elif return_stats == 'both':
            return [knn_points, (end_time - start_time).microseconds, distance_calculated_cnt, self.data.shape[0], k]
        else:
            return np.array([(end_time - start_time).microseconds, distance_calculated_cnt, self.data.shape[0], k])

    def save_to_file(self, file_name):
        np.savetxt(f'{file_name}.data', self.data)
        np.savetxt(f'{file_name}.pivots', self.pivots)
        np.savetxt(f'{file_name}.index', self.index)

    def load_from_file(self, file_name):
        self.data = np.loadtxt(f'{file_name}.data')
        self.pivots = np.loadtxt(f'{file_name}.pivots')
        self.index = np.loadtxt(f'{file_name}.index')

        self.dim = self.data.shape[1]

        return self

    def test_range_query(self, point, max_range):
        r1 = self.range_query(point, max_range, use_index=False, quiet=True)
        r2 = self.range_query(point, max_range, use_index=True, quiet=True)

        # sort by first column
        r1 = r1[r1[:, 0].argsort()]
        r2 = r2[r2[:, 0].argsort()]

        return np.array_equal(r1, r2)

    def test_knn_query(self, point, k):
        r1 = self.knn_query(point, k, use_index=False, quiet=True)
        r2 = self.knn_query(point, k, use_index=True, quiet=True)

        # sort by first column
        r1 = r1[r1[:, 0].argsort()]
        r2 = r2[r2[:, 0].argsort()]

        return np.array_equal(r1, r2)

    @staticmethod
    def plot(dim, *, gray=None, green=None, red=None, blue=None):
        if dim == 2:
            if gray is not None:
                plt.scatter(gray[:, 0], gray[:, 1], color='gray')
            if green is not None:
                plt.scatter(green[:, 0], green[:, 1], color='green')
            if red is not None:
                plt.scatter(red[:, 0], red[:, 1], color='red')
            if blue is not None:
                plt.scatter(blue[:, 0], blue[:, 1], color='blue')

        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            if gray is not None:
                ax.scatter(gray[:, 0], gray[:, 1], gray[:, 2], color='gray', alpha=0.05)
            if green is not None:
                ax.scatter(green[:, 0], green[:, 1], green[:, 2], color='green')
            if red is not None:
                ax.scatter(red[:, 0], red[:, 1], red[:, 2], color='red')
            if blue is not None:
                plt.scatter(blue[:, 0], blue[:, 1], blue[:, 2], color='blue')

        plt.show()

    @staticmethod
    def calc_distance_data(point, data, metric, matrix=None):
        """Calculates distance between point and all data."""

        if metric == 'matrix':
            return np.sqrt(np.sum(((point - data) @ matrix) * (point - data), axis=1))

        return np.linalg.norm(point - data, axis=1)

    @staticmethod
    def calc_distance(a, b, metric, matrix=None):
        """Calculates distance between two points."""

        if metric == 'matrix':
            return np.sqrt((a - b) @ matrix @ (a - b).transpose())

        return np.linalg.norm(a - b)

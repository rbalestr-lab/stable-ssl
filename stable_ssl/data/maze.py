from queue import Queue

import matplotlib.pyplot as plt
import numpy as np


class Maze:
    def __init__(self, straightness, turns, traps):
        self.start = np.random.randn(2)
        self.start /= np.max(np.abs(self.start))
        self.end = self.start.copy()
        self.end[0] *= -1
        self.radii = 0.1

        main_path = np.random.rand(turns)
        main_points = main_path.outer(self.start) + (1 - main_path).outer(self.end)
        self.paths = [main_points]

    def on_wall(self, point):
        for path in self.paths:
            if self.on_path(point, path):
                return True
        return False

    def on_path(self, point, path):
        d = np.inf
        for i in range(len(path) - 1):
            d = np.min(d, self.distance_to_segment(point, path[i], path[i + 1]))
        return d < self.radii

    @staticmethod
    def distance_to_segment(point, segment_start, segment_end):
        """
        Calculates the shortest distance between a point and a line segment.

        Args:
            point: A numpy array of shape (2,) representing the point (x, y).
            segment_start: A numpy array of shape (2,) representing the starting point of the segment.
            segment_end: A numpy array of shape (2,) representing the ending point of the segment.

        Returns
        -------
            The shortest distance between the point and the line segment.
        """
        # Calculate the vector of the segment
        segment_vector = segment_end - segment_start

        # If the segment is just a point, calculate the distance to that point
        if np.allclose(segment_vector, 0):
            return np.linalg.norm(point - segment_start, axis=-1)

        # Project the point onto the line containing the segment
        t = np.tensordot(
            point - segment_start, segment_vector, [[-1], [-1]]
        ) / np.linalg.norm(segment_vector, axis=-1)

        # If the projection falls outside the segment, clamp it to the segment endpoints
        t = max(0, min(1, t))

        # Calculate the closest point on the segment to the point
        closest_point = segment_start + t * segment_vector

        # Calculate the distance between the point and the closest point on the segment
        return np.linalg.norm(point - closest_point, axis=-1)


def find_shortest_path(maze, dt):
    # BFS algorithm to find the shortest path
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = (dt, 0)
    end = (maze.shape[0] - 2 * dt, maze.shape[1] - 1)
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))
    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0] + dx, node[1] + dy)
            if next_node == end:
                ij = np.stack(path + [next_node])
                xy = ij + 0.5
                return xy
            if (
                next_node[0] >= 0
                and next_node[1] >= 0
                and next_node[0] < maze.shape[0]
                and next_node[1] < maze.shape[1]
                and maze[next_node] == 0
                and not visited[next_node]
            ):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))


def noise_path(path, radii=1):
    noisy = path.copy().astype("float")
    noisy[1:-1] += np.clip(np.random.randn(len(path) - 2, 2) * radii / 2, -radii, radii)
    return noisy


def make_trajectory(maze, dt):
    # BFS algorithm to find the shortest path
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = (dt, 0)
    end = (maze.shape[0] - 2 * dt, maze.shape[1] - 1)
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))
    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0] + dx, node[1] + dy)
            if next_node == end:
                return path + [next_node]
            if (
                next_node[0] >= 0
                and next_node[1] >= 0
                and next_node[0] < maze.shape[0]
                and next_node[1] < maze.shape[1]
                and maze[next_node] == 0
                and not visited[next_node]
            ):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))


if __name__ == "__main__":
    width = 4
    height = 4
    dt = 2
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2)
    for i, complexity in enumerate([0.2, 0.7]):
        for j, density in enumerate([0.2, 0.7]):
            maze = create_maze(width)
            maze = np.repeat(np.repeat(maze, dt, axis=0), dt, axis=1)
            # print(maze)
            path = find_shortest_path(maze, dt)
            noisy_path = noise_path(path)

            axs[i, j].imshow(maze, aspect="auto", origin="lower")
            x_coords = [x[1] for x in path]
            y_coords = [y[0] for y in path]
            axs[i, j].plot(x_coords, y_coords, color="red", linewidth=2)
            x_coords = [x[1] for x in noisy_path]
            y_coords = [y[0] for y in noisy_path]
            axs[i, j].plot(x_coords, y_coords, color="green", linewidth=2)
    plt.tight_layout()
    plt.savefig("../../assets/data/maze.png")
    plt.close()

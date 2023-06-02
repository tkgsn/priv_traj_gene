import numpy as np
import pickle

class Grid():
    # A Grid instance has a bidirectional mapping between a state and a lat/lon pair
    # Each cell size is variable

    @staticmethod
    def make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins):
        x_axis = np.linspace(lon_range[0], lon_range[1], n_bins+1)
        y_axis = np.linspace(lat_range[0], lat_range[1], n_bins+1)

        # compute the bin size
        x_bin_size = x_axis[1] - x_axis[0]
        y_bin_size = y_axis[1] - y_axis[0]

        # insert the first element (-infty, x_axis[0]) and the last element (x_axis[-1], +infty)
        x_axis = np.insert(x_axis, 0, x_axis[0]-x_bin_size)
        x_axis = np.append(x_axis, x_axis[-1]+x_bin_size)

        # insert the first element (-infty, y_axis[0]) and the last element (y_axis[-1], +infty)
        y_axis = np.insert(y_axis, 0, y_axis[0]-y_bin_size)
        y_axis = np.append(y_axis, y_axis[-1]+y_bin_size)

        ranges = []
        for i in range(len(x_axis)-1):
            for j in range(len(y_axis)-1):
                ranges.append([(x_axis[i], x_axis[i+1]), (y_axis[j], y_axis[j+1])])
        return ranges

    @staticmethod
    def make_ranges_from_privtrace_info(info_path):
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        x_ranges, y_ranges = info
        ranges = []
        for x_ranges_level2, y_ranges_level2 in zip(x_ranges, y_ranges):
            if len(x_ranges_level2) == 2:
                ranges.append([(x_ranges_level2[0], x_ranges_level2[1]), (y_ranges_level2[0], y_ranges_level2[1])])
            else:
                ranges += [[(x_ranges_level2[i], x_ranges_level2[i+1]), (y_ranges_level2[j], y_ranges_level2[j+1])] for i in range(len(x_ranges_level2)-1) for j in range(len(y_ranges_level2)-1)]
        return ranges

    def __init__(self, ranges):
        self.make_grid_from_ranges(ranges)
        self.vocab_size = len(self.grids)
        self.max_distance = self.compute_max_distance()

    def compute_max_distance(self):
        max_distance = 0
        for i, (x_range, y_range) in self.grids.items():
            for j, (x_range2, y_range2) in self.grids.items():
                if i != j:
                    max_distance = max(max_distance, np.sqrt((x_range[0]-x_range2[0])**2 + (y_range[0]-y_range2[0])**2))
        return max_distance

    def save_gps(self, gps_path):

        with open(gps_path, "w") as f:
            for state in self.grids:
                lon_center, lat_center = self.state_to_center_latlon(state)
                f.write(f"{lat_center},{lon_center}\n")


    def make_grid_from_ranges(self, ranges):
        grids = {}
        for i, (x_range, y_range) in enumerate(ranges):
            grids[i] = [x_range, y_range]
        self.grids = grids
        assert not self.check_grid_overlap(), "Grids overlap"

    def state_to_center_latlon(self, state):
        x_range, y_range = self.grids[state]
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        return y_center, x_center

    def state_to_random_latlon_in_the_cell(self, state):
        x_range, y_range = self.grids[state]
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        return y, x

    # check if the grids have overlap
    def check_grid_overlap(self):
        for i, (x_range, y_range) in self.grids.items():
            for j, (x_range2, y_range2) in self.grids.items():
                if i != j and x_range[0] < x_range2[0] < x_range[1] and y_range[0] < y_range2[0] < y_range[1]:
                    print(x_range, x_range2, y_range, y_range2)
                    return True
        return False

    # convert latlon to state by bisect search
    def latlon_to_state(self, lat, lon):
        for state, (x_range, y_range) in self.grids.items():
            if x_range[0] <= lon < x_range[1] and y_range[0] <= lat < y_range[1]:
                return state
        return None
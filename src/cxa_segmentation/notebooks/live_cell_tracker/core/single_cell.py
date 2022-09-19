import numpy as np


class SingleCellStatic:
    """Single cell at one time frame."""

    def __init__(self, timeframe, bbox=None, regionprops=None, img_dataset=None, feature_dict={}) -> None:
        self.regionprops = regionprops
        self.timeframe = timeframe
        self.img_dataset = img_dataset
        self.feature_dict = feature_dict
        self.bbox = bbox

        # infer bbox from regionprops
        if (bbox is None) and (regionprops is not None):
            self.bbox = regionprops.bbox

    def get_img(self):
        return self.img_dataset[self.timeframe]

    def get_bbox(self) -> np.array:
        return np.array(self.bbox)


class SingleCellTrajectory:
    """
    Single cell trajectory containing trajectory information for one single cell at all timeframes.
    """

    def __init__(self, raw_img_dataset, track_id: int = None) -> None:
        self.timeframe_set = set()
        self.timeframe_to_single_cell = {}
        self.raw_img_dataset = raw_img_dataset
        self.raw_total_timeframe = len(raw_img_dataset)
        self.track_id = track_id

    def add_timeframe_data(self, timeframe, cell: SingleCellStatic):
        self.timeframe_to_single_cell[timeframe] = cell
        self.timeframe_set.add(timeframe)

    def get_img(self, timeframe):
        return self.raw_img_dataset[timeframe]

    def get_timeframe_span(self):
        return (min(self.timeframe_set), max(self.timeframe_set))

    def get_timeframe_span_length(self):
        min_t, max_t = self.get_timeframe_span()
        return max_t - min_t

    def get_single_cell(self, timeframe: int) -> SingleCellStatic:
        return self.timeframe_to_single_cell[timeframe]

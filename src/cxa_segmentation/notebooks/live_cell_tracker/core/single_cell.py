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


class SingleCellTrajectory:
    """
    Single cell trajectory containing trajectory information for one single cell at all timeframes.
    """

    def __init__(self, raw_img_dataset) -> None:
        self.timeframe_set = set()
        self.timeframe2singleCell = {}
        self.raw_img_dataset = raw_img_dataset
        self.raw_total_timeframe = len(raw_img_dataset)

    def append_timeframe_data(self, timeframe, cell: SingleCellStatic):
        self.timeframe2singleCell[timeframe] = cell
        self.timeframe_set.add(timeframe)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from live_cell_tracker.core.single_cell import SingleCellTrajectory


def generate_single_trajectory_movie(
    sc_traj: SingleCellTrajectory,
    raw_imgs,
    save_path="./tmp.gif",
    min_length=10,
):
    fig, ax = plt.subplots()

    def init():
        return []

    def update(frame):
        frame_idx, raw_img, bbox, img_crop = frame
        ax.cla()
        frame_text = ax.text(
            -10,
            -10,
            "frame: {}".format(frame_idx),
            fontsize=10,
            color="red",
            ha="center",
            va="center",
        )
        ax.imshow(img_crop)
        return []

    if min_length is not None:
        if sc_traj.get_timeframe_span_length() < min_length:
            print("[Viz] skipping the current trajectory track_id: ", sc_traj.track_id)
            return

    frame_data = []
    for frame_idx in sc_traj.timeframe_to_single_cell:
        sc_timepoint = sc_traj.get_single_cell(frame_idx)
        img = raw_imgs[frame_idx]
        bbox = sc_timepoint.get_bbox()
        min_x, max_x, min_y, max_y = (
            int(bbox[0]),
            int(bbox[2]),
            int(bbox[1]),
            int(bbox[3]),
        )
        img_crop = img[min_x:max_x, min_y:max_y]
        frame_data.append((frame_idx, img, bbox, img_crop))

    ani = FuncAnimation(fig, update, frames=frame_data, init_func=None, blit=True)
    print("saving to: %s..." % save_path)
    ani.save(save_path)

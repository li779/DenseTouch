import time
import threading
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    PerspectiveCameras,
    PointLights,
    look_at_view_transform
)
import imageio
import torch
import matplotlib.pyplot as plt
import numpy as np

def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def render_points(
    points,
    cameras,
    image_size,
    save=False,
    file_prefix='',
    color=[0.7, 0.7, 1]
):
    device = points.device
    if device is None:
        device = get_device()

    # Get the renderer.
    points_renderer = get_points_renderer(image_size=image_size[0], radius=0.01)

    textures = torch.ones(points.size()).to(device)   # (1, N_v, 3)
    rgb = textures * torch.tensor(color).to(device)  # (1, N_v, 3)

    point_cloud = pytorch3d.structures.pointclouds.Pointclouds(
        points=points, features=rgb
    )
    
    all_images = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for cam_idx in range(len(cameras)):
            image = points_renderer(point_cloud, cameras=cameras[cam_idx].to(device))
            image = image[0,:,:,:3].detach().cpu().numpy()
            all_images.append(image)

            # Save
            if save:
                plt.imsave(
                    f'{file_prefix}_{cam_idx}.png',
                    image
                )
    
    return all_images

def create_surround_cameras(radius, n_poses=20, up=(0.0, 1.0, 0.0), focal_length=1.0):
    cameras = []

    for theta in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:

        if np.abs(up[1]) > 0:
            eye = [np.cos(theta + np.pi / 2) * radius, 1.0, -np.sin(theta + np.pi / 2) * radius]
        else:
            eye = [np.cos(theta + np.pi / 2) * radius, np.sin(theta + np.pi / 2) * radius, 2.0]

        R, T = look_at_view_transform(
            eye=(eye,),
            at=([0.0, 0.0, 0.0],),
            up=(up,),
        )

        cameras.append(
            PerspectiveCameras(
                focal_length=torch.tensor([focal_length])[None],
                principal_point=torch.tensor([0.0, 0.0])[None],
                R=R,
                T=T,
            )
        )
    
    return cameras

def render(data, fname):
    data = torch.tensor(data).to(get_device()).float()*1e2
    cameras = create_surround_cameras(20, n_poses=20, up=(1.0, 0.0, 0.0))
    all_images = render_points(data.unsqueeze(0), cameras, [128,128])
    imageio.mimsave(fname, [np.uint8(im * 255) for im in all_images])

class SoftRealTimeClock:
    """
    Convenience class for sleeping in a loop at a specified rate
    """

    def __init__(self, hz=None, period=None):
        assert (
            hz is not None or period is not None
        ), "Use either SoftRealTimeClock(hz=10) or SoftRealTimeClock(period=0.1)"
        self.last_time = self.gettime()
        self.sleep_dur = 1.0 / hz if hz is not None else period

    def gettime(self):
        return time.clock_gettime(time.CLOCK_REALTIME)

    def _remaining(self, curr_time):
        """
        Calculate the time remaining for clock to sleep.
        """
        elapsed = curr_time - self.last_time
        return self.sleep_dur - elapsed

    def _sleep(self, duration):
        if duration < 0:
            return
        time.sleep(duration)

    def sleep(self):
        """
        Attempt sleep at the specified rate.
        """
        curr_time = self.gettime()
        self._sleep(self._remaining(curr_time))
        self.last_time += self.sleep_dur


def test_soft_real_time_clock():
    clock = SoftRealTimeClock(100)
    for i in range(200):
        print(clock.gettime())
        clock.sleep()

    clock = SoftRealTimeClock(period=0.1)
    for i in range(20):
        print(clock.gettime())
        clock.sleep()


class LoopThread(threading.Thread):
    def __init__(self, interval, callback):
        super().__init__()
        self.interval = interval
        self._callback = callback

    def run(self):
        """
        Use a soft real-time clock (Soft RTC) to call callback periodically.
        """
        clock = SoftRealTimeClock(period=self.interval)
        while threading.main_thread().is_alive():
            self._callback()
            clock.sleep()
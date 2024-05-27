import numpy as np
import open3d
from open3d import *

model_path = "./"


class Visualizer:
    def __init__(self):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window(window_name='DGripper', width=1920, height=1080)
        self.pixel_per_mm = 0.0318

        self.points = np.zeros([640 * 480, 3])
        self.X, self.Y = np.meshgrid(
            np.arange(640), np.arange(480))
        Z = np.zeros_like(self.X)
        self.points[:, 0] = np.ndarray.flatten(
            self.X) * self.pixel_per_mm-640*0.5*self.pixel_per_mm
        self.points[:, 1] = np.ndarray.flatten(
            self.Y) * self.pixel_per_mm-480*0.5*self.pixel_per_mm
        self.points[:, 2] = np.ndarray.flatten(Z)

        self.pcd = [open3d.geometry.PointCloud(
        ), open3d.geometry.PointCloud(), open3d.geometry.PointCloud()]
        for i in self.pcd:
            i.points = open3d.utility.Vector3dVector(self.points)
            self.vis.add_geometry(i)
        # self.vis.add_geometry(self.pcd)

        self.colors = np.zeros([self.points.shape[0], 3])
        coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[
                                                                     0, 0, 0])
        self.vis.add_geometry(coord)

        self.ctr = self.vis.get_view_control()
        self.ctr.change_field_of_view(-25)
        print("fov", self.ctr.get_field_of_view())
        self.ctr.convert_to_pinhole_camera_parameters()
        self.ctr.set_zoom(0.7)
        # self.ctr.rotate(0, -40)  # mouse drag in x-axis, y-axis
        self.ctr.set_front([0, 1, 0])
        self.ctr.set_up([0, 0, 1])
        self.vis.update_renderer()

    def update(self, points, gradients, yaw, roll):
        dx, dy = gradients
        np_colors = dx + dy
        if abs(np_colors.max()) > 0:
            np_colors = (np_colors - np_colors.min()) / \
                (np_colors.max() - np_colors.min()) * 0.6 + 0.2
        np_colors = np.ndarray.flatten(np_colors)

        # set the non-contact areas as black
        np_colors[points[:, 2] <= 0.08] = 0
        for _ in range(3):
            self.colors[:, _] = np_colors

        for i in self.pcd:
            i.points = open3d.utility.Vector3dVector(points)
            i.colors = open3d.utility.Vector3dVector(self.colors)
            i.translate([40, 0, 0])
            i.rotate(open3d.geometry.get_rotation_matrix_from_xyz(
                [0, -roll, 0]), center=(0, 0, 0))

        self.pcd[1].rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [0, 0, -yaw]), center=(0, 0, 0))
        self.pcd[2].rotate(open3d.geometry.get_rotation_matrix_from_xyz(
            [0, 0, yaw]), center=(0, 0, 0))

        r = 35
        # for i in range(3):
        self.pcd[0].translate([r, 0, 0])
        self.pcd[2].translate([-0.5*r, 0.866*r, 0])
        self.pcd[1].translate([-0.5*r, -0.866*r, 0])

        for i in self.pcd:
            self.vis.update_geometry(i)
        self.vis.poll_events()
        self.vis.update_renderer()
        # return self.pcd


if __name__ == "__main__":
    vis = Visualizer()
    # vis.vis.run()
    while 1:
        vis.points[:, 2] = np.random.randn(640*480) * 0.1
        # vis.update(vis.points, np.zeros_like(2, 640*480))
        yaw = np.random.randn(1)*np.pi/6
        roll = np.random.randn(1)*np.pi/3
        yaw = np.pi/3*2
        roll = np.pi/2+np.pi/6
        vis.update(vis.points, np.zeros([2, 640*480]), yaw, roll)
        # vis.update(vis.points, np.ones([2, 640*480]))
        # vi

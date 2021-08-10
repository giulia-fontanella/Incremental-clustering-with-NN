import numpy as np
from PAL.Learn.EnvironmentModels.MapModel import MapModel


class Mapper:

    def __init__(self):
        # Set map model
        self.map_model = MapModel()


    def get_point_cloud(self, depth_matrix):

        # Get intrinsic parameters
        height, width = depth_matrix.shape
        K = self.intrinsic_from_fov(height, width, 90)  # +- 45 degrees
        K_inv = np.linalg.inv(K)

        # Get pixel coordinates
        pixel_coords = self.pixel_coord_np(width, height)  # [3, npoints]

        # Apply back-projection: K_inv @ pixels * depth
        cam_coords = K_inv[:3, :3] @ pixel_coords * depth_matrix.flatten()

        return cam_coords

    """
    def get_map_point_from_depth_pixel(self, depth_matrix, angle, pos, pixel_pos, camera_pitch_angle):

        # pixel_pos is (x=column, y=row)

        # Get intrinsic parameters
        height, width = depth_matrix.shape
        K = self.intrinsic_from_fov(height, width, 90)  # +- 45 degrees
        K_inv = np.linalg.inv(K)

        # Get pixel coordinates
        pixel_coords = self.pixel_coord_np(width, height)  # [3, npoints]

        # Apply back-projection: K_inv @ pixels * depth
        cam_coords = K_inv[:3, :3] @ pixel_coords * depth_matrix.flatten()

        # Rescale depth_matrix values according to camera pitch before projection in the xy plane

        x, y, z = cam_coords

        # flip the y-axis to positive upwards
        cam_coords = np.array((x, -y, z))

        angle = (angle - 90) % 360 # rescale angle according to simulator reference system

        # Get agent depth view occupancy points
        occupancy_points = np.column_stack((x, z))
        rot_matrix = np.array(([np.cos(np.deg2rad(angle)), - np.sin(np.deg2rad(angle))],
                               [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]))

        # Rotate agent view according to agent orientation
        occupancy_points = np.dot(occupancy_points, rot_matrix.T)

        # Add agent offset position to agent view
        occupancy_points[:,0] += pos['x']
        occupancy_points[:,1] += pos['y']

        # Here -y is z (height)
        map_coords = np.array((occupancy_points[:,0], occupancy_points[:,1], -y)).reshape(depth_matrix.shape[0], depth_matrix.shape[1])

        return map_coords[pixel_pos[1]][pixel_pos[0]]
    """

    def update_topview(self, depth_matrix, file_name, angle, pos):

        angle = (angle - 90) % 360 # rescale angle according to simulator reference system

        # Limit points to 150m in the z-direction for visualisation
        # cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]
        x, y, z = self.get_point_cloud(depth_matrix)

        # flip the y-axis to positive upwards
        cam_coords = np.array((x, -y, z))

        # Filter cam coordinates according to agent view horizon
        cam_coords = cam_coords[:, np.where(cam_coords[2] <= 20)[0]] # ???

        # Filter cam coordinates according to agent height
        cam_coords = cam_coords[:, np.where(cam_coords[1] <= 0)[0]]
        cam_coords = cam_coords[:, np.where(cam_coords[1] >= -1.5)[0]]

        # # Visualize 3D point cloud
        # pcd_cam = o3d.geometry.PointCloud()
        # pcd_cam.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
        # # Flip it, otherwise the pointcloud will be upside down
        # pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd_cam])

        # Do top view projection
        # Get camera points
        x, y, z = cam_coords

        # Get agent depth view occupancy points
        occupancy_points = np.column_stack((x, z))
        rot_matrix = np.array(([np.cos(np.deg2rad(angle)), - np.sin(np.deg2rad(angle))],
                               [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]))

        # Rotate agent view according to agent orientation
        occupancy_points = np.dot(occupancy_points, rot_matrix.T)

        # Add agent offset position to agent view
        occupancy_points[:,0] += pos['x']
        occupancy_points[:,1] += pos['y']

        # Update map model occupancy points
        self.map_model.update_occupancy(occupancy_points, pos, angle, file_name)

    def pixel_coord_np(self, width, height):
        """
        Pixel in homogenous coordinate
        Returns:
            Pixel coordinate:       [3, width * height]
        """
        x = np.linspace(0, width - 1, width).astype(np.int)
        y = np.linspace(0, height - 1, height).astype(np.int)
        [x, y] = np.meshgrid(x, y)
        return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


    def intrinsic_from_fov(self,height, width, fov=90):
        """
        Basic Pinhole Camera Model
        intrinsic params from fov and sensor width and height in pixels
        Returns:
            K:      [4, 4]
        """
        px, py = (width / 2, height / 2)
        hfov = fov / 360. * 2. * np.pi
        fx = width / (2. * np.tan(hfov / 2.))

        vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
        fy = height / (2. * np.tan(vfov / 2.))

        return np.array([[fx,  0, px, 0.],
                         [ 0, fy, py, 0.],
                         [ 0,  0, 1., 0.],
                         [0., 0., 0., 1.]])



import numpy as np
from FindFeatures import FindFeatures
import open3d as o3d

if __name__ == '__main__':
    pcd = o3d.geometry.PointCloud()
    K = np.array([[5.904732862936928541e+03, 0, 1.582265898473643801e+03], [0, 5.899357315262350312e+03, 2.345954788306215505e+03], [0, 0, 1]])
    # K = np.array([[2.81333732e+03, 0, 1.57412539e+03], [0, 2.80789073e+03, 2.04139012e+03], [0, 0, 1]])
    features = FindFeatures("rmbg", K)
    points_3d, points_color = features.find_features()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(points_color)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
                                                       

                                                                                 
                                                                             
                                                                         
   
from __future__ import annotations
import argparse,base64,contextlib,errno,gzip,hashlib,io,json,os,queue,sys,threading,urllib.parse,webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from scipy.optimize import linear_sum_assignment
os.environ.setdefault("MPLBACKEND","Agg")
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from sklearn.cluster import DBSCAN
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
from matplotlib.patches import Rectangle

DATASET_PATH = "/Users/islamelsayed/Documents/Python/Project"
RANSAC_DISTANCE_THRESHOLD = 0.15
RANSAC_N = 3
RANSAC_ITERATIONS = 3000
DBSCAN_EPS = 0.4
DBSCAN_MIN_SAMPLES = 8
ENABLE_GROUND_MODEL_GUARD = True
ENABLE_ELEVATED_FRAGMENT_FILTER = True
MIN_CLUSTER_POINTS = 10
SHOW_VISUALIZATIONS = False

ROOT=Path(__file__).resolve().parent
nusc=None
SCENE_LOCK=threading.Lock()
PAYLOADS={}


def initialize_dataset():
    global nusc
    if nusc is None:
        print("Loading nuScenes Mini from",DATASET_PATH,flush=True)
        nusc=NuScenes(version="v1.0-mini",dataroot=DATASET_PATH,verbose=True)
    return nusc


def backend():
    initialize_dataset()
    return sys.modules[__name__]


def load_lidar_scan(sample):

                            
    lidar_token = sample["data"]["LIDAR_TOP"]

                         
    lidar_path = nusc.get_sample_data_path(lidar_token)

    raw_points = np.fromfile(
        lidar_path,
        dtype=np.float32
    ).reshape(-1, 5)

                             
    xyz = raw_points[:, :3].astype(np.float64)

                           
    lidar_data = nusc.get("sample_data", lidar_token)

    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        lidar_data["calibrated_sensor_token"]
    )

                                  
    T_ego_sensor = transform_matrix(
        calibrated_sensor["translation"],
        Quaternion(calibrated_sensor["rotation"])
    )

    rotation = T_ego_sensor[:3, :3]
    translation = T_ego_sensor[:3, 3]

                                                      
    xyz = np.einsum(
        "ij,nj->ni",
        rotation,
        xyz
    ) + translation

                                      
    if not np.isfinite(xyz).all():
        raise ValueError("Invalid values detected in transformed LiDAR points.")

    print("\nLIDAR TRANSFORMATION")
    print("Number of points:", len(xyz))
    print("All points finite:", np.isfinite(xyz).all())
    print("XYZ range:", xyz.min(), "to", xyz.max())

                               
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz)

    return xyz, pcd


def load_radar_scan(sample):
                           
    radar_token = sample["data"]["RADAR_FRONT"]
                           
    radar_data = nusc.get("sample_data", radar_token)
                            
    radar_path = nusc.get_sample_data_path(radar_token)
    radar = RadarPointCloud.from_file(radar_path)
                      
    radar_xyz = radar.points[:3, :]
                                             
    vx_comp = radar.points[8, :]
    vy_comp = radar.points[9, :]
                                 
    quality_valid = radar.points[10, :]
                           
    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        radar_data["calibrated_sensor_token"]
    )
    ego_pose = nusc.get(
        "ego_pose",
        radar_data["ego_pose_token"]
    )
                                     
    T_ego_sensor = transform_matrix(
        calibrated_sensor["translation"],
        Quaternion(calibrated_sensor["rotation"])
    )
                                     
    T_global_ego = transform_matrix(
        ego_pose["translation"],
        Quaternion(ego_pose["rotation"])
    )
                                        
    T_global_sensor = T_global_ego @ T_ego_sensor
                                     
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_pose = nusc.get(
        "ego_pose",
        lidar_data["ego_pose_token"]
    )
    T_global_reference = transform_matrix(
        lidar_pose["translation"],
        Quaternion(lidar_pose["rotation"])
    )
                                               
    T_reference_global = np.linalg.inv(T_global_reference)
                             
    T_reference_sensor = (
        T_reference_global @ T_global_sensor
    )
                               
    homogeneous = np.vstack((
        radar_xyz,
        np.ones(radar_xyz.shape[1])
    ))
    radar_xyz_reference = (
        T_reference_sensor @ homogeneous
    )[:3, :].T
                                                                 
                                                          
    R_reference_sensor = T_reference_sensor[:3, :3]
    radar_velocity_sensor = np.vstack((
        vx_comp,
        vy_comp,
        np.zeros(len(vx_comp))
    ))
    radar_velocity_reference = (
        R_reference_sensor @ radar_velocity_sensor
    ).T
                                    
    time_difference = (
        radar_data["timestamp"] - lidar_data["timestamp"]
    ) / 1000.0
    print("\nRADAR PROCESSING RESULTS")
    print("RADAR returns:", radar_xyz_reference.shape[0])
    print("Timestamp difference:", time_difference, "ms")
    print("RADAR position range X:",
          radar_xyz_reference[:, 0].min(),
          "to",
          radar_xyz_reference[:, 0].max())
    print("RADAR position range Y:",
          radar_xyz_reference[:, 1].min(),
          "to",
          radar_xyz_reference[:, 1].max())
    print("RADAR compensated velocity X range:",
          radar_velocity_reference[:, 0].min(),
          "to",
          radar_velocity_reference[:, 0].max())
    print("RADAR compensated velocity Y range:",
          radar_velocity_reference[:, 1].min(),
          "to",
          radar_velocity_reference[:, 1].max())
    return {
        "positions": radar_xyz_reference,
        "velocities": radar_velocity_reference,
        "quality_valid": quality_valid,
        "timestamp_difference_ms": time_difference
    }


def visualize_sensor_fusion(lidar_points, radar_results):
                                          
    mask = (
    (lidar_points[:, 0] > 0) &
    (lidar_points[:, 0] < 100) &
    (lidar_points[:, 1] > -50) &
    (lidar_points[:, 1] < 50)
)

    lidar_points = lidar_points[mask]
                              
    lidar_pcd = o3d.geometry.PointCloud()
    lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points)
                  
    lidar_pcd.paint_uniform_color([0.0, 0.3, 1.0])
                              
    radar_pcd = o3d.geometry.PointCloud()
    radar_pcd.points = o3d.utility.Vector3dVector(
        radar_results["positions"]
    )
                 
    radar_pcd.paint_uniform_color([1.0, 0.0, 0.0])
                     
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0,
        origin=[0, 0, 0]
    )
    print("\nDisplaying LiDAR and RADAR together...")
    print("Blue points: LiDAR")
    print("Red points: RADAR")
    o3d.visualization.draw_geometries(
        [lidar_pcd, radar_pcd, axes],
        window_name="LiDAR + RADAR Sensor Visualization",
        width=1200,
        height=800
    )


def ground_model_is_plausible(plane_model):
                                                                              
    plane = np.asarray(plane_model, dtype=float)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        return False
    normal_length = float(np.linalg.norm(plane[:3]))
    if normal_length < 1e-9 or abs(plane[2]) < 1e-9:
        return False
    cosine = np.clip(abs(plane[2]) / normal_length, 0.0, 1.0)
    tilt_degrees = float(np.degrees(np.arccos(cosine)))
    height_at_origin = float(-plane[3] / plane[2])
    return tilt_degrees <= 15.0 and abs(height_at_origin) <= 0.50


def fit_road_ground_fallback(pcd):
                                                                          

                                                                          
                                                                           
       
    points = np.asarray(pcd.points)
    radial_distance = np.linalg.norm(points[:, :2], axis=1)
    seed_mask = (
        (radial_distance >= 2.0) & (radial_distance <= 30.0)
        & (points[:, 2] >= -0.50) & (points[:, 2] <= 0.50)
    )
    plane = np.array([0.0, 0.0, 1.0, 0.0])
    mode = "horizontal_fallback"
    if int(seed_mask.sum()) >= 50:
        seed_cloud = o3d.geometry.PointCloud()
        seed_cloud.points = o3d.utility.Vector3dVector(points[seed_mask])
        o3d.utility.random.seed(12345)
        candidate, _ = seed_cloud.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            ransac_n=RANSAC_N,
            num_iterations=RANSAC_ITERATIONS,
        )
        if ground_model_is_plausible(candidate):
            plane = np.asarray(candidate, dtype=float)
            mode = "road_seed_fit"
    normal_length = float(np.linalg.norm(plane[:3]))
    distances = np.abs(np.einsum("ij,j->i", points, plane[:3]) + plane[3]) / normal_length
    inliers = np.flatnonzero(distances <= RANSAC_DISTANCE_THRESHOLD).tolist()
    return plane, inliers, mode, int(seed_mask.sum())


def remove_ground(pcd):
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERATIONS
    )



    if ENABLE_GROUND_MODEL_GUARD and not ground_model_is_plausible(plane_model):
        print("GROUND MODEL GUARD: replacing implausible dominant plane:", plane_model)
        plane_model, inliers, mode, seed_count = fit_road_ground_fallback(pcd)
        print("Replacement ground mode:", mode, "| seed points:", seed_count)
        print("Replacement plane:", plane_model)

    ground = pcd.select_by_index(inliers)

  

    obstacles = pcd.select_by_index(
        inliers,
        invert=True
    )

    return ground, obstacles, plane_model


def cluster_obstacles(obstacles):
       
                                         

            
                                                       
                                                      
       

    obstacle_points = np.asarray(obstacles.points)

                                                     

    if len(obstacle_points) == 0:
        return obstacle_points, np.array([], dtype=int)


    dbscan = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES
    )

    labels = dbscan.fit_predict(obstacle_points)

    return obstacle_points, labels


def detect_clusters(obstacle_points, labels):
  

    detections = []

    if len(labels) == 0:
        return detections

                                        
                                              

    cluster_labels = sorted(set(labels) - {-1})

    for label in cluster_labels:

        

        cluster_points = obstacle_points[labels == label]

       

        if len(cluster_points) < MIN_CLUSTER_POINTS:
            continue

        cluster_pcd = o3d.geometry.PointCloud()

        cluster_pcd.points = o3d.utility.Vector3dVector(
            cluster_points
        )


        bbox = cluster_pcd.get_axis_aligned_bounding_box()


        bbox.color = (1.0, 0.0, 0.0)


        center = bbox.get_center()

       

        dimensions = bbox.get_extent()

        distance = np.linalg.norm(center)

        detection = {
            "label": int(label),
            "number_of_points": len(cluster_points),
            "center": center,
            "dimensions": dimensions,
            "distance": distance,
            "bounding_box": bbox,
            "point_cloud": cluster_pcd
        }

        detections.append(detection)

    return detections


def filter_elevated_horizontal_fragments(detections, plane_model):
                                                             

                                                                       
                                                                         
                                                                  
       
    if not ground_model_is_plausible(plane_model):
        return detections, 0
    plane = np.asarray(plane_model, dtype=float).copy()
    plane /= np.linalg.norm(plane[:3])
    if plane[2] < 0:
        plane = -plane
    retained = []
    rejected = 0
    for detection in detections:
        points = np.asarray(detection["point_cloud"].points)
        clearance = float(np.min(np.einsum("ij,j->i", points, plane[:3]) + plane[3]))
        dx, dy, height = detection["dimensions"]
        if clearance > 1.5 and height < 1.5 and max(dx, dy) > 1.3 * height:
            rejected += 1
        else:
            retained.append(detection)
    return retained, rejected


def filter_object_candidates(detections):

    filtered = []

    rejected_counts = {
    "too_flat": 0,
    "too_large": 0,
    "too_narrow": 0,
    "ego_vehicle": 0
}

    for detection in detections:

        dx, dy, dz = detection["dimensions"]

                                               
        center = detection["center"]

        if (
            abs(center[0]) < 1.5
            and abs(center[1]) < 1.5
            and detection["number_of_points"] > 1000
        ):
            rejected_counts["ego_vehicle"] += 1
            continue

        if dz < 0.4:
            rejected_counts["too_flat"] += 1
            continue

        if dx > 11.0 or dy > 8.0 or dz > 5.0:
            rejected_counts["too_large"] += 1
            continue

        if dx < 0.3 or dy < 0.3:
            rejected_counts["too_narrow"] += 1
            continue

        filtered.append(detection)

    print("\nFILTER REJECTION REASONS")

    for reason, count in rejected_counts.items():
        print(f"{reason}: {count}")

    return filtered


def associate_radar_with_lidar(detections, radar_results):

    radar_positions = radar_results["positions"]
    radar_velocities = radar_results["velocities"]

    ASSOCIATION_MARGIN = 1.0          

                                                                
    assigned_velocities = [[] for _ in detections]

                                         
    for radar_index, radar_point in enumerate(radar_positions):

        best_detection_index = None
        best_distance = float("inf")

        for detection_index, detection in enumerate(detections):

            bbox = detection["bounding_box"]

                                             
            min_bound = bbox.get_min_bound()[:2] - ASSOCIATION_MARGIN
            max_bound = bbox.get_max_bound()[:2] + ASSOCIATION_MARGIN

            radar_xy = radar_point[:2]

                                                                   
            inside = np.all(
                (radar_xy >= min_bound) &
                (radar_xy <= max_bound)
            )

            if not inside:
                continue

                                                   
            lidar_center = detection["center"][:2]

            distance = np.linalg.norm(radar_xy - lidar_center)

            if distance < best_distance:
                best_distance = distance
                best_detection_index = detection_index

                                                        
        if best_detection_index is not None:
            assigned_velocities[best_detection_index].append(
                radar_velocities[radar_index, :2]
            )

                              
    for detection_index, detection in enumerate(detections):

        velocities = assigned_velocities[detection_index]

        detection["radar_points"] = len(velocities)

        if len(velocities) > 0:

            detection["radar_velocity"] = np.mean(
                np.asarray(velocities),
                axis=0
            )

            detection["radar_available"] = True

        else:

            detection["radar_velocity"] = None
            detection["radar_available"] = False

    return detections


def visualize_results(pcd, ground, obstacles, detections):
       
                                                           
       

                                          
    print("\nDisplaying original LiDAR point cloud...")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="1 - Original LiDAR Point Cloud",
        width=1280,
        height=720
    )

                              
    ground.paint_uniform_color([0.5, 0.5, 0.5])
    obstacles.paint_uniform_color([1.0, 0.3, 0.0])

    print("Displaying ground removal...")

    o3d.visualization.draw_geometries(
        [ground, obstacles],
        window_name="2 - Ground Removal",
        width=1280,
        height=720
    )

                                        
    geometries = []

    for detection in detections:
        bbox = detection["bounding_box"]
        bbox.color = (1.0, 0.0, 0.0)

        geometries.append(detection["point_cloud"])
        geometries.append(bbox)

    if geometries:
        print("Displaying detected clusters and bounding boxes...")

        o3d.visualization.draw_geometries(
            geometries,
            window_name="3 - 3D Obstacle Detection",
            width=1280,
            height=720
        )
    else:
        print("No clusters available for visualization.")


def create_thick_bbox(bbox, color, radius=0.04):

    points = np.asarray(bbox.get_box_points())

                                          
    lines = [
        [0, 1], [0, 2], [0, 3],
        [1, 6], [1, 7],
        [2, 5], [2, 7],
        [3, 5], [3, 6],
        [4, 5], [4, 6], [4, 7]
    ]

    geometries = []

    for start, end in lines:

        p1 = points[start]
        p2 = points[end]

        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length < 1e-6:
            continue

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=length,
            resolution=8
        )

        cylinder.compute_vertex_normals()
        cylinder.paint_uniform_color(color)

        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.cross([0, 0, 1], direction / length)
            * np.arccos(
                np.clip(
                    np.dot([0, 0, 1], direction / length),
                    -1.0,
                    1.0
                )
            )
        )

        cylinder.rotate(rotation, center=(0, 0, 0))
        cylinder.translate((p1 + p2) / 2)

        geometries.append(cylinder)

    return geometries


def visualize_fusion_results(pcd, detections, radar_results):

    print("\nPreparing Window 4...")

                       
    lidar_pcd = o3d.geometry.PointCloud()
                                      
    all_points = np.asarray(pcd.points)

    mask = (
    (all_points[:, 0] >= -10) &
    (all_points[:, 0] <= 40) &
    (all_points[:, 1] >= -20) &
    (all_points[:, 1] <= 20)
)

    lidar_pcd.points = o3d.utility.Vector3dVector(
        all_points[mask]
)
        
    
    lidar_pcd.paint_uniform_color([0.0, 0.3, 1.0])

                      
    radar_pcd = o3d.geometry.PointCloud()
    radar_pcd.points = o3d.utility.Vector3dVector(
        radar_results["positions"]
    )
    radar_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    geometries = [lidar_pcd, radar_pcd]

                                    
    for detection in detections:

        original_bbox = detection["bounding_box"]

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            original_bbox.get_min_bound(),
            original_bbox.get_max_bound()
        )

        if detection["radar_available"]:
            bbox.color = [0.0, 1.0, 0.0]
        else:
            bbox.color = [0.5, 0.5, 0.5]

        geometries.append(bbox)

    print("Number of geometries:", len(geometries))
    print("Opening Window 4 now...")

    
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="4 - LiDAR RADAR Association",
        width=1280,
        height=720
    )

    for geometry in geometries:
        vis.add_geometry(geometry)

    render = vis.get_render_option()
    render.point_size = 4.0
    render.line_width = 3.0
    render.background_color = np.array([0.05, 0.05, 0.05])

    vis.run()
    vis.destroy_window()


def inspect_pedestrian_points(
    sample,
    xyz,
    obstacles,
    obstacle_points,
    labels
):
       
                                                      

                                                    
                                     
       

    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)

    ego_pose = nusc.get(
        "ego_pose",
        lidar_data["ego_pose_token"]
    )

    rotation = Quaternion(
        ego_pose["rotation"]
    ).rotation_matrix

    translation = np.asarray(
        ego_pose["translation"],
        dtype=float
    )

    print("\n" + "=" * 55)
    print("PEDESTRIAN POINT INSPECTION")
    print("=" * 55)

    for annotation_token in sample["anns"]:

        annotation = nusc.get(
            "sample_annotation",
            annotation_token
        )

        if not annotation["category_name"].startswith(
            "human.pedestrian."
        ):
            continue

        center_global = np.asarray(
            annotation["translation"]
        )

        center_ego = rotation.T @ (
            center_global - translation
        )

        if np.linalg.norm(center_ego[:2]) > 25:
            continue

        radius = 0.8

                        
        original_distances = np.linalg.norm(
            xyz[:, :2] - center_ego[:2],
            axis=1
        )

        original_count = int(
            np.sum(original_distances < radius)
        )

                              
        remaining_distances = np.linalg.norm(
            obstacle_points[:, :2] - center_ego[:2],
            axis=1
        )

        remaining_mask = remaining_distances < radius
        remaining_count = int(np.sum(remaining_mask))

                                      
        local_labels = labels[remaining_mask]

        cluster_labels = set(local_labels) - {-1}

        noise_count = int(
            np.sum(local_labels == -1)
        )

        print(
            f"\nPedestrian at "
            f"{np.round(center_ego[:2], 2)}"
        )

        print("Original LiDAR points:", original_count)
        print("After ground removal:", remaining_count)
        print("DBSCAN noise points:", noise_count)
        print("DBSCAN cluster labels:", sorted(cluster_labels))


def process_scan(sample, visualize=False, random_seed=1700):
    start_time = time.perf_counter()
    (xyz, pcd) = load_lidar_scan(sample)
    total_points = len(xyz)
    o3d.utility.random.seed(random_seed)
    (ground, obstacles, plane_model) = remove_ground(pcd)
    ground_points = len(ground.points)
    obstacle_points_count = len(obstacles.points)
    (obstacle_points, labels) = cluster_obstacles(obstacles)
    detections = detect_clusters(obstacle_points, labels)
    raw_detections = detections
    raw_detection_count = len(raw_detections)
    detections = filter_object_candidates(raw_detections)
    fragment_rejected_count = 0
    if ENABLE_ELEVATED_FRAGMENT_FILTER:
        (detections, fragment_rejected_count) = filter_elevated_horizontal_fragments(detections, plane_model)
    retained_parent_count = len(detections)
    print('\nOBJECT CANDIDATE FILTERING')
    print('Raw clusters:', raw_detection_count)
    print('Clusters retained after geometric filters:', retained_parent_count)
    print('Clusters rejected:', raw_detection_count - retained_parent_count)
    print('Elevated horizontal fragments rejected:', fragment_rejected_count)
    print('Final detections:', len(detections))
    if len(labels) > 0:
        noise_points = int(np.sum(labels == -1))
    else:
        noise_points = 0
    radar_results = load_radar_scan(sample)
    detections = associate_radar_with_lidar(detections, radar_results)
    associated_count = sum((detection['radar_available'] for detection in detections))
    elapsed_time = time.perf_counter() - start_time
    print('\n' + '=' * 55)
    print('LIDAR-RADAR SENSOR PROCESSING RESULTS')
    print('=' * 55)
    print('\nLIDAR RESULTS')
    print('Total LiDAR points:', total_points)
    print('Ground points:', ground_points)
    print('Non-ground points:', obstacle_points_count)
    print('Raw clusters:', raw_detection_count)
    print('Detected clusters after filtering:', len(detections))
    print('DBSCAN noise points:', noise_points)
    print('Estimated plane:', plane_model)
    print('\nRADAR RESULTS')
    print('RADAR returns:', len(radar_results['positions']))
    print('Timestamp difference:', radar_results['timestamp_difference_ms'], 'ms')
    print('\nASSOCIATION RESULTS')
    print('LiDAR detections:', len(detections))
    print('Detections with RADAR:', associated_count)
    print('Detections without RADAR:', len(detections) - associated_count)
    print('\nINDIVIDUAL DETECTIONS')
    for detection in detections:
        print(f"\nCluster {detection['label']}:")
        print('LiDAR center:', np.round(detection['center'], 2))
        print('Distance:', f"{detection['distance']:.2f} m")
        print('Dimensions:', np.round(detection['dimensions'], 2))
        print('LiDAR points in cluster:', detection['number_of_points'])
        print('RADAR returns associated:', detection['radar_points'])
        if detection['radar_available']:
            print('Average RADAR velocity (m/s):', np.round(detection['radar_velocity'], 2))
        else:
            print('RADAR velocity: Not available')
    print(f'\nTotal processing time: {elapsed_time:.3f} seconds')
    if visualize:
        print('\nStarting LiDAR visualizations...')
        visualize_results(pcd, ground, obstacles, detections)
        print('\nOpening LiDAR-RADAR association window...')
        visualize_fusion_results(pcd, detections, radar_results)
        print('Visualizations completed.')
    return {'total_points': total_points, 'ground_points': ground_points, 'obstacle_points': obstacle_points_count, 'clusters': len(detections), 'noise_points': noise_points, 'time_seconds': elapsed_time, 'detections': detections, 'radar_results': radar_results, 'associated_detections': associated_count}


def convert_detections_to_global(sample, detections):

    lidar_token = sample["data"]["LIDAR_TOP"]

    lidar_data = nusc.get(
        "sample_data",
        lidar_token
    )

    ego_pose = nusc.get(
        "ego_pose",
        lidar_data["ego_pose_token"]
    )

    T_global_ego = transform_matrix(
        ego_pose["translation"],
        Quaternion(ego_pose["rotation"])
    )

    rotation = T_global_ego[:3, :3]
    translation = T_global_ego[:3, 3]

    for detection in detections:

        center_ego = detection["center"]

        center_global = (
            rotation @ center_ego + translation
        )

        detection["center_global"] = center_global

        if detection["radar_available"]:

            velocity_ego = np.array([
                detection["radar_velocity"][0],
                detection["radar_velocity"][1],
                0.0
            ])

            velocity_global = rotation @ velocity_ego

            detection["radar_velocity_global"] = (
                velocity_global[:2]
            )

        else:
            detection["radar_velocity_global"] = None

    return detections


class KalmanFilter2D:
       
                                       

          
                      

                   
              

                   
                
       

    def __init__(self, position, velocity=None):

                       
        self.x = np.zeros(4, dtype=float)

        self.x[:2] = position

        if velocity is not None:
            self.x[2:] = velocity

                             
        self.P = np.diag([
            4.0, 4.0, 25.0, 25.0
        ])

                                       
        self.R_lidar = np.diag([
            0.5**2, 0.5**2
        ])

                                    
        self.R_radar = np.diag([
            1.0**2, 1.0**2
        ])

                                  
        self.H_lidar = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

                                  
        self.H_radar = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

    def predict(self, dt):

                                        
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

                                                 
        acceleration_std = 2.0

        G = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])

        Q = acceleration_std**2 * (G @ G.T)

        self.x = F @ self.x

        self.P = F @ self.P @ F.T + Q

    def update(self, measurement, H, R):

        z = np.asarray(measurement, dtype=float)

                                
        innovation = z - H @ self.x

        S = H @ self.P @ H.T + R

                     
        K = np.linalg.solve(
            S,
            H @ self.P
        ).T

                       
        self.x = self.x + K @ innovation

                                  
        I = np.eye(4)
        A = I - K @ H

        self.P = A @ self.P @ A.T + K @ R @ K.T

    def update_lidar(self, position):

        self.update(
            position,
            self.H_lidar,
            self.R_lidar
        )

    def update_radar(self, velocity):

        self.update(
            velocity,
            self.H_radar,
            self.R_radar
        )

    def get_state(self):

        return {
            "position": self.x[:2].copy(),
            "velocity": self.x[2:].copy()
        }


class MultiObjectTracker:

    def __init__(self, max_distance=3.0, max_missed=2):
        self.tracks = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, detections, dt):

                                     
        for track in self.tracks.values():
            track["filter"].predict(dt)
            track["missed"] += 1

                                                 
        candidates = []

        for track_id, track in self.tracks.items():

            predicted_position = track["filter"].x[:2]

            for detection_index, detection in enumerate(detections):

                measured_position = detection["center_global"][:2]

                distance = np.linalg.norm(
                    predicted_position - measured_position
                )

                if distance <= self.max_distance:
                    candidates.append((
                        distance,
                        track_id,
                        detection_index
                    ))

                                                
        candidates.sort(key=lambda item: item[0])

        matched_tracks = set()
        matched_detections = set()

        for distance, track_id, detection_index in candidates:

            if track_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            track = self.tracks[track_id]
            detection = detections[detection_index]

                                       
            track["filter"].update_lidar(
                detection["center_global"][:2]
            )

                                                        
            if detection["radar_available"]:
                track["filter"].update_radar(
                    detection["radar_velocity_global"]
                )

            track["missed"] = 0
            track["hits"] += 1

            detection["track_id"] = track_id

            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

                                                    
        for detection_index, detection in enumerate(detections):

            if detection_index in matched_detections:
                continue

            velocity = detection["radar_velocity_global"]

            kf = KalmanFilter2D(
                position=detection["center_global"][:2],
                velocity=velocity
            )

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                "filter": kf,
                "missed": 0,
                "hits": 1
            }

            detection["track_id"] = track_id

                                                  
        expired_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed"] > self.max_missed
        ]

        for track_id in expired_ids:
            del self.tracks[track_id]

                                                    
                             
                                                    

        print("\nTRACK ASSOCIATION STATISTICS")

        print(
            "Matched existing tracks:",
            len(matched_tracks)
        )

        print(
            "New tracks created:",
            len(detections) - len(matched_detections)
        )

        print(
            "Tracks removed:",
            len(expired_ids)
        )

        print(
            "Active tracks:",
            len(self.tracks)
        )

        confirmed_tracks = sum(
            track["hits"] >= 3
            for track in self.tracks.values()
        )

        print(
            "Tracks with at least 3 hits:",
            confirmed_tracks
        )

        return self.tracks


def process_scene_sequence(number_of_scans=5, scene_index=0):

    print("\n" + "=" * 55)
    print("PROCESSING CONSECUTIVE SCANS")
    print("=" * 55)

                                                                    
    scene = nusc.scene[scene_index]

                                         
    sample_token = scene["first_sample_token"]

    previous_timestamp = None
    sequence_results = []

                                         
    tracker = MultiObjectTracker(
        max_distance=3.0,
        max_missed=2
    )

    for scan_index in range(number_of_scans):

        if not sample_token:
            break

        sample = nusc.get("sample", sample_token)

                                                 
        timestamp = sample["timestamp"] / 1_000_000.0

        if previous_timestamp is None:
            dt = 0.0
        else:
            dt = timestamp - previous_timestamp

        print("\n" + "=" * 55)
        print(f"SCAN {scan_index + 1}")
        print(f"Time difference: {dt:.3f} seconds")
        print("=" * 55)

                                    
        result = process_scan(
            sample,
            visualize=False,
            random_seed=1700 + scene_index * 100 + scan_index
        )

                                                     
        result["detections"] = convert_detections_to_global(
            sample,
            result["detections"]
        )

                                                
        tracks = tracker.update(
            result["detections"],
            dt
        )

        print("\nTRACKING RESULTS")
        print("Active tracks:", len(tracks))

                                              
        for detection in result["detections"]:

            if not detection["radar_available"]:
                continue

            track_id = detection["track_id"]

            track = tracks[track_id]

            state = track["filter"].get_state()

            print(
                f"Track {track_id} | "
                f"Position: {np.round(state['position'], 2)} | "
                f"Velocity: {np.round(state['velocity'], 2)} | "
                f"Hits: {track['hits']}"
            )

                                   
        sequence_results.append({
            "sample": sample,
            "timestamp": timestamp,
            "dt": dt,
            "detections": result["detections"],
            "radar_results": result["radar_results"],
            "active_track_count": len(tracks),
            "track_states": {
                track_id: {
                    "position": track["filter"].x[:2].copy(),
                    "velocity": track["filter"].x[2:].copy(),
                    "hits": track["hits"]
                }
                for track_id, track in tracks.items()
            }
        })

        print(
            "\nDetected objects:",
            result["clusters"]
        )

        print(
            "RADAR-associated objects:",
            result["associated_detections"]
        )

        previous_timestamp = timestamp

                                                   
        sample_token = sample["next"]

    print("\n" + "=" * 55)
    print("Sequence processing completed.")
    print("Scans processed:", len(sequence_results))
    print("Total track IDs created:", tracker.next_id - 1)
    print("Tracks still active:", len(tracker.tracks))

    return sequence_results


def calculate_bev_iou(box_a, box_b):
       
                                                        

                      
                                
       

    ix_min = max(box_a[0], box_b[0])
    iy_min = max(box_a[1], box_b[1])
    ix_max = min(box_a[2], box_b[2])
    iy_max = min(box_a[3], box_b[3])

    intersection_width = max(0.0, ix_max - ix_min)
    intersection_height = max(0.0, iy_max - iy_min)

    intersection = intersection_width * intersection_height

    area_a = max(0.0, box_a[2] - box_a[0]) * \
             max(0.0, box_a[3] - box_a[1])

    area_b = max(0.0, box_b[2] - box_b[0]) * \
             max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def visualize_ground_truth_comparison(
    scan_result,
    max_range=25.0
):

    sample = scan_result["sample"]
    detections = scan_result["detections"]
    radar_results = scan_result["radar_results"]

                                                
                           
                                                

    lidar_token = sample["data"]["LIDAR_TOP"]

    lidar_data = nusc.get(
        "sample_data",
        lidar_token
    )

    ego_pose = nusc.get(
        "ego_pose",
        lidar_data["ego_pose_token"]
    )

    rotation = Quaternion(
        ego_pose["rotation"]
    ).rotation_matrix

    translation = np.asarray(
        ego_pose["translation"],
        dtype=float
    )

                                                
                      
                                                

    fig, ax = plt.subplots(figsize=(11, 10))

    fig.patch.set_facecolor("#101820")
    ax.set_facecolor("#17212B")

    gt_color = "#39D6AA"
    pred_color = "#FFB45B"
    radar_color = "#F46A70"

                                                
                                
                                                

    gt_count = 0

    for annotation_token in sample["anns"]:

        annotation = nusc.get(
            "sample_annotation",
            annotation_token
        )

        category = annotation["category_name"]

        if not (
            category.startswith("vehicle.")
            or category.startswith("human.pedestrian.")
        ):
            continue

                                                
        gt_box = nusc.get_box(annotation_token)

                                  
        corners_global = gt_box.corners()

                                                     
        corners_ego = rotation.T @ (
            corners_global - translation[:, None]
        )

        center_ego = rotation.T @ (
            np.asarray(annotation["translation"]) -
            translation
        )

        if np.linalg.norm(center_ego[:2]) > max_range:
            continue

       
                                                          
                                                     
                                                 

        footprint = corners_ego[:2, [2, 3, 7, 6]]

        footprint_center = np.mean(
            footprint,
            axis=1
        )

        angles = np.arctan2(
            footprint[1] - footprint_center[1],
            footprint[0] - footprint_center[0]
        )

        corner_order = np.argsort(angles)

        footprint = footprint[:, corner_order]

                             
        footprint = np.column_stack((
            footprint,
            footprint[:, 0]
        ))


        ax.plot(
            footprint[0],
            footprint[1],
            color=gt_color,
            linewidth=2.2,
            alpha=0.95
        )

                       
        ax.scatter(
            center_ego[0],
            center_ego[1],
            color=gt_color,
            s=20,
            zorder=5
        )

                           
        short_name = category.split(".")[-1]

        ax.text(
            center_ego[0],
            center_ego[1] + 0.8,
            short_name,
            color=gt_color,
            fontsize=8,
            ha="center"
        )

        gt_count += 1

                                                
                                   
                                                

    prediction_count = 0

    for detection in detections:

        center = detection["center"]

        if np.linalg.norm(center[:2]) > max_range:
            continue

        bbox = detection["bounding_box"]

        minimum = bbox.get_min_bound()
        maximum = bbox.get_max_bound()

        width = maximum[0] - minimum[0]
        height = maximum[1] - minimum[1]

        rectangle = Rectangle(
            (minimum[0], minimum[1]),
            width,
            height,
            fill=False,
            edgecolor=pred_color,
            linewidth=1.8,
            linestyle="--"
        )

        ax.add_patch(rectangle)

                    
        ax.text(
            center[0],
            center[1] - 0.8,
            f"C{detection['label']}",
            color=pred_color,
            fontsize=8,
            ha="center"
        )

        prediction_count += 1

                                                
                          
                                                

    radar_positions = radar_results["positions"]

    if len(radar_positions) > 0:

        radar_mask = (
            np.linalg.norm(
                radar_positions[:, :2],
                axis=1
            ) <= max_range
        )

        radar_xy = radar_positions[radar_mask, :2]

        if len(radar_xy) > 0:
            ax.scatter(
                radar_xy[:, 0],
                radar_xy[:, 1],
                color=radar_color,
                s=16,
                marker="x",
                alpha=0.8,
                zorder=4
            )

                                                
                    
                                                

    ax.scatter(
        0,
        0,
        color="#5B9FDF",
        s=140,
        marker="^",
        zorder=10
    )

    ax.text(
        0,
        -1.5,
        "EGO",
        color="#A9D2FF",
        fontsize=10,
        ha="center",
        weight="bold"
    )

                                                
                      
                                                

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    ax.set_aspect("equal")

    ax.set_xlabel(
        "X - Forward (m)",
        color="white"
    )

    ax.set_ylabel(
        "Y - Left (m)",
        color="white"
    )

    ax.set_title(
        "WINDOW 5 - GROUND TRUTH VS LIDAR DETECTIONS",
        color="white",
        fontsize=14,
        weight="bold",
        pad=18
    )

    ax.grid(
        True,
        color="white",
        alpha=0.12,
        linestyle="--"
    )

    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("#657484")

            
    ax.plot(
        [], [],
        color=gt_color,
        linewidth=2.2,
        label="Ground Truth"
    )

    ax.plot(
        [], [],
        color=pred_color,
        linestyle="--",
        linewidth=1.8,
        label="Predicted LiDAR"
    )

    ax.scatter(
        [], [],
        color=radar_color,
        marker="x",
        label="RADAR"
    )

    ax.legend(
        facecolor="#263441",
        edgecolor="#657484",
        labelcolor="white",
        loc="upper right"
    )

                                 
    info = (
        f"GT objects: {gt_count}\n"
        f"Predictions: {prediction_count}"
    )

    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        va="top",
        bbox={
            "facecolor": "#263441",
            "edgecolor": "#657484",
            "alpha": 0.9
        }
    )

    print("\nWINDOW 5 SUMMARY")
    print("Ground-truth objects displayed:", gt_count)
    print("Predicted boxes displayed:", prediction_count)

    plt.tight_layout()
    plt.show()


def evaluate_sequence_detections(
    sequence_results,
    max_range=25.0,
    iou_threshold=0.1
):
       
                                              

              
              
                 

                                  
       

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_matched_ious = []

    print("\n" + "=" * 55)
    print("GROUND-TRUTH DETECTION EVALUATION")
    print("=" * 55)

    for scan_index, scan in enumerate(sequence_results):

        sample = scan["sample"]
        detections = scan["detections"]

                            
        lidar_token = sample["data"]["LIDAR_TOP"]

        lidar_data = nusc.get(
            "sample_data",
            lidar_token
        )

        ego_pose = nusc.get(
            "ego_pose",
            lidar_data["ego_pose_token"]
        )

        rotation = Quaternion(
            ego_pose["rotation"]
        ).rotation_matrix

        translation = np.asarray(
            ego_pose["translation"],
            dtype=float
        )

                                                    
                                     
                                                    

        gt_boxes = []

        for annotation_token in sample["anns"]:

            annotation = nusc.get(
                "sample_annotation",
                annotation_token
            )

            category = annotation["category_name"]

            if not (
                category.startswith("vehicle.")
                or category.startswith("human.pedestrian.")
            ):
                continue

                                                    
            center_global = np.asarray(
                annotation["translation"],
                dtype=float
            )

            center_ego = (
                rotation.T @ (center_global - translation)
            )

            if np.linalg.norm(center_ego[:2]) > max_range:
                continue

                                                       
            gt_box = nusc.get_box(annotation_token)

            corners_global = gt_box.corners()

                                                        
            corners_ego = rotation.T @ (
                corners_global - translation[:, None]
            )

                                          
            min_xy = corners_ego[:2].min(axis=1)
            max_xy = corners_ego[:2].max(axis=1)

            gt_boxes.append({
                "category": category,
                "box": np.array([
                    min_xy[0],
                    min_xy[1],
                    max_xy[0],
                    max_xy[1]
                ])
            })

                                                    
                                        
                                                    

        predicted_boxes = []

        for detection in detections:

            center = detection["center"]

            if np.linalg.norm(center[:2]) > max_range:
                continue

            bbox = detection["bounding_box"]

            minimum = bbox.get_min_bound()
            maximum = bbox.get_max_bound()

            predicted_boxes.append(np.array([
                minimum[0],
                minimum[1],
                maximum[0],
                maximum[1]
            ]))

                                                    
                                 
                                                    

        candidates = []

        for pred_idx, predicted in enumerate(predicted_boxes):

            for gt_idx, ground_truth in enumerate(gt_boxes):

                iou = calculate_bev_iou(
                    predicted,
                    ground_truth["box"]
                )

                if iou >= iou_threshold:
                    candidates.append((
                        iou,
                        pred_idx,
                        gt_idx
                    ))

                                     
        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        matched_predictions = set()
        matched_ground_truth = set()

        for iou, pred_idx, gt_idx in candidates:

            if pred_idx in matched_predictions:
                continue

            if gt_idx in matched_ground_truth:
                continue

            matched_predictions.add(pred_idx)
            matched_ground_truth.add(gt_idx)

            all_matched_ious.append(iou)

        tp = len(matched_predictions)
        fp = len(predicted_boxes) - tp
        fn = len(gt_boxes) - len(matched_ground_truth)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"\nSCAN {scan_index + 1}")
        print("Ground-truth objects:", len(gt_boxes))
        print("Predicted detections:", len(predicted_boxes))
        print("True positives:", tp)
        print("False positives:", fp)
        print("False negatives:", fn)

                                                
                     
                                                

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0 else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0 else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    mean_iou = (
        float(np.mean(all_matched_ious))
        if all_matched_ious else 0.0
    )

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)

    print("Total true positives:", total_tp)
    print("Total false positives:", total_fp)
    print("Total false negatives:", total_fn)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1_score:.3f}")
    print(f"Mean matched BEV IoU: {mean_iou:.3f}")

    return {
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_bev_iou": mean_iou
    }


def build_frames(backend, scene_index, max_scans, messages, stop):
    scene = backend.nusc.scene[scene_index]
    total = scene['nbr_samples'] if max_scans <= 0 else min(max_scans, scene['nbr_samples'])
    signature = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    cache = ROOT/'.phase2_cache'/f"{scene['name']}-{signature}-{total}"
    manifest_path = cache/'frames.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if len(manifest) == total and all((cache/f'{i:04d}.npz').exists() for i in range(total)):
            frames=[]
            for i,meta in enumerate(manifest):
                if stop.is_set(): return
                with np.load(cache/f'{i:04d}.npz', allow_pickle=False) as data:
                    frames.append({**meta,'points':data['points'].copy(),'radar':data['radar'].copy()})
            messages.put(('ready',frames,scene['name']))
            return
    cache.mkdir(parents=True,exist_ok=True)
    tracker=backend.MultiObjectTracker(max_distance=3.,max_missed=2)
    token=scene['first_sample_token'];previous=None;first_time=None
    frames=[];manifest=[];history={}
    with (ROOT/'phase2_processing.log').open('w') as logfile, contextlib.redirect_stdout(logfile):
        for i in range(total):
            if stop.is_set(): return
            sample=backend.nusc.get('sample',token)
            result=backend.process_scan(sample,visualize=False,random_seed=1700+scene_index*100+i)
            detections=backend.convert_detections_to_global(sample,result['detections'])
            lidar_data=backend.nusc.get('sample_data',sample['data']['LIDAR_TOP'])
            timestamp=lidar_data['timestamp']/1_000_000.
            if first_time is None: first_time=timestamp
            dt=0. if previous is None else timestamp-previous
            tracks=tracker.update(detections,dt)
            xyz,_=backend.load_lidar_scan(sample)
            pose=backend.nusc.get('ego_pose',lidar_data['ego_pose_token'])
            rotation=Quaternion(pose['rotation']).rotation_matrix
            translation=np.asarray(pose['translation'])
            boxes=[]
            for d in detections:
                tid=int(d['track_id']); state=tracks[tid]['filter'].get_state()
                boxes.append({'track_id':tid,'label':int(d['label']),
                    'low':d['bounding_box'].get_min_bound().tolist(),
                    'high':d['bounding_box'].get_max_bound().tolist(),
                    'center':d['center'].tolist(),
                    'radar':bool(d['radar_available']),
                    'radar_returns':int(d['radar_points']),
                    'velocity_global':state['velocity'].tolist(),
                    'hits':int(tracks[tid]['hits'])})
                history.setdefault(tid,[]).append(np.r_[state['position'],d['center_global'][2]].tolist())
            trails={str(tid):positions[-12:] for tid,positions in history.items() if tid in tracks}
            predictions=[]
            observed={d['track_id'] for d in boxes}
            for tid,track in tracks.items():
                if tid in observed: continue
                z=history.get(tid,[[0,0,translation[2]+.4]])[-1][2]
                predictions.append({'track_id':int(tid),'position_global':[*track['filter'].x[:2].tolist(),z],
                                    'missed':int(track['missed'])})
            references=[]
            for annotation_token in sample['anns']:
                ann=backend.nusc.get('sample_annotation',annotation_token)
                if not (ann['category_name'].startswith('vehicle.') or ann['category_name'].startswith('human.pedestrian.')): continue
                box=backend.nusc.get_box(annotation_token)
                corners=np.einsum('ij,jk->ik',rotation.T,box.corners()-translation[:,None]).T
                references.append({'corners':corners.tolist(),'category':ann['category_name']})
            meta={'index':i,'token':token,'time':timestamp-first_time,'dt':dt,
                  'boxes':boxes,'tracks':len(tracks),'trails':trails,'predictions':predictions,
                  'references':references,'rotation':rotation.tolist(),'translation':translation.tolist(),
                  'radar_associated':sum(b['radar'] for b in boxes)}
            radar=np.asarray(result['radar_results']['positions'],dtype=np.float32)
            points=np.asarray(xyz,dtype=np.float32)
            np.savez_compressed(cache/f'{i:04d}.npz',points=points,radar=radar)
            frames.append({**meta,'points':points,'radar':radar});manifest.append(meta)
            messages.put(('progress',i+1,total))
            previous=timestamp;token=sample['next']
    manifest_path.write_text(json.dumps(manifest,separators=(',',':'),allow_nan=False))
    messages.put(('ready',frames,scene['name']))


def scene_payload(index):
    with SCENE_LOCK:
        if index in PAYLOADS:return PAYLOADS[index]
        p=backend()
        if not 0<=index<len(p.nusc.scene):raise ValueError('Unknown scene index')
        messages=queue.Queue();build_frames(p,index,0,messages,threading.Event())
        ready=None
        while not messages.empty():
            message=messages.get()
            if message[0]=='ready':ready=message
        if ready is None:raise RuntimeError('Scene processing did not produce frames')
        _,frames,name=ready
        arrays=[];metadata=[];offset=0
        for frame in frames:
            points=frame['points'];mask=(np.linalg.norm(points[:,:2],axis=1)<=75.)&(points[:,2]>-2.)&(points[:,2]<9.)
            points=np.asarray(points[mask],dtype='<f4')
            radar=np.asarray(frame['radar'],dtype='<f4')
            meta={k:v for k,v in frame.items() if k not in ('points','radar')}
            meta.update({'point_offset':offset,'point_count':len(points)})
            arrays.append(points.ravel());offset+=points.size
            meta.update({'radar_offset':offset,'radar_count':len(radar)})
            arrays.append(radar.ravel());offset+=radar.size
            metadata.append(meta)
        raw=np.concatenate(arrays).astype('<f4').tobytes()
        payload={'version':1,'scene_index':index,'scene_name':name,
                 'scenes':[s['name'] for s in p.nusc.scene],
                 'frames':metadata,'point_data_gzip_base64':base64.b64encode(gzip.compress(raw,compresslevel=6)).decode('ascii')}
        PAYLOADS[index]=payload
        return payload


def html_document(payload=None):
    template=HTML_TEMPLATE
    data='null' if payload is None else json.dumps({**payload,'standalone':True},separators=(',',':'),allow_nan=False).replace('</','<\\/')
    return template.replace('/*__SCENE_DATA__*/null',data)


def export_scene(index=0, output_path=None):
    payload=scene_payload(index)
    path=Path(output_path) if output_path is not None else ROOT/'phase2_viewer.html'
    path.write_text(html_document(payload))
    print('VIEWER_HTML',path,flush=True)
    return path


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=urllib.parse.urlparse(self.path).path
        try:
            if path in ('/','/viewer.html'):
                data=html_document().encode();mime='text/html; charset=utf-8';compressed=False
            elif path.startswith('/api/scene/'):
                index=int(path.rsplit('/',1)[1])
                data=gzip.compress(json.dumps(scene_payload(index),separators=(',',':'),allow_nan=False).encode(),compresslevel=3)
                mime='application/json';compressed=True
            elif path=='/health':data=b'ok';mime='text/plain';compressed=False
            elif path=='/favicon.ico':self.send_response(204);self.end_headers();return
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime)
            if compressed:self.send_header('Content-Encoding','gzip')
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as error:
            import traceback
            (ROOT/'viewer_error.log').write_text(traceback.format_exc())
            data=json.dumps({'error':str(error)}).encode()
            self.send_response(500);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)


def assign_centers(track_positions, target_positions, gate=2.0):
    tracks=np.asarray(track_positions,dtype=float).reshape(-1,2)
    targets=np.asarray(target_positions,dtype=float).reshape(-1,2)
    n,m=len(tracks),len(targets)
    if not n or not m:return []
    delta=tracks[:,None,:]-targets[None,:,:]
    distances=np.sqrt(np.sum(delta*delta,axis=2))
                                                                           
                                                                           
    penalty=(n+m+1)*(gate+1)
    costs=np.full((n,m+n),penalty,dtype=float)
    costs[:,:m]=np.where(distances<=gate,distances,penalty*(n+m+1))
    rows,cols=linear_sum_assignment(costs)
    return [(int(i),int(j),float(distances[i,j])) for i,j in zip(rows,cols)
            if j<m and distances[i,j]<=gate]


def summarize_tracking(stats):
    gt=stats['gt'];tp=stats['tp'];fp=stats['fp']
    errors=stats.pop('_position_errors')
    velocities=stats.pop('_velocity_errors')
    stats['precision']=tp/(tp+fp) if tp+fp else 0.
    stats['recall']=tp/gt if gt else 0.
    stats['f1']=2*tp/(2*tp+fp+stats['fn']) if gt+tp+fp else 0.
    stats['mean_position_error_m']=float(np.mean(errors)) if errors else None
    stats['median_position_error_m']=float(np.median(errors)) if errors else None
    stats['velocity_rmse_mps']=float(np.sqrt(np.mean(np.square(velocities)))) if velocities else None
    stats['finite_velocity_matches']=len(velocities)
    stats['consecutive_id_consistency']=1-stats['consecutive_id_changes']/stats['consecutive_matched_pairs'] if stats['consecutive_matched_pairs'] else None
    return stats


def tracking_stats():
    return {'frames':0,'gt':0,'tp':0,'fp':0,'fn':0,'eligible_gt_instances':0,
            'matched_gt_instances':0,'created_track_ids':0,'observed_id_changes':0,
            'consecutive_id_changes':0,'consecutive_matched_pairs':0,
            'reacquisitions_after_missed_eligible_frames':0,'_position_errors':[],
            '_velocity_errors':[],'by_class':{'vehicle':{'gt':0,'tp':0},'pedestrian':{'gt':0,'tp':0}}}


def evaluate_tracker_replay(stream,tracker_type=MultiObjectTracker):
    total=tracking_stats();by_scene={};snapshots=[]
    scenes=list(dict.fromkeys(row['scene'] for row in stream))
    for scene in scenes:
        tracker=tracker_type(max_distance=3.,max_missed=2)
        stats=tracking_stats();history={};previous_matches={};instances=set();matched_instances=set()
        for frame in (row for row in stream if row['scene']==scene):
            detections=[{'center_global':np.asarray(d['center_global'],dtype=float),
                         'radar_available':d['radar_available'],
                         'radar_velocity_global':np.asarray(d['radar_velocity_global'],dtype=float) if d['radar_velocity_global'] is not None else None} for d in frame['detections']]
            with contextlib.redirect_stdout(io.StringIO()):tracks=tracker.update(detections,frame['dt'])
            ego=np.asarray(frame['ego_xy']);gt=frame['ground_truth'];scan=frame['scan_index']
            tids=[tid for tid,t in tracks.items() if np.linalg.norm(t['filter'].x[:2]-ego)<=25.]
            pairs=assign_centers([tracks[tid]['filter'].x[:2] for tid in tids],[g['position'] for g in gt])
            stats['frames']+=1;stats['gt']+=len(gt);stats['tp']+=len(pairs)
            stats['fp']+=len(tids)-len(pairs);stats['fn']+=len(gt)-len(pairs)
            current={}
            for g in gt:
                instances.add(g['instance']);cls='vehicle' if g['category'].startswith('vehicle.') else 'pedestrian'
                stats['by_class'][cls]['gt']+=1
            for i,j,error in pairs:
                tid=tids[i];g=gt[j];inst=g['instance'];current[inst]=tid;matched_instances.add(inst)
                cls='vehicle' if g['category'].startswith('vehicle.') else 'pedestrian'
                stats['by_class'][cls]['tp']+=1;stats['_position_errors'].append(error)
                if g['velocity'] is not None:
                    stats['_velocity_errors'].append(float(np.linalg.norm(tracks[tid]['filter'].x[2:]-np.asarray(g['velocity']))))
                if inst in history:
                    old_id,last_frame,stayed_eligible=history[inst]
                    stats['observed_id_changes']+=int(old_id!=tid)
                    if last_frame<scan-1 and stayed_eligible:stats['reacquisitions_after_missed_eligible_frames']+=1
                if inst in previous_matches:
                    stats['consecutive_matched_pairs']+=1
                    stats['consecutive_id_changes']+=int(previous_matches[inst]!=tid)
                history[inst]=(tid,scan,True)
            eligible_now={g['instance'] for g in gt}
            for inst,(tid,last,stayed) in list(history.items()):
                if inst not in eligible_now:history[inst]=(tid,last,False)
            previous_matches=current
            snapshots.append({'scene':scene,'scan_index':scan,'pairs':len(pairs),'active_in_range':len(tids),
                              'assignments':[[g,tid] for g,tid in sorted(current.items())]})
        stats['eligible_gt_instances']=len(instances);stats['matched_gt_instances']=len(matched_instances);stats['created_track_ids']=tracker.next_id-1
        for key in total:
            if key=='by_class':
                for cls in total[key]:
                    for field in total[key][cls]:total[key][cls][field]+=stats[key][cls][field]
            elif key.startswith('_'):total[key].extend(stats[key])
            else:total[key]+=stats[key]
        by_scene[scene]=summarize_tracking(stats)
    return {'total':summarize_tracking(total),'by_scene':by_scene,'frames':snapshots}


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LiDAR + RADAR · 3D Playback</title>
<style>
:root{color-scheme:dark;--bg:#070c16;--panel:#0d1524;--muted:#8192aa;--line:#223148;--green:#8bee75;--text:#e5edf8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}button,select,input{font:inherit}button,select{background:#162236;border:1px solid #2c3e58;border-radius:7px;color:var(--text);padding:9px 12px;cursor:pointer}button:hover{background:#23334b}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #9feb80;outline-offset:3px}button:disabled{opacity:.45;cursor:default}header{height:70px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 24px;background:#0b1220;gap:20px}h1{font-size:19px;margin:0;letter-spacing:.04em;font-weight:650}header small{font-size:11px;color:var(--muted);display:block;margin-top:5px;letter-spacing:.12em}#dataset{color:#9eaec3;font-size:12px;text-align:right}main{height:calc(100dvh - 70px);display:grid;grid-template-columns:252px minmax(0,1fr)}aside{background:var(--panel);border-right:1px solid var(--line);padding:22px 18px;overflow:auto}h2{font-size:10px;font-weight:650;letter-spacing:.14em;color:#91a4bf;margin:24px 0 12px}h2:first-child{margin-top:0}aside select{width:100%;font-size:13px}.check{display:flex;align-items:center;gap:9px;padding:7px 0;color:#bcc9da;font-size:13px;cursor:pointer}.check input{accent-color:#8bee75;width:15px;height:15px;margin:0}.sub{color:var(--muted);font-size:11px;line-height:1.65}input[type=range]{accent-color:#9ce781;width:100%;cursor:pointer}.row{display:flex;align-items:center;justify-content:space-between;gap:8px}.views{display:flex;gap:6px}.views button{flex:1;font-size:12px;padding:8px 4px}.legend{font-size:11px;line-height:1.9;color:#9badc3}.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:8px}.stats{border-top:1px solid var(--line);margin-top:19px;padding-top:16px;display:grid;gap:10px}.stat{display:flex;justify-content:space-between;font-size:11px;color:#8fa1ba}.stat strong{font-size:13px;color:#d7e4f4;font-variant-numeric:tabular-nums}#stage{position:relative;overflow:hidden;background:#030713;min-width:0}#gl,#labels{position:absolute;width:100%;height:100%;top:0;left:0;display:block}#labels{pointer-events:none}#gl{touch-action:none;cursor:grab}#gl:active{cursor:grabbing}.hud{position:absolute;top:20px;left:22px;pointer-events:none}.badge{color:#a8f08c;font-size:10px;letter-spacing:.14em;display:flex;align-items:center;gap:7px}.badge:before{content:"";width:6px;height:6px;border-radius:50%;background:#9ff17b}#sceneTitle{font-size:18px;margin-top:8px;letter-spacing:.03em}.hint{position:absolute;top:20px;right:22px;color:#768aa5;font-size:11px;pointer-events:none;text-align:right;line-height:1.7}#detail{position:absolute;right:20px;top:78px;background:#101b2ddd;border:1px solid #33445d;padding:14px 16px;border-radius:8px;display:none;white-space:pre-line;font-size:12px;line-height:1.8;max-width:250px;pointer-events:none}.transport{position:absolute;bottom:20px;left:22px;right:22px;background:#0c1626ed;border:1px solid #26374e;border-radius:12px;padding:15px 18px;backdrop-filter:blur(10px)}.timeline{display:flex;align-items:center;gap:15px}#timeline{flex:1;width:auto}#frameText{font-size:11px;color:#c3d2e5;font-variant-numeric:tabular-nums;min-width:105px;text-align:right}.transport-bottom{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:12px}.buttons{display:flex;gap:7px;align-items:center}#play{background:#a0e77b;color:#102014;border-color:#a0e77b;font-weight:650;min-width:76px}.speed{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--muted)}.speed select{padding:7px 8px;font-size:12px}.status{font-size:10px;color:#7e94af}.load{position:absolute;inset:0;background:#060c18ed;display:flex;align-items:center;justify-content:center;flex-direction:column;z-index:5;gap:16px;color:#a9bbd2;text-align:center;padding:30px}.spinner{border:2px solid #25344b;border-top-color:#a0e77b;width:34px;height:34px;border-radius:50%;animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.load strong{color:#d8e7f6;font-size:17px}.load small{font-size:12px;line-height:1.8;max-width:440px}.footer-note{margin-top:16px;font-size:10px;color:#71859f;line-height:1.65}#save{width:100%;font-size:12px;margin-top:12px}@media(max-width:850px){main{grid-template-columns:205px minmax(0,1fr)}aside{padding:18px 12px}.hint{display:none}header{padding:0 15px}#dataset{max-width:250px}}@media(max-width:650px){main{grid-template-columns:160px minmax(0,1fr)}h1{font-size:16px}.status{display:none}.transport{left:10px;right:10px;padding:10px}.buttons button{padding:7px}.speed{gap:3px}aside{font-size:11px}.check{font-size:11px}.hud{left:12px}.hint{display:none}}
</style></head><body>
<header><div><h1>LiDAR + RADAR</h1><small>3D SENSOR PLAYBACK</small></div><div id="dataset">nuScenes Mini · real recorded sensor data</div></header>
<main><aside><h2>SCENE</h2><select id="scene" aria-label="Scene"></select><div id="sceneNote" class="sub" style="margin-top:9px">Python detections and tracked identities</div>
<h2>VISIBLE LAYERS</h2>
<label class="check"><input id="boxes" type="checkbox" checked>Detection boxes</label><label class="check"><input id="radar" type="checkbox" checked>Radar returns</label><label class="check"><input id="ids" type="checkbox" checked>Track IDs and speed</label><label class="check"><input id="trails" type="checkbox" checked>Track trails</label><label class="check"><input id="velocity" type="checkbox" checked>Velocity arrows</label><label class="check"><input id="predictions" type="checkbox">Unobserved predictions</label><label class="check"><input id="groundtruth" type="checkbox">Ground truth reference</label>
<h2>DISPLAY</h2><div class="row sub"><span>Point size</span><span id="sizeValue">2.0</span></div><input id="pointsize" aria-label="Point size" type="range" min="1" max="4" step=".25" value="2"><div class="row sub" style="margin-top:10px"><span>Range</span><span id="rangeValue">50 m</span></div><input id="range" aria-label="Display range" type="range" min="20" max="70" step="5" value="50">
<div class="views" style="margin-top:14px"><button id="view3d">3D view</button><button id="viewTop">Top view</button></div><button id="save">Save scene image</button>
<h2>LEGEND</h2><div class="legend"><div><span class="dot" style="background:#ffbd15"></span>Detection without radar</div><div><span class="dot" style="background:#37ff87"></span>Radar-associated detection</div><div><span class="dot" style="background:#ff3da4"></span>Radar return</div><div><span class="dot" style="background:#e0ecff"></span>Ground truth when enabled</div></div><div class="footer-note">Point color indicates distance.<br>Speed is a Kalman estimate.<br>Arrows show 1 s, capped at 8 m.<br>The ego vehicle is an illustrative marker.</div>
<div class="stats"><div class="stat"><span>Visible LiDAR points</span><strong id="pointCount">—</strong></div><div class="stat"><span>In-range detections</span><strong id="boxCount">—</strong></div><div class="stat"><span>Active tracks</span><strong id="trackCount">—</strong></div><div class="stat"><span>Radar-associated</span><strong id="radarCount">—</strong></div></div>
</aside><section id="stage" aria-label="Interactive 3D LiDAR scene"><canvas id="gl"></canvas><canvas id="labels"></canvas><div class="hud"><div class="badge">RECORDED SCAN</div><div id="sceneTitle">Preparing scene</div></div><div class="hint">Drag to orbit · scroll to zoom<br>Shift + drag to pan · click a track label<br>Space to play · ← → to step · R to reset</div><div id="detail"></div>
<div class="transport"><div class="timeline"><input id="timeline" aria-label="Scan timeline" type="range" min="0" max="1" step="1" value="0"><span id="frameText">0 / 0</span></div><div class="transport-bottom"><div class="buttons"><button id="prev" aria-label="Previous scan">←</button><button id="play">Play</button><button id="next" aria-label="Next scan">→</button><label class="check" style="font-size:11px;margin-left:6px"><input id="loop" type="checkbox" checked>Loop</label></div><div class="speed"><span>Speed</span><select id="speed" aria-label="Playback speed"><option value=".5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></div><span class="status" id="stateText">Paused · actual detections</span></div></div>
<div id="loading" class="load"><div class="spinner"></div><strong id="loadTitle">Preparing your 3D scene</strong><small id="loadMessage">Loading the recorded LiDAR points and the actual Python detection results.</small></div>
</section></main>
<script id="embedded-data" type="application/json">/*__SCENE_DATA__*/null</script>
<script>
'use strict';
const $=id=>document.getElementById(id), canvas=$('gl'),overlay=$('labels'),ctx=overlay.getContext('2d');
let gl,scene=null,index=0,playing=false,lastAdvance=0,dirty=true,selected=null,labelTargets=[],errorMessage=null;
const state={yaw:-2.30,pitch:.70,distance:52,target:[4,0,.2],top:false};
const toggles=['boxes','radar','ids','trails','velocity','predictions','groundtruth'];
const V={add:(a,b)=>a.map((v,i)=>v+b[i]),sub:(a,b)=>a.map((v,i)=>v-b[i]),mul:(a,s)=>a.map(v=>v*s),dot:(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross:(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm:a=>Math.hypot(...a)};
V.unit=a=>V.mul(a,1/(V.norm(a)||1));
function multiply(a,b){let c=new Float32Array(16);for(let col=0;col<4;col++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)c[col*4+row]+=a[k*4+row]*b[col*4+k];return c;}
function perspective(fov,aspect,n,f){const t=1/Math.tan(fov/2);return new Float32Array([t/aspect,0,0,0,0,t,0,0,0,0,(f+n)/(n-f),-1,0,0,2*f*n/(n-f),0]);}
function lookAt(eye,target,up){const z=V.unit(V.sub(eye,target)),x=V.unit(V.cross(up,z)),y=V.cross(z,x);return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-V.dot(x,eye),-V.dot(y,eye),-V.dot(z,eye),1]);}
function transform(m,p){return [m[0]*p[0]+m[4]*p[1]+m[8]*p[2]+m[12],m[1]*p[0]+m[5]*p[1]+m[9]*p[2]+m[13],m[2]*p[0]+m[6]*p[1]+m[10]*p[2]+m[14],m[3]*p[0]+m[7]*p[1]+m[11]*p[2]+m[15]];}
function toEgo(p,f){const d=V.sub(p,f.translation),r=f.rotation;return [r[0][0]*d[0]+r[1][0]*d[1]+r[2][0]*d[2],r[0][1]*d[0]+r[1][1]*d[1]+r[2][1]*d[2],r[0][2]*d[0]+r[1][2]*d[1]+r[2][2]*d[2]];}
function velocityEgo(v,f){const r=f.rotation;return [r[0][0]*v[0]+r[1][0]*v[1],r[0][1]*v[0]+r[1][1]*v[1],0];}
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
function program(v,f){const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,v));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,f));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));return p;}
let solidProgram,lineProgram,pointBuffer,radarBuffer,lineBuffer,referenceBuffer,egoBuffer,pointVertices=0,radarVertices=0,lineVertices=0,referenceVertices=0,egoVertices=0,geometryKey='';
function initGL(){gl=canvas.getContext('webgl2',{antialias:true,preserveDrawingBuffer:true,alpha:false});if(!gl)throw Error('WebGL 2 is unavailable. Open this file in a browser with hardware acceleration.');
solidProgram=program(`#version 300 es
layout(location=0) in vec3 aPosition;layout(location=1) in vec3 aColor;uniform mat4 uMatrix;uniform float uSize;out vec3 vColor;void main(){gl_Position=uMatrix*vec4(aPosition,1.);gl_PointSize=uSize;vColor=aColor;}`,`#version 300 es
precision highp float;in vec3 vColor;uniform int uPoint;out vec4 outColor;void main(){if(uPoint==1&&length(gl_PointCoord-vec2(.5))>.5)discard;outColor=vec4(vColor,1.);}`);
lineProgram=program(`#version 300 es
layout(location=0) in vec3 aPosition;layout(location=1) in vec3 aOther;layout(location=2) in float aSide;layout(location=3) in vec3 aColor;uniform mat4 uMatrix;uniform vec2 uResolution;uniform float uWidth;out vec3 vColor;void main(){vec4 a=uMatrix*vec4(aPosition,1.),b=uMatrix*vec4(aOther,1.);vec2 direction=(b.xy/b.w-a.xy/a.w)*uResolution;direction=normalize(direction+vec2(.00001));vec2 normal=vec2(-direction.y,direction.x);a.xy+=normal*aSide*uWidth/uResolution*a.w;gl_Position=a;vColor=aColor;}`,`#version 300 es
precision highp float;in vec3 vColor;out vec4 outColor;void main(){outColor=vec4(vColor,1.);}`);
pointBuffer=gl.createBuffer();radarBuffer=gl.createBuffer();lineBuffer=gl.createBuffer();referenceBuffer=gl.createBuffer();egoBuffer=gl.createBuffer();
gl.enable(gl.DEPTH_TEST);gl.clearColor(.012,.021,.045,1);buildEgo();}
function upload(buffer,data){gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);}
function line(out,a,b,color){if(V.norm(V.sub(a,b))<.0001)return;for(const [p,q,s]of[[a,b,-1],[a,b,1],[b,a,-1],[a,b,-1],[b,a,-1],[b,a,1]])out.push(...p,...q,s,...color);}
function makeBox(out,low,high,color){const [x,y,z]=low,[X,Y,Z]=high,p=[[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]];for(const[a,b]of[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])line(out,p[a],p[b],color);}
function buildEgo(){let a=[];const light=V.unit([-.4,-.7,1]);function triangle(p,q,r,color,normal){const brightness=.48+.52*Math.max(0,V.dot(normal,light));for(const v of[p,q,r])a.push(...v,...V.mul(color,brightness));}
function box(size,c,color){const lo=V.sub(c,V.mul(size,.5)),hi=V.add(c,V.mul(size,.5)),[x,y,z]=lo,[X,Y,Z]=hi,p=[[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]];for(const [ids,n]of[[[0,3,2,1],[0,0,-1]],[[4,5,6,7],[0,0,1]],[[0,1,5,4],[0,-1,0]],[[1,2,6,5],[1,0,0]],[[2,3,7,6],[0,1,0]],[[3,0,4,7],[-1,0,0]]]){triangle(p[ids[0]],p[ids[1]],p[ids[2]],color,n);triangle(p[ids[0]],p[ids[2]],p[ids[3]],color,n);}}
box([4.5,1.84,.66],[0,0,.76],[.30,.41,.46]);box([4.25,1.86,.12],[0,0,.46],[.08,.12,.15]);box([2.3,1.62,.73],[-.36,0,1.34],[.35,.43,.48]);box([2.08,1.66,.39],[-.36,0,1.47],[.025,.065,.10]);box([2.4,1.68,.07],[-.36,0,1.75],[.36,.43,.48]);box([1.05,1.73,.06],[1.47,0,1.10],[.42,.95,.12]);box([.08,1.42,.15],[2.25,0,.90],[.88,1.,.76]);box([.08,1.42,.14],[-2.25,0,.90],[.9,.04,.06]);
for(const x of[-1.43,1.43])for(const y of[-.94,.94]){const c=[x,y,.38],n=20;for(let j=0;j<n;j++){const t=j/n*2*Math.PI,T=(j+1)/n*2*Math.PI,pa=[x+.35*Math.cos(t),y-.12,.38+.35*Math.sin(t)],pb=[x+.35*Math.cos(T),y-.12,.38+.35*Math.sin(T)],pc=[pb[0],y+.12,pb[2]],pd=[pa[0],y+.12,pa[2]],normal=V.unit([Math.cos((t+T)/2),0,Math.sin((t+T)/2)]);triangle(pa,pb,pc,[.035,.04,.045],normal);triangle(pa,pc,pd,[.035,.04,.045],normal);triangle([x,y-.12,.38],pb,pa,[.2,.24,.27],[0,-1,0]);triangle([x,y+.12,.38],pd,pc,[.2,.24,.27],[0,1,0]);}}
upload(egoBuffer,a);egoVertices=a.length/6;}
function pointColor(distance,z,radius){let stops=[0,.16,.42,.70,1],colors=[[.65,1.,.12],[.12,1.,.12],[.02,.90,.48],[.03,.58,.92],[.12,.25,.80]],t=Math.min(1,distance/radius),i=0;while(i<3&&t>stops[i+1])i++;let u=Math.max(0,Math.min(1,(t-stops[i])/(stops[i+1]-stops[i]))),brightness=Math.max(.58,Math.min(1,.84+.05*z));return colors[i].map((v,k)=>(v*(1-u)+colors[i+1][k]*u)*brightness);}
function updateGeometry(){if(!scene)return;const f=scene.frames[index],radius=+$('range').value,key=[scene.scene_name,index,radius,...toggles.map(id=>$(id).checked),selected].join('|');if(key===geometryKey)return;geometryKey=key;let pv=[],rv=[],lv=[],refs=[];
for(let i=0;i<f.point_count;i++){const offset=f.point_offset+i*3,x=scene.points[offset],y=scene.points[offset+1],z=scene.points[offset+2],d=Math.hypot(x,y);if(d>radius||z>8||Math.abs(x)<2.35&&Math.abs(y)<1.05&&z<1.9)continue;pv.push(x,y,z,...pointColor(d,z,radius));}upload(pointBuffer,pv);pointVertices=pv.length/6;
if($('radar').checked)for(let i=0;i<f.radar_count;i++){const o=f.radar_offset+i*3,p=[scene.points[o],scene.points[o+1],scene.points[o+2]];if(Math.hypot(p[0],p[1])<=radius)rv.push(...p,1.,.24,.64);}upload(radarBuffer,rv);radarVertices=rv.length/6;
let visible=0;for(const box of f.boxes){if(Math.hypot(...box.center.slice(0,2))>radius)continue;visible++;const color=box.track_id===selected?[1,1,.95]:box.radar?[.2,1,.5]:[1,.73,.08];if($('boxes').checked)makeBox(lv,box.low,box.high,color);if($('velocity').checked){let v=velocityEgo(box.velocity_global,f),length=V.norm(v);if(length>.2){v=V.mul(v,Math.min(1,8/length));const end=V.add(box.center,v),direction=V.unit(v),side=[-direction[1],direction[0],0],h=Math.min(.6,Math.max(.22,V.norm(v)*.25));line(lv,box.center,end,color);line(lv,end,V.add(V.sub(end,V.mul(direction,h)),V.mul(side,h*.45)),color);line(lv,end,V.sub(V.sub(end,V.mul(direction,h)),V.mul(side,h*.45)),color);}}}
if($('trails').checked)for(const positions of Object.values(f.trails)){if(positions.length<2)continue;const trail=positions.map(p=>toEgo(p,f));if(Math.hypot(...trail.at(-1).slice(0,2))>radius)continue;for(let i=1;i<trail.length;i++)line(lv,trail[i-1],trail[i],[.12,.61,.82]);}
if($('predictions').checked)for(const prediction of f.predictions){const p=toEgo(prediction.position_global,f);if(Math.hypot(p[0],p[1])>radius)continue;line(lv,V.add(p,[-.3,0,0]),V.add(p,[.3,0,0]),[.6,.65,.75]);line(lv,V.add(p,[0,-.3,0]),V.add(p,[0,.3,0]),[.6,.65,.75]);}
if($('groundtruth').checked)for(const reference of f.references){const corners=reference.corners,center=corners.reduce((a,b)=>V.add(a,V.mul(b,.125)),[0,0,0]);if(Math.hypot(center[0],center[1])>radius)continue;for(const[a,b]of[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]){const vector=V.sub(corners[b],corners[a]),segments=Math.max(1,Math.floor(V.norm(vector)/.23));for(let j=0;j<segments;j+=2)line(refs,V.add(corners[a],V.mul(vector,j/segments)),V.add(corners[a],V.mul(vector,Math.min(j+1,segments)/segments)),[.82,.9,1]);}}
upload(lineBuffer,lv);lineVertices=lv.length/10;upload(referenceBuffer,refs);referenceVertices=refs.length/10;
$('pointCount').textContent=pointVertices.toLocaleString();$('boxCount').textContent=visible;$('trackCount').textContent=f.tracks;$('radarCount').textContent=f.radar_associated;$('frameText').textContent=`${String(index+1).padStart(2,'0')} / ${scene.frames.length}  ·  ${f.time.toFixed(2)} s`;$('timeline').value=index;updateDetail();}
function drawSolid(buffer,count,mode,matrix,size){if(!count)return;gl.useProgram(solidProgram);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,24,0);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.FLOAT,false,24,12);gl.uniformMatrix4fv(gl.getUniformLocation(solidProgram,'uMatrix'),false,matrix);gl.uniform1f(gl.getUniformLocation(solidProgram,'uSize'),size);gl.uniform1i(gl.getUniformLocation(solidProgram,'uPoint'),mode===gl.POINTS?1:0);gl.drawArrays(mode,0,count);}
function drawLines(buffer,count,matrix,width,dpr){if(!count)return;gl.useProgram(lineProgram);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);for(const[loc,size,offset]of[[0,3,0],[1,3,12],[2,1,24],[3,3,28]]){gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,40,offset);}gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram,'uMatrix'),false,matrix);gl.uniform2f(gl.getUniformLocation(lineProgram,'uResolution'),canvas.width,canvas.height);gl.uniform1f(gl.getUniformLocation(lineProgram,'uWidth'),width*dpr);gl.drawArrays(gl.TRIANGLES,0,count);gl.disableVertexAttribArray(2);gl.disableVertexAttribArray(3);}
function render(){if(!gl||!scene)return;const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height,dpr=Math.min(2,devicePixelRatio||1);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);overlay.width=canvas.width;overlay.height=canvas.height;}gl.viewport(0,0,canvas.width,canvas.height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);updateGeometry();const p=state.pitch,y=state.yaw,d=state.distance,eye=V.add(state.target,[d*Math.cos(p)*Math.cos(y),d*Math.cos(p)*Math.sin(y),d*Math.sin(p)]),view=lookAt(eye,state.target,[0,0,1]),matrix=multiply(perspective(.90,w/h,.1,300),view);drawSolid(pointBuffer,pointVertices,gl.POINTS,matrix,+$('pointsize').value*dpr);drawSolid(egoBuffer,egoVertices,gl.TRIANGLES,matrix,1);drawLines(lineBuffer,lineVertices,matrix,2.2,dpr);drawLines(referenceBuffer,referenceVertices,matrix,1.0,dpr);drawSolid(radarBuffer,radarVertices,gl.POINTS,matrix,6*dpr);
ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);labelTargets=[];const f=scene.frames[index],radius=+$('range').value;function label(p,text,color,box){const v=transform(matrix,p);if(v[3]<=0)return;const x=(v[0]/v[3]+1)*w/2,y=(1-v[1]/v[3])*h/2;if(x<0||x>w||y<0||y>h-120)return;ctx.font='500 11px -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif';ctx.textAlign='center';ctx.lineWidth=3.5;ctx.strokeStyle='#050a15';ctx.strokeText(text,x,y);ctx.fillStyle=color;ctx.fillText(text,x,y);if(box)labelTargets.push({x,y,box});}
if($('ids').checked)for(const box of f.boxes){if(Math.hypot(...box.center.slice(0,2))>radius)continue;label([box.center[0],box.center[1],box.high[2]+.35],`T${box.track_id}  ${Math.hypot(...box.velocity_global).toFixed(1)} m/s`,box.radar?'#75ffaa':'#ffe08b',box);}if($('predictions').checked&&$('ids').checked)for(const prediction of f.predictions){let p=toEgo(prediction.position_global,f);p[2]+=.5;if(Math.hypot(p[0],p[1])<=radius)label(p,`T${prediction.track_id} predicted`,'#9aabc4');}label([0,0,2.25],'EGO','#b7c9df');window.__viewerStatus={scene:scene.scene_name,frame:index,frames:scene.frames.length,points:pointVertices,boxes:f.boxes.length,playing,webglError:gl.getError()};}
function updateDetail(){const f=scene?.frames[index],box=f?.boxes.find(b=>b.track_id===selected);if(!box){$('detail').style.display='none';return;}$('detail').style.display='block';$('detail').textContent=`TRACK ${box.track_id}\nSpeed estimate   ${Math.hypot(...box.velocity_global).toFixed(2)} m/s\nRadar returns    ${box.radar_returns}\nTrack hits       ${box.hits}\nDimensions       ${box.high.map((x,i)=>(x-box.low[i]).toFixed(2)).join(' × ')} m`;}
function setFrame(i){if(!scene)return;index=Math.max(0,Math.min(scene.frames.length-1,i));dirty=true;}
function play(value=!playing){playing=value;lastAdvance=performance.now();$('play').textContent=playing?'Pause':'Play';$('stateText').textContent=playing?'Playing recorded scans':'Paused · actual detections';dirty=true;}
function resetCamera(){Object.assign(state,{yaw:-2.30,pitch:.70,distance:52,target:[4,0,.2],top:false});dirty=true;}
async function decodePayload(payload){const bytes=Uint8Array.from(atob(payload.point_data_gzip_base64),c=>c.charCodeAt(0)),stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')),buffer=await new Response(stream).arrayBuffer();return {...payload,points:new Float32Array(buffer)};}
function fail(message){errorMessage=message;$('loading').style.display='flex';$('loading').querySelector('.spinner').style.display='none';$('loadTitle').textContent='The scene could not be opened';$('loadMessage').textContent=message;console.error(message);window.__viewerError=message;}
async function loadScene(sceneIndex,embedded=null){play(false);$('loading').style.display='flex';$('loadTitle').textContent='Preparing your 3D scene';$('loadMessage').textContent='Loading recorded LiDAR points and actual Python detections. A new scene is processed once and cached.';$('scene').disabled=true;try{let payload=embedded;if(!payload){const response=await fetch(`/api/scene/${sceneIndex}`);if(!response.ok){const e=await response.json();throw Error(e.error||'Scene load failed');}payload=await response.json();}scene=await decodePayload(payload);index=0;selected=null;geometryKey='';$('sceneTitle').textContent=scene.scene_name;$('dataset').textContent=`nuScenes Mini · ${scene.frames.length} recorded scans · real sensor data`;$('timeline').max=scene.frames.length-1;$('scene').replaceChildren();scene.scenes.forEach((name,i)=>{const option=document.createElement('option');option.value=i;option.textContent=name;if((location.protocol==='file:'||scene.standalone)&&i!==scene.scene_index)option.disabled=true;$('scene').append(option);});$('scene').value=scene.scene_index;$('scene').disabled=false;if(location.protocol==='file:'||scene.standalone)$('sceneNote').textContent='Standalone scene. Use RUN_VIEWER.command to switch scenes.';$('loading').style.display='none';dirty=true;render();window.__viewerReady=true;window.__viewerSetFrame=i=>{play(false);setFrame(i);render();};}catch(error){fail(error.message);}}
for(const id of toggles)$(id).addEventListener('change',()=>{dirty=true;});$('pointsize').addEventListener('input',()=>{$('sizeValue').textContent=(+$('pointsize').value).toFixed(1);dirty=true;});$('range').addEventListener('input',()=>{$('rangeValue').textContent=$('range').value+' m';dirty=true;});$('timeline').addEventListener('input',()=>{play(false);setFrame(+$('timeline').value);});$('play').onclick=()=>play();$('prev').onclick=()=>{play(false);setFrame(index-1);};$('next').onclick=()=>{play(false);setFrame(index+1);};$('view3d').onclick=resetCamera;$('viewTop').onclick=()=>{state.pitch=1.565;state.yaw=-Math.PI/2;state.distance=65;state.target=[0,0,0];dirty=true;};$('scene').onchange=()=>loadScene(+$('scene').value);$('save').onclick=()=>{if(!scene)return;render();const image=document.createElement('canvas');image.width=canvas.width;image.height=canvas.height;const c=image.getContext('2d');c.drawImage(canvas,0,0);c.drawImage(overlay,0,0);image.toBlob(blob=>{if(!blob)return;const link=document.createElement('a'),url=URL.createObjectURL(blob);link.download=`${scene.scene_name}-frame-${String(index+1).padStart(3,'0')}.png`;link.href=url;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);},'image/png');};
let drag=null;canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,pan:e.shiftKey||e.button===2};};canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(drag.pan){const scale=state.distance*.0018,side=[-Math.sin(state.yaw),Math.cos(state.yaw),0],forward=[Math.cos(state.yaw),Math.sin(state.yaw),0];state.target=V.add(state.target,V.add(V.mul(side,-dx*scale),V.mul(forward,dy*scale)));}else{state.yaw-=dx*.006;state.pitch=Math.max(.08,Math.min(1.565,state.pitch+dy*.005));}drag.x=e.clientX;drag.y=e.clientY;dirty=true;};canvas.onpointerup=e=>{if(drag&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<5){const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top,near=labelTargets.filter(v=>Math.hypot(v.x-x,v.y-y)<30).sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y))[0];selected=near?.box.track_id??null;dirty=true;}drag=null;};canvas.oncontextmenu=e=>e.preventDefault();canvas.addEventListener('wheel',e=>{e.preventDefault();state.distance=Math.max(8,Math.min(130,state.distance*Math.exp(e.deltaY*.001)));dirty=true;},{passive:false});window.addEventListener('resize',()=>{dirty=true;});document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();play();}else if(e.code==='ArrowRight'){e.preventDefault();play(false);setFrame(index+1);}else if(e.code==='ArrowLeft'){e.preventDefault();play(false);setFrame(index-1);}else if(e.code==='KeyR')resetCamera();else if(e.code==='KeyT')$('viewTop').click();});
function tick(now){if(playing&&scene){const next=Math.min(index+1,scene.frames.length-1),interval=Math.max(.05,scene.frames[next].dt||.5)*1000/(+$('speed').value);if(now-lastAdvance>=interval){if(index+1<scene.frames.length)setFrame(index+1);else if($('loop').checked)setFrame(0);else play(false);lastAdvance=now;}}if(dirty&&scene){render();dirty=false;}requestAnimationFrame(tick);}try{initGL();const embedded=JSON.parse($('embedded-data').textContent);loadScene(embedded?.scene_index??+(new URLSearchParams(location.search).get('scene')||0),embedded);requestAnimationFrame(tick);}catch(error){fail(error.message);}
</script></body></html>
"""


def compare_fusion(replay_path,output_path):
    path=Path(replay_path)
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt") as stream_file:
        stream=json.load(stream_file)
    lidar_only=[{**frame,"detections":[{**d,"radar_available":False,"radar_velocity_global":None} for d in frame["detections"]]} for frame in stream]
    results={"lidar_only":evaluate_tracker_replay(lidar_only),"lidar_radar":evaluate_tracker_replay(stream),
        "protocol":{"frames":len(stream),"changed":"RADAR velocity initialization and updates only","tracking_gate_m":3.0,"scoring_gate_m":2.0,"range_m":25.0}}
    Path(output_path).write_text(json.dumps(results,indent=2))
    for method in ("lidar_only","lidar_radar"):
        print(method,json.dumps(results[method]["total"]))
    print("RESULTS",Path(output_path).resolve())
    return results


def run_self_tests():
    import unittest
    class Phase2Checks(unittest.TestCase):
        def test_box_overlap(self):
            self.assertAlmostEqual(calculate_bev_iou([0,0,2,2],[0,0,2,2]),1.)
            self.assertAlmostEqual(calculate_bev_iou([0,0,1,1],[2,2,3,3]),0.)
        def test_velocity_prediction(self):
            kalman=KalmanFilter2D([0,0],[2,-1]);kalman.predict(.5)
            np.testing.assert_allclose(kalman.get_state()["position"],[1,-.5])
            np.testing.assert_allclose(kalman.get_state()["velocity"],[2,-1])
        def test_covariance_and_update(self):
            kalman=KalmanFilter2D([0,0]);kalman.predict(.5)
            kalman.update_lidar([1,1]);kalman.update_radar([.5,.5])
            self.assertTrue(np.isfinite(kalman.x).all())
            np.testing.assert_allclose(kalman.P,kalman.P.T,atol=1e-10)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(kalman.P).min()),-1e-10)
        def test_track_lifecycle(self):
            tracker=MultiObjectTracker(max_distance=3.,max_missed=2)
            det={"center_global":np.array([0.,0.,0.]),"radar_available":False,"radar_velocity_global":None}
            with contextlib.redirect_stdout(io.StringIO()):
                tracker.update([det],0.);self.assertEqual(det["track_id"],1)
                shifted={**det,"center_global":np.array([.1,0.,0.])}
                tracker.update([shifted],.5);self.assertEqual(shifted["track_id"],1)
                self.assertEqual(len(tracker.update([],.5)),1)
                self.assertEqual(len(tracker.update([],.5)),1)
                self.assertEqual(len(tracker.update([],.5)),0)
        def test_ground_plausibility(self):
            self.assertTrue(ground_model_is_plausible([0,0,1,0]))
            self.assertFalse(ground_model_is_plausible([0,0,1,-2]))
            self.assertFalse(ground_model_is_plausible([1,0,0,0]))
        def test_empty_cluster_input(self):
            cloud=o3d.geometry.PointCloud();points,labels=cluster_obstacles(cloud)
            self.assertEqual(len(points),0);self.assertEqual(len(labels),0)
        def test_radar_association_is_unique(self):
            detections=[]
            for label,x in enumerate((0.,.8)):
                detections.append({"label":label,"center":np.array([x,0.,.5]),"bounding_box":o3d.geometry.AxisAlignedBoundingBox([x-.5,-.5,0],[x+.5,.5,1])})
            radar={"positions":np.array([[.1,0.,.5]]),"velocities":np.array([[2.,0.,0.]])}
            result=associate_radar_with_lidar(detections,radar)
            self.assertEqual(sum(d["radar_points"] for d in result),1)
        def test_scorer_assignment(self):
            pairs=assign_centers([[1.4,0],[0,0]],[[0,0],[3,0]],2.)
            self.assertEqual({(i,j) for i,j,_ in pairs},{(0,1),(1,0)})
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Phase2Checks))
    return 0 if outcome.wasSuccessful() else 1


def main(argv=None):
    global DATASET_PATH
    parser=argparse.ArgumentParser(description="Phase 2 LiDAR and RADAR fusion tracking and 3D viewer")
    parser.add_argument("--mode",choices=("viewer","evaluate","compare","self-test"),default="viewer")
    parser.add_argument("--dataset-path",default=DATASET_PATH)
    parser.add_argument("--scene",type=int,default=0)
    parser.add_argument("--scans",type=int,default=0,help="0 uses the complete scene")
    parser.add_argument("--all-scenes",action="store_true",help="evaluate all Mini scenes")
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--no-open",action="store_true")
    parser.add_argument("--export-html",metavar="FILE")
    parser.add_argument("--replay",default=str(ROOT/"results"/"tracking_replay.json.gz"))
    parser.add_argument("--output",default=str(ROOT/"phase2_results.json"))
    args=parser.parse_args(argv)
    if not 0<=args.scene<10:parser.error("--scene must be between 0 and 9")
    if args.scans<0:parser.error("--scans must be 0 or positive")
    if not 0<=args.port<=65535:parser.error("--port must be between 0 and 65535")
    DATASET_PATH=args.dataset_path
    if args.mode=="self-test":return run_self_tests()
    if args.mode=="compare":
        if not Path(args.replay).is_file():parser.error("Replay file not found. Use the supplied results folder or --replay FILE.")
        compare_fusion(args.replay,args.output);return 0
    if not (Path(DATASET_PATH)/"v1.0-mini").is_dir():parser.error("nuScenes Mini tables not found at --dataset-path")
    if args.export_html:
        export_scene(args.scene,args.export_html);return 0
    if args.mode=="evaluate":
        initialize_dataset();scans=[]
        selected=range(len(nusc.scene)) if args.all_scenes else [args.scene]
        for scene_index in selected:
            count=nusc.scene[scene_index]["nbr_samples"] if args.scans==0 else min(args.scans,nusc.scene[scene_index]["nbr_samples"])
            scans.extend(process_scene_sequence(count,scene_index))
        metrics=evaluate_sequence_detections(scans,max_range=25.,iou_threshold=.1)
        Path(args.output).write_text(json.dumps({"frames":len(scans),"detection":metrics},indent=2));return 0
    try:server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    except OSError as error:
        if error.errno!=errno.EADDRINUSE:raise
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    url=f"http://127.0.0.1:{server.server_port}/?scene={args.scene}"
    print("VIEWER_URL",url,flush=True)
    if not args.no_open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
    return 0


if __name__=="__main__":
    raise SystemExit(main())

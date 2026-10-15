"""
Fast-Calib 鱼眼/针孔通用激光相机外参标定主程序
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import logging
import os
from itertools import combinations
from typing import Optional, Tuple
import cv2
import numpy as np
import av
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from utils import (
    setup_logger, load_yaml_config, pointcloud2_to_xyzi, msg_stamp_ns,
    save_T_yaml, save_point_cloud, colorize_by_holes, circle_fit_2d, residual_3d
)
from detector import CalibrationBoardDetector
from cameras import BaseCamera, FisheyeCamera, PinholeCamera
try:
    from second_board_detector import SECONDBoardDetector
except ImportError:
    SECONDBoardDetector = None


class CompressedVideoDecoder:
    def __init__(self, fmt: str, format_map: dict):
        codec = format_map.get(str(fmt).lower().strip())
        if codec is None:
            raise ValueError(f"unsupported CompressedVideo format: {fmt}")
        self.codec = av.CodecContext.create(codec, "r")
        self.codec.open()
        self._ts_by_pts = {}
        self._idx = 0
        self.n_err = 0

    def decode(self, data, t_ns):
        pts = self._idx
        self._idx += 1
        self._ts_by_pts[pts] = t_ns
        pkt = av.Packet(bytes(data))
        pkt.pts = pkt.dts = pts
        out = []
        try:
            for frame in self.codec.decode(pkt):
                t = self._ts_by_pts.pop(frame.pts, t_ns) if frame.pts is not None else t_ns
                out.append((t, frame.to_ndarray(format="bgr24")))
        except Exception as e:
            self.n_err += 1
            if self.n_err <= 5:
                logging.warning(f"video frame #{pts} decode failed: {e}")
        return out


class FastCalibLidarCamCalibrator:
    def __init__(
        self,
        hole_centers_board: np.ndarray,
        marker_size: float,
        hole_radius: float,
        marker_centers_board: dict,
        max_dt_ns: int,
        plane_tol: float,
        plane_iters: int,
        board_vertical_prior: bool,
        camera: BaseCamera,
        hole_palette: np.ndarray,
        save_intermediate: bool,
        save_pointcloud: bool,
        save_roi_pointcloud: bool,
        output_root: str, # 替换原来两个out_root
        aruco_dict_id=cv2.aruco.DICT_6X6_250,
        board_detector: Optional[CalibrationBoardDetector] = None
    ):
        self.hole_centers_board = hole_centers_board
        self._hole_pair_dists = np.sort(pdist(hole_centers_board))
        ux = np.unique(hole_centers_board[:, 0])
        uy = np.unique(hole_centers_board[:, 1])
        self.delta_width = float(ux[-1] - ux[0])
        self.delta_height = float(uy[-1] - uy[0])
        self.marker_centers_board = marker_centers_board
        self.marker_size = marker_size
        self.hole_radius = hole_radius
        self.plane_tol = plane_tol
        self.plane_iters = plane_iters
        self.board_vertical_prior = board_vertical_prior
        self.max_dt_ns = max_dt_ns
        self.save_intermediate = save_intermediate
        self.save_pointcloud = save_pointcloud
        self.save_roi_pointcloud = save_roi_pointcloud
        # 从总output_root派生目录
        self.out_root_result = os.path.join(output_root, "result")
        self.out_root_intermediate = os.path.join(output_root, "intermediate")
        os.makedirs(self.out_root_result, exist_ok=True)
        os.makedirs(self.out_root_intermediate, exist_ok=True)
        self.camera = camera
        self.hole_palette = hole_palette
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.board_detector = board_detector

    def _make_cam_dir(self, cam_name: str) -> str:
        d = os.path.join(self.out_root_intermediate, cam_name)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        return d

    def detect_fastcalib_holes_2d(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None or len(ids) < 4:
            return None, corners, ids, None, None
        ids = ids.flatten()
        idx_map = {int(i): k for k, i in enumerate(ids)}
        try:
            m_centers = {}
            for marker_id in [1, 2, 3, 4]:
                c = corners[idx_map[marker_id]].reshape(4, 2).mean(axis=0)
                m_centers[marker_id] = c
        except KeyError:
            return None, corners, ids, None, None

        hole_uvs = []
        for hx, hy in self.hole_centers_board[:, :2]:
            u = 0.0
            v = 0.0
            for marker_id, (mx, my) in self.marker_centers_board.items():
                w = 1.0 / ((hx - mx) ** 2 + (hy - my) ** 2 + 1e-6)
                u += w * m_centers[marker_id][0]
                v += w * m_centers[marker_id][1]
            w_sum = sum(1.0 / ((hx - mx) ** 2 + (hy - my) ** 2 + 1e-6) for mx, my in self.marker_centers_board.values())
            hole_uvs.append([u / w_sum, v / w_sum])
        hole_uvs = np.array(hole_uvs, dtype=np.float64)

        pts2d = np.array([m_centers[i] for i in [1, 2, 3, 4]], dtype=np.float64)
        pts3d = np.array([self.marker_centers_board[i] for i in [1, 2, 3, 4]], dtype=np.float64)
        ret, rvec, tvec = cv2.solvePnP(pts3d, pts2d, self.camera.K, self.camera.dist, flags=cv2.SOLVEPNP_P3P)
        if not ret:
            return hole_uvs, corners, ids, None, None
        R_board = cv2.Rodrigues(rvec)[0]
        return hole_uvs, corners, ids, R_board, tvec

    def detect_hole_centers_lidar(self, pc_roi: np.ndarray):
        if pc_roi.shape[0] < 300:
            return None, None
        # 第一轮 RANSAC: 拟合板子平面
        plane_eq, inlier_ratio = self._fit_plane_ransac(pc_roi[:, :3])
        if plane_eq is None:
            return None, None
        n = plane_eq[:3]
        d = plane_eq[3]
        dist = np.abs(pc_roi[:, 0] * n[0] + pc_roi[:, 1] * n[1] + pc_roi[:, 2] * n[2] + d)
        inlier_mask = dist < self.plane_tol
        pc_plane = pc_roi[inlier_mask]
        ring_inlier = pc_roi[inlier_mask, 4] if pc_roi.shape[1] > 4 else np.arange(len(pc_plane))
        if pc_plane.shape[0] < 200:
            return None, None

        # 第二轮 RANSAC: 在平面内找 4 个孔
        holes3d = []
        rest = pc_plane[:, :3].copy()
        rest_ring = ring_inlier.copy()
        for _ in range(4):
            if rest.shape[0] < 100:
                break
            center, radius, ok = self._fit_hole_ransac(rest)
            if not ok or radius < self.hole_radius * 0.5 or radius > self.hole_radius * 2.0:
                break
            holes3d.append(center)
            # 删除该孔周围的点
            r = np.linalg.norm(rest - center, axis=1)
            keep = r > (radius + 0.05)
            rest = rest[keep]
            rest_ring = rest_ring[keep]

        if len(holes3d) < 4:
            return None, None
        holes3d = np.array(holes3d, dtype=np.float64)

        # 按板子坐标系排序 (和 hole_centers_board 对应)
        centroid = holes3d.mean(axis=0)
        _, eigvecs = np.linalg.eigh(np.cov(holes3d - centroid, rowvar=False))
        v1 = eigvecs[:, -1]  # 长轴
        v2 = eigvecs[:, -2]  # 短轴
        # 保证 z 轴向上
        z_axis = np.cross(v1, v2)
        if z_axis[2] < 0:
            v1 = -v1
            z_axis = np.cross(v1, v2)
        # 投影到 2D
        pts_2d = np.stack([(holes3d - centroid) @ v1, (holes3d - centroid) @ v2], axis=1)
        # 和标准孔中心做匹配 (最小距离)
        std_2d = self.hole_centers_board[:, :2].copy()
        std_cent = std_2d.mean(axis=0)
        std_2d = std_2d - std_cent
        # 尝试 4 种旋转 (0, 90, 180, 270) 找最佳匹配
        best_err = 1e9
        best_pts = holes3d.copy()
        for rot_k in range(4):
            cos_r = np.cos(rot_k * np.pi / 2)
            sin_r = np.sin(rot_k * np.pi / 2)
            rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            pts_2d_rot = pts_2d @ rot_mat.T
            # 排序匹配
            err = 0.0
            for i in range(4):
                d = np.linalg.norm(pts_2d_rot - std_2d[i], axis=1)
                err += d.min()
            if err < best_err:
                best_err = err
                # 找到每个标准孔对应的最近点
                matched = np.zeros(4, dtype=int)
                used = set()
                for i in range(4):
                    d = np.linalg.norm(pts_2d_rot - std_2d[i], axis=1)
                    sorted_idx = np.argsort(d)
                    for j in sorted_idx:
                        if j not in used:
                            matched[i] = j
                            used.add(j)
                            break
                best_pts = holes3d[matched]
        return best_pts, plane_eq

    def _fit_plane_ransac(self, pts: np.ndarray):
        if pts.shape[0] < 10:
            return None, 0.0
        best_eq = None
        best_inlier = 0
        for _ in range(self.plane_iters):
            idx = np.random.choice(pts.shape[0], 3, replace=False)
            p1, p2, p3 = pts[idx]
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n = n / norm
            d = -n @ p1
            dist = np.abs(pts @ n + d)
            inlier = np.sum(dist < self.plane_tol)
            if inlier > best_inlier:
                best_inlier = inlier
                best_eq = np.array([n[0], n[1], n[2], d])
        return best_eq, best_inlier / pts.shape[0] if pts.shape[0] > 0 else 0.0

    def _fit_hole_ransac(self, pts: np.ndarray):
        if pts.shape[0] < 10:
            return None, None, False
        best_center = None
        best_radius = None
        best_inlier = 0
        for _ in range(100):
            idx = np.random.choice(pts.shape[0], 3, replace=False)
            p1, p2, p3 = pts[idx]
            center, radius = circle_fit_2d(np.stack([p1[:2], p2[:2], p3[:2]]))
            r = np.linalg.norm(pts[:, :2] - center, axis=1)
            inlier = np.sum(np.abs(r - radius) < 0.02)
            if inlier > best_inlier:
                best_inlier = inlier
                best_center = np.array([center[0], center[1], pts[:, 2].mean()])
                best_radius = radius
        if best_inlier < 20:
            return None, None, False
        return best_center, best_radius, True

    def calibrate_single_cam(self, lidar_3d_list, img_2d_list, board_poses, edge_frames=None):
        pts_3d = []
        pts_2d = []
        for h3d, h2d in zip(lidar_3d_list, img_2d_list):
            pts_3d.append(h3d)
            pts_2d.append(h2d)
        pts_3d = np.vstack(pts_3d)
        pts_2d = np.vstack(pts_2d)

        # 初始值: 用 solvePnP 估计
        ret, rvec_init, tvec_init = cv2.solvePnP(
            pts_3d.astype(np.float64), pts_2d.astype(np.float64),
            self.camera.K, self.camera.dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ret:
            rvec_init = np.zeros(3)
            tvec_init = np.array([0.0, 0.0, 2.0])

        # LM 优化
        x0 = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
        result = least_squares(
            residual_3d, x0, args=(pts_3d, pts_2d),
            method="lm", max_nfev=2000,
        )
        R_opt = Rotation.from_rotvec(result.x[:3]).as_matrix()
        t_opt = result.x[3:6]
        T_opt = np.eye(4)
        T_opt[:3, :3] = R_opt
        T_opt[:3, 3] = t_opt

        # 计算重投影误差
        pts_2d_proj = self.camera.project(T_opt, pts_3d)
        err = np.linalg.norm(pts_2d_proj - pts_2d, axis=1)
        mean_err = float(err.mean())
        return T_opt, mean_err, result

    def _read_lidar_msgs(self, mcap_path, lidar_topic, point_field_datatype_to_np):
        lidar_msgs = []
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, msg, ros_msg in reader.iter_decoded_messages(topics=[lidar_topic]):
                t_ns = msg_stamp_ns(ros_msg, msg)
                pc = pointcloud2_to_xyzi(ros_msg, point_field_datatype_to_np)
                lidar_msgs.append((t_ns, pc))
        logging.info(f"lidar frames count = {len(lidar_msgs)}")
        return lidar_msgs

    def _iter_image_frames(self, mcap_path, img_topic, format_map):
        video_dec = None
        n_img = 0
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for schema, channel, msg, ros_msg in reader.iter_decoded_messages(topics=[img_topic]):
                t_ns = msg_stamp_ns(ros_msg, msg)
                name = type(ros_msg).__name__
                # 兼容 foxglove_msgs/msg/CompressedVideo 等带前缀的消息类型
                is_compressed_video = "CompressedVideo" in name
                is_compressed_image = "CompressedImage" in name
                is_image = name == "Image"
                try:
                    if is_compressed_video:
                        if video_dec is None:
                            video_dec = CompressedVideoDecoder(ros_msg.format, format_map)
                            logging.info(f"CompressedVideo format={ros_msg.format}")
                        for ft, fimg in video_dec.decode(bytes(ros_msg.data), t_ns):
                            n_img += 1
                            yield ft, fimg
                    elif is_compressed_image:
                        img = cv2.imdecode(np.frombuffer(bytes(ros_msg.data), np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            n_img += 1
                            yield t_ns, img
                    elif is_image:
                        img = np.frombuffer(bytes(ros_msg.data), np.uint8).reshape(ros_msg.height, ros_msg.width, -1)
                        enc = (ros_msg.encoding or "").lower()
                        if img.shape[2] == 1:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        elif enc.startswith("rgb"):
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        n_img += 1
                        yield t_ns, img.copy()
                    else:
                        logging.warning(f"unsupported image msg type: {name}, skip")
                except Exception as e:
                    logging.error(f"decode image err ({name}): {e}")
        extra_err = f", video decode errors={video_dec.n_err}" if video_dec else ""
        logging.info(f"img frames count = {n_img}{extra_err}")

    def read_mcap_sync(self, mcap_path, img_topic, lidar_topic, point_field_datatype_to_np, format_map, max_frames=50, seed=0):
        lidar_msgs = self._read_lidar_msgs(mcap_path, lidar_topic, point_field_datatype_to_np)
        if not lidar_msgs:
            logging.warning("no lidar data, cannot sync")
            return
        # 先把所有图像读出来, 以雷达为基准找最近图像
        img_msgs = list(self._iter_image_frames(mcap_path, img_topic, format_map))
        if len(img_msgs) == 0:
            logging.error("没有读取到任何图像帧, 请检查图像话题和消息类型是否正确!")
            return
        img_ts = np.array([t for t, _ in img_msgs])
        imgs = [img for _, img in img_msgs]
        n_pair = n_drop = 0
        sync_pairs = []
        for t_lidar, pc in lidar_msgs:
            idx = int(np.argmin(np.abs(img_ts - t_lidar)))
            if abs(int(img_ts[idx]) - t_lidar) < self.max_dt_ns:
                sync_pairs.append((imgs[idx], pc))
                n_pair += 1
            else:
                n_drop += 1
        logging.info(f"synced pairs = {n_pair}, dropped = {n_drop}")
        if len(sync_pairs) > max_frames:
            rng = np.random.default_rng(seed)
            sel = rng.choice(len(sync_pairs), max_frames, replace=False)
            sel.sort()
            sync_pairs = [sync_pairs[i] for i in sel]
            logging.info(f"sampled {len(sync_pairs)} / {n_pair} frames seed={seed}")
        for img, pc in sync_pairs:
            yield img, pc

    def run(self, mcap_path, cam_topic_dict, lidar_topic, pc_roi_lim, min_sample, pcl_fmt, n_calib_frames, calib_seed, point_field_datatype_to_np, format_map):
        out_result = {}
        manual_xlim, manual_ylim, manual_zlim = pc_roi_lim
        for cam_name, img_topic in cam_topic_dict.items():
            logging.info(f"\n======== Calibrate {cam_name} ========")
            cam_out_dir = self._make_cam_dir(cam_name) if self.save_intermediate else ""
            pointcloud_dir = os.path.join(cam_out_dir, "pointcloud")
            roi_pointcloud_dir = os.path.join(cam_out_dir, "roi_pointcloud")
            if self.save_pointcloud:
                os.makedirs(pointcloud_dir, exist_ok=True)
                logging.info(f"save_pointcloud is ON, raw full point cloud save to: {pointcloud_dir}")
            if self.save_roi_pointcloud:
                os.makedirs(roi_pointcloud_dir, exist_ok=True)
                logging.info(f"save_roi_pointcloud is ON, roi cropped point cloud save to: {roi_pointcloud_dir}")
            lidar_3d_list, img_2d_list, board_poses = [], [], []
            valid_frames, viz_frames, edge_frames = [], [], []
            n_pairs = n_fail_img = n_fail_lidar = n_fail_roi = 0
            logging.info(f"ROI: x=[{manual_xlim[0]:.2f},{manual_xlim[1]:.2f}] y=[{manual_ylim[0]:.2f},{manual_ylim[1]:.2f}] z=[{manual_zlim[0]:.2f},{manual_zlim[1]:.2f}]")
            logging.info(f"detector: {'SECOND auto' if self.board_detector is not None else 'manual ROI'}")
            for frame_idx, (img, pc) in enumerate(self.read_mcap_sync(mcap_path, img_topic, lidar_topic, point_field_datatype_to_np, format_map, max_frames=n_calib_frames, seed=calib_seed)):
                n_pairs += 1
                logging.info(f"--- Frame {frame_idx}/{n_calib_frames} ---")
                if self.save_pointcloud:
                    pcd_path = os.path.join(pointcloud_dir, f"frame_{frame_idx:04d}.pcd")
                    save_point_cloud(pc, pcd_path)

                # 1. 图像 Aruco 检测
                logging.info(f"  [1/4] 检测图像 Aruco 标记...")
                holes2d, corners, ids, R_board, tvec_board = self.detect_fastcalib_holes_2d(img)
                if holes2d is not None:
                    logging.info(f"        图像检测成功: {len(holes2d)} 个孔中心")
                else:
                    logging.warning(f"        图像 Aruco 检测失败, 跳过该帧")
                    n_fail_img += 1
                    continue

                # 2. ROI 区域检测 (SECOND 或手动)
                logging.info(f"  [2/4] 确定 ROI 区域...")
                if self.board_detector is not None:
                    detect_ok, auto_roi = self.board_detector.detect(pc)
                    if detect_ok and auto_roi is not None:
                        xlim, ylim, zlim = auto_roi
                        logging.info(f"        SECOND 自动检测: x=[{xlim[0]:.2f},{xlim[1]:.2f}] y=[{ylim[0]:.2f},{ylim[1]:.2f}]")
                    else:
                        logging.warning(f"        SECOND 检测失败, 回退到手动 ROI")
                        xlim, ylim, zlim = manual_xlim, manual_ylim, manual_zlim
                else:
                    xlim, ylim, zlim = manual_xlim, manual_ylim, manual_zlim
                    logging.info(f"        手动 ROI: x=[{xlim[0]:.2f},{xlim[1]:.2f}]")

                roi_mask = ((pc[:, 0] >= xlim[0]) & (pc[:, 0] <= xlim[1])
                            & (pc[:, 1] >= ylim[0]) & (pc[:, 1] <= ylim[1])
                            & (pc[:, 2] >= zlim[0]) & (pc[:, 2] <= zlim[1]))
                pc_roi = pc[roi_mask]
                logging.info(f"        ROI 内点数: {pc_roi.shape[0]}")

                if pc_roi.shape[0] < 300:
                    logging.warning(f"        ROI 内点数太少 ({pc_roi.shape[0]} < 300), 跳过该帧")
                    n_fail_roi += 1
                    continue

                # 3. 雷达孔中心检测 (RANSAC 平面 + 圆拟合)
                logging.info(f"  [3/4] 雷达 RANSAC 平面拟合 + 孔中心提取...")
                holes3d, _ = self.detect_hole_centers_lidar(pc_roi)
                if holes3d is not None and len(holes3d) == 4:
                    logging.info(f"        雷达检测成功: 4 个孔中心")
                else:
                    logging.warning(f"        雷达孔中心检测失败 (holes3d={holes3d is not None and len(holes3d) if holes3d is not None else 0}), 跳过该帧")
                    n_fail_lidar += 1
                    continue

                # 4. 保存匹配对
                logging.info(f"  [4/4] 保存匹配对 (累计 {len(lidar_3d_list)+1} 对)")
                if self.save_intermediate:
                    img_draw = img.copy()
                    if corners is not None:
                        cv2.aruco.drawDetectedMarkers(img_draw, corners, ids)
                    if holes2d is not None:
                        for uv in holes2d:
                            cv2.circle(img_draw, (int(uv[0]), int(uv[1])), 6, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(cam_out_dir, "images", f"frame_{frame_idx:04d}.jpg"), img_draw)
                if self.save_roi_pointcloud:
                    pc_base = os.path.join(roi_pointcloud_dir, f"frame_{frame_idx:04d}")
                    if pcl_fmt == "bin":
                        pc_roi[:, :4].astype(np.float32).tofile(pc_base + ".bin")
                    else:
                        cols = colorize_by_holes(pc_roi[:, :3], holes3d, self.hole_radius, self.hole_palette)
                        pts, cls = [pc_roi[:, :3]], [cols]
                        if holes3d is not None:
                            pts.append(holes3d)
                            cls.append(self.hole_palette[:len(holes3d)])
                        pcd_o = o3d.geometry.PointCloud()
                        pcd_o.points = o3d.utility.Vector3dVector(np.vstack(pts).astype(np.float64))
                        pcd_o.colors = o3d.utility.Vector3dVector(np.vstack(cls).astype(np.float64))
                        o3d.io.write_point_cloud(pc_base + (".pcd" if pcl_fmt == "pcd" else ".ply"), pcd_o)

                lidar_3d_list.append(holes3d)
                img_2d_list.append(holes2d)
                board_poses.append((R_board.copy(), tvec_board.copy()))
                valid_frames.append(frame_idx)
                if len(viz_frames) < 6:
                    viz_frames.append((frame_idx, img.copy(), pc.copy()))
                if len(edge_frames) < 8:
                    ed = cv2.dilate(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 60, 160), np.ones((3, 3), np.uint8))
                    edge_frames.append((ed, pc.copy()))

            logging.info(f"\n==== 帧处理完成 ====")
            logging.info(f"总帧数: {n_pairs}, 有效帧数: {len(lidar_3d_list)}")
            logging.info(f"失败原因: 图像={n_fail_img}, 雷达={n_fail_lidar}, ROI={n_fail_roi}")

            if len(lidar_3d_list) < min_sample:
                logging.warning(f"\n有效样本数 {len(lidar_3d_list)} < 最小要求 {min_sample}, 跳过标定!")
                logging.warning(f"请检查: 1) SECOND 检测器是否正常 2) 图像 Aruco 是否能识别 3) 雷达 ROI 内点数是否足够")
                continue
            if len(lidar_3d_list) >= 2:
                cent = np.array([h.mean(axis=0) for h in lidar_3d_list])
                spread = float(np.max(np.linalg.norm(cent - cent.mean(axis=0), axis=1)))
                logging.info(f"标定板移动范围: {spread:.3f}m (建议 > 0.2m)")
                if spread < 0.2:
                    logging.warning(f"board movement {spread:.3f}m, recommend multi‑pose capture")

            logging.info(f"\n==== 开始 LM 优化 ====")
            logging.info(f"输入: {len(lidar_3d_list)} 组 3D-2D 匹配点")
            T_opt, mean_err, _ = self.calibrate_single_cam(lidar_3d_list, img_2d_list, board_poses, edge_frames=edge_frames or None)
            logging.info(f"LM 优化完成, 平均重投影误差: {mean_err:.3f} px")
            npy_out = os.path.join(self.out_root_result, f"{cam_name}_lidar2cam_fastcalib.npy")
            yaml_out = os.path.join(self.out_root_result, f"{cam_name}_lidar2cam_fastcalib.yaml")
            np.save(npy_out, T_opt)
            save_T_yaml(yaml_out, T_opt, {"mean_reproj_err_px": float(mean_err)})
            logging.info(f"saved {npy_out} / {yaml_out}")
            if self.save_intermediate:
                for fi, h3, h2 in zip(valid_frames, lidar_3d_list, img_2d_list):
                    imgd = cv2.imread(os.path.join(cam_out_dir, "images", f"frame_{fi:04d}.jpg"))
                    if imgd is None:
                        continue
                    pts_2d_proj = self.camera.project(T_opt, h3)
                    for uv in pts_2d_proj:
                        cv2.circle(imgd, (int(uv[0]), int(uv[1])), 5, (0, 255, 0), -1)
                    cv2.imwrite(os.path.join(cam_out_dir, "overlays", f"frame_{fi:04d}.jpg"), imgd)
            out_result[cam_name] = T_opt
        return out_result

    def visualize_with_T(self, mcap_path, img_topic, lidar_topic, pc_roi_lim, T, frame_idx, point_field_datatype_to_np, format_map, out_dir=None):
        if out_dir is None:
            out_dir = os.path.join(self.out_root_intermediate, "vis_overlays")
        os.makedirs(out_dir, exist_ok=True)
        manual_xlim, manual_ylim, manual_zlim = pc_roi_lim
        for i, (img, pc) in enumerate(self.read_mcap_sync(mcap_path, img_topic, lidar_topic, point_field_datatype_to_np, format_map, max_frames=frame_idx + 1)):
            if i != frame_idx:
                continue
            xlim, ylim, zlim = manual_xlim, manual_ylim, manual_zlim
            if self.board_detector is not None:
                detect_ok, auto_roi = self.board_detector.detect(pc)
                if detect_ok and auto_roi is not None:
                    xlim, ylim, zlim = auto_roi
            roi_mask = ((pc[:, 0] >= xlim[0]) & (pc[:, 0] <= xlim[1])
                        & (pc[:, 1] >= ylim[0]) & (pc[:, 1] <= ylim[1])
                        & (pc[:, 2] >= zlim[0]) & (pc[:, 2] <= zlim[1]))
            pc_roi = pc[roi_mask]
            holes3d, _ = self.detect_hole_centers_lidar(pc_roi)
            img_draw = img.copy()
            if holes3d is not None and len(holes3d) == 4:
                pts_2d_proj = self.camera.project(T, holes3d)
                for uv in pts_2d_proj:
                    cv2.circle(img_draw, (int(uv[0]), int(uv[1])), 5, (0, 255, 0), -1)
            out_path = os.path.join(out_dir, f"vis_frame_{frame_idx:04d}.jpg")
            cv2.imwrite(out_path, img_draw)
            logging.info(f"saved {out_path}")


def load_T(path):
    if path.endswith(".npy"):
        T = np.load(path)
        yaml_path = path.replace(".npy", ".yaml")
        if os.path.exists(yaml_path):
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                meta = yaml.safe_load(f)
            return T, meta
        return T, {}
    elif path.endswith(".yaml"):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        R = np.array(d["Rcl"]).reshape(3, 3)
        t = np.array(d["Pcl"])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T, d
    else:
        raise ValueError(f"unsupported file type: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis", default=None, help="外参npy/yaml路径，质检投影")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--config", type=str, default="config/config.yaml", help="配置yaml路径")
    args = ap.parse_args()
    cfg = load_yaml_config(args.config)
    output_root = cfg["output_dir"]
    log_file = os.path.join(output_root, "calib.log")
    os.makedirs(output_root, exist_ok=True)
    setup_logger(log_file)

    calib_board = cfg["calib_board"]
    hole_centers_board = np.array(calib_board["hole_centers_board"], dtype=np.float64)
    marker_size = calib_board["marker_size"]
    hole_radius = calib_board["hole_radius"]
    marker_centers_board = calib_board["marker_centers_board"]
    hole_palette = np.array(cfg["hole_palette"], dtype=np.float64)
    point_field_datatype_to_np = cfg["point_field_datatype_to_np"]
    format_map = cfg["format_map"]
    cam_int = cfg["camera_intrinsic"]
    K = np.array(cam_int["K"], dtype=np.float64)
    dist = np.array(cam_int["dist"], dtype=np.float64)
    cam_type = cam_int.get("type", "fisheye")
    # 根据配置实例化相机
    if cam_type == "fisheye":
        camera = FisheyeCamera(K, dist)
    elif cam_type == "pinhole":
        camera = PinholeCamera(K, dist)
    else:
        raise ValueError(f"unsupported camera type {cam_type}")
    algo_cfg = cfg["calib_algo"]
    default_pc_roi_lim = tuple(tuple(x) for x in cfg["default_pc_roi_lim"])
    data_cfg = cfg["data_config"]
    mcap_file = data_cfg["mcap_file"]
    cam_topics = data_cfg["cam_topics"]
    lidar_topic = data_cfg["lidar_topic"]

    # SECOND 检测器 (可选)
    board_detector = None
    second_cfg = cfg.get("second_detector", {})
    if second_cfg.get("enabled", False) and SECONDBoardDetector is not None:
        logging.info(f"loading SECOND detector: {second_cfg['model_path']}")
        board_detector = SECONDBoardDetector(
            second_cfg["model_path"],
            score_thr=second_cfg.get("score_thr", 0.3),
        )

    calibrator = FastCalibLidarCamCalibrator(
        hole_centers_board=hole_centers_board,
        marker_size=marker_size,
        hole_radius=hole_radius,
        marker_centers_board=marker_centers_board,
        max_dt_ns=algo_cfg["max_dt_ns"],
        plane_tol=algo_cfg["plane_tol"],
        plane_iters=algo_cfg["plane_iters"],
        board_vertical_prior=algo_cfg["board_vertical_prior"],
        camera=camera,
        hole_palette=hole_palette,
        save_intermediate=algo_cfg["save_intermediate"],
        save_pointcloud=algo_cfg["save_pointcloud"],
        save_roi_pointcloud=algo_cfg["save_roi_pointcloud"],
        output_root=output_root,
        aruco_dict_id=cv2.aruco.DICT_6X6_250,
        board_detector=board_detector
    )
    if args.vis:
        T, meta_info = load_T(args.vis)
        logging.info("==== Load calib yaml meta info ====")
        logging.info(meta_info)
        vis_out_dir = os.path.join(output_root, "vis_overlays")
        cam_topic = list(cam_topics.values())[0]
        calibrator.visualize_with_T(
            mcap_file,
            cam_topic,
            lidar_topic,
            default_pc_roi_lim,
            T,
            args.frame,
            point_field_datatype_to_np,
            format_map,
            out_dir=vis_out_dir
        )
    else:
        calibrator.run(
            mcap_path=mcap_file,
            cam_topic_dict=cam_topics,
            lidar_topic=lidar_topic,
            pc_roi_lim=default_pc_roi_lim,
            min_sample=algo_cfg["min_sample"],
            pcl_fmt=algo_cfg["pcl_fmt"],
            n_calib_frames=algo_cfg["n_calib_frames"],
            calib_seed=algo_cfg["calib_seed"],
            point_field_datatype_to_np=point_field_datatype_to_np,
            format_map=format_map
        )
    logging.info("\n===== Calibration done =====")


if __name__ == "__main__":
    main()

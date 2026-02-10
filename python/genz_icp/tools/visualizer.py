import importlib
import os
import time
from abc import ABC
import numpy as np

# --- CẤU HÌNH GIAO DIỆN ---
START_BUTTON = " START\n[SPACE]"
PAUSE_BUTTON = " PAUSE\n[SPACE]"
NEXT_FRAME_BUTTON = "NEXT FRAME\n\t\t [N]"
SCREENSHOT_BUTTON = "SCREENSHOT\n\t\t  [S]"
LOCAL_VIEW_BUTTON = "LOCAL VIEW\n\t\t [G]"
GLOBAL_VIEW_BUTTON = "GLOBAL VIEW\n\t\t  [G]"
CENTER_VIEWPOINT_BUTTON = "CENTER VIEWPOINT\n\t\t\t\t[C]"
QUIT_BUTTON = "QUIT\n  [Q]"

BACKGROUND_COLOR = [0.0, 0.0, 0.0]

# --- BẢNG MÀU ---
FRAME_COLOR = [0.8470, 0.1058, 0.3764]       # Đỏ hồng (Source - Raw Frame)
PLANAR_COLOR = [0.0, 0.4, 1.0]               # Xanh dương (Planar)
NON_PLANAR_COLOR = [1.0, 0.8, 0.0]           # Vàng tươi (Non-Planar)
LOCAL_MAP_COLOR = [0.0, 0.3019, 0.2509]      # Xanh lá đậm (Map)
TRAJECTORY_COLOR = [1.0, 0.0, 0.0]           # Đỏ tươi (Đường đi)

# Kích thước điểm mặc định
FRAME_PTS_SIZE = 0.05
PLANAR_PTS_SIZE = 0.1
NON_PLANAR_PTS_SIZE = 0.15
MAP_PTS_SIZE = 0.08

class StubVisualizer(ABC):
    # Cập nhật signature: Thêm source vào đầu
    def update(self, source, planar, non_planar, local_map, pose, vis_infos=None): pass
    def close(self): pass

class RegistrationVisualizer(StubVisualizer):
    def __init__(self):
        try:
            self._ps = importlib.import_module("polyscope")
            self._gui = self._ps.imgui
        except ModuleNotFoundError:
            raise ModuleNotFoundError('polyscope is not installed. Run "pip install polyscope"')

        # Trạng thái giao diện
        self._background_color = BACKGROUND_COLOR
        self._frame_size = FRAME_PTS_SIZE
        self._planar_size = PLANAR_PTS_SIZE
        self._non_planar_size = NON_PLANAR_PTS_SIZE
        self._map_size = MAP_PTS_SIZE
        
        self._block_execution = True
        self._play_mode = False

        # Toggles hiển thị
        self._toggle_frame = False        # Mặc định tắt Source (để nhìn rõ feature hơn)
        self._toggle_planar = True
        self._toggle_non_planar = True
        self._toggle_map = True
        self._global_view = False

        # Dữ liệu
        self._trajectory = []
        self._last_pose = np.eye(4)
        self._vis_infos = {}
        self._selected_pose = ""
        
        self._initialize_visualizer()

    # --- HÀM UPDATE MỚI (NHẬN 4 LOẠI DỮ LIỆU HÌNH HỌC) ---
    def update(self, source: np.ndarray, planar: np.ndarray, non_planar: np.ndarray, local_map: np.ndarray, pose: np.ndarray, vis_infos: dict = None):
        """
        source: Raw Frame (Toàn cảnh)
        planar: Điểm mặt phẳng
        non_planar: Điểm không phẳng
        local_map: Bản đồ
        pose: Vị trí xe
        """
        if vis_infos is not None:
            self._vis_infos = dict(sorted(vis_infos.items(), key=lambda item: len(item[0])))
        
        self._update_geometries(source, planar, non_planar, local_map, pose)
        self._last_pose = pose

        while self._block_execution:
            self._ps.frame_tick()
            if self._play_mode:
                break
        self._block_execution = not self._block_execution

    def close(self):
        self._ps.unshow()

    # --- PRIVATE METHODS ---

    def _initialize_visualizer(self):
        self._ps.set_program_name("GenZ-ICP Visualizer")
        self._ps.init()
        self._ps.set_ground_plane_mode("none")
        self._ps.set_background_color(BACKGROUND_COLOR)
        self._ps.set_verbosity(0)
        self._ps.set_user_callback(self._main_gui_callback)
        self._ps.set_build_default_gui_panels(False)

    def _update_geometries(self, source, planar, non_planar, target_map, pose):
        # 0. SOURCE / RAW FRAME (MÀU ĐỎ HỒNG)
        frame_cloud = self._ps.register_point_cloud("current_frame", source, color=FRAME_COLOR, point_render_mode="quad")
        frame_cloud.set_radius(self._frame_size, relative=False)
        if self._global_view: frame_cloud.set_transform(pose)
        else: frame_cloud.set_transform(np.eye(4))
        frame_cloud.set_enabled(self._toggle_frame)

        # 1. PLANAR POINTS (MÀU XANH DƯƠNG)
        planar_cloud = self._ps.register_point_cloud("planar_points", planar, color=PLANAR_COLOR, point_render_mode="quad")
        planar_cloud.set_radius(self._planar_size, relative=False)
        if self._global_view: planar_cloud.set_transform(pose)
        else: planar_cloud.set_transform(np.eye(4))
        planar_cloud.set_enabled(self._toggle_planar)

        # 2. NON-PLANAR POINTS (MÀU VÀNG)
        non_planar_cloud = self._ps.register_point_cloud("non_planar_points", non_planar, color=NON_PLANAR_COLOR, point_render_mode="quad")
        non_planar_cloud.set_radius(self._non_planar_size, relative=False)
        if self._global_view: non_planar_cloud.set_transform(pose)
        else: non_planar_cloud.set_transform(np.eye(4))
        non_planar_cloud.set_enabled(self._toggle_non_planar)

        # 3. LOCAL MAP (MÀU XANH LÁ ĐẬM)
        map_cloud = self._ps.register_point_cloud("local_map", target_map, color=LOCAL_MAP_COLOR, point_render_mode="quad")
        map_cloud.set_radius(self._map_size, relative=False)
        if self._global_view: map_cloud.set_transform(np.eye(4))
        else: map_cloud.set_transform(np.linalg.inv(pose))
        map_cloud.set_enabled(self._toggle_map)

        # 4. TRAJECTORY (MÀU ĐỎ)
        self._trajectory.append(pose[:3, 3])
        if self._global_view:
            self._register_trajectory()

    def _register_trajectory(self):
        if len(self._trajectory) > 0:
            traj_arr = np.asarray(self._trajectory)
            trajectory_cloud = self._ps.register_point_cloud("trajectory", traj_arr, color=TRAJECTORY_COLOR)
            trajectory_cloud.set_radius(0.3, relative=False)

    def _unregister_trajectory(self):
        self._ps.remove_point_cloud("trajectory")

    # --- GUI CALLBACKS ---
    def _main_gui_callback(self):
        self._start_pause_callback()
        if not self._play_mode:
            self._gui.SameLine()
            self._next_frame_callback()
        self._gui.SameLine()
        self._screenshot_callback()
        self._gui.Separator()
        self._vis_infos_callback()
        self._gui.Separator()
        self._toggle_buttons_andslides_callback()
        self._background_color_callback()
        self._global_view_callback()
        self._gui.SameLine()
        self._center_viewpoint_callback()
        self._gui.Separator()
        self._quit_callback()

    def _start_pause_callback(self):
        button_name = PAUSE_BUTTON if self._play_mode else START_BUTTON
        if self._gui.Button(button_name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Space):
            self._play_mode = not self._play_mode

    def _next_frame_callback(self):
        if self._gui.Button(NEXT_FRAME_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_N):
            self._block_execution = False

    def _screenshot_callback(self):
        if self._gui.Button(SCREENSHOT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_S):
            fn = "genz_shot_" + (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg")
            self._ps.screenshot(fn)

    def _vis_infos_callback(self):
        if self._gui.TreeNodeEx("Odometry Info", self._gui.ImGuiTreeNodeFlags_DefaultOpen):
            for key, val in self._vis_infos.items():
                self._gui.TextUnformatted(f"{key}: {val}")
            self._gui.TreePop()

    def _center_viewpoint_callback(self):
        if self._gui.Button(CENTER_VIEWPOINT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_C):
            self._ps.reset_camera_to_home_view()

    def _toggle_buttons_andslides_callback(self):
        # 0. SOURCE (Mới thêm)
        changed, self._frame_size = self._gui.SliderFloat("##frame", self._frame_size, 0.01, 0.6)
        if changed: self._ps.get_point_cloud("current_frame").set_radius(self._frame_size, relative=False)
        self._gui.SameLine(); changed, self._toggle_frame = self._gui.Checkbox("Source (Red)", self._toggle_frame)
        if changed: self._ps.get_point_cloud("current_frame").set_enabled(self._toggle_frame)

        # 1. PLANAR
        changed, self._planar_size = self._gui.SliderFloat("##planar", self._planar_size, 0.01, 0.6)
        if changed: self._ps.get_point_cloud("planar_points").set_radius(self._planar_size, relative=False)
        self._gui.SameLine(); changed, self._toggle_planar = self._gui.Checkbox("Planar (Blue)", self._toggle_planar)
        if changed: self._ps.get_point_cloud("planar_points").set_enabled(self._toggle_planar)

        # 2. NON-PLANAR
        changed, self._non_planar_size = self._gui.SliderFloat("##nonplanar", self._non_planar_size, 0.01, 0.6)
        if changed: self._ps.get_point_cloud("non_planar_points").set_radius(self._non_planar_size, relative=False)
        self._gui.SameLine(); changed, self._toggle_non_planar = self._gui.Checkbox("Non-Planar (Yellow)", self._toggle_non_planar)
        if changed: self._ps.get_point_cloud("non_planar_points").set_enabled(self._toggle_non_planar)

        # 3. MAP
        changed, self._map_size = self._gui.SliderFloat("##map", self._map_size, 0.01, 0.6)
        if changed: self._ps.get_point_cloud("local_map").set_radius(self._map_size, relative=False)
        self._gui.SameLine(); changed, self._toggle_map = self._gui.Checkbox("Map (Green)", self._toggle_map)
        if changed: self._ps.get_point_cloud("local_map").set_enabled(self._toggle_map)

    def _background_color_callback(self):
        changed, self._background_color = self._gui.ColorEdit3("Bg Color", self._background_color)
        if changed: self._ps.set_background_color(self._background_color)

    def _global_view_callback(self):
        name = LOCAL_VIEW_BUTTON if self._global_view else GLOBAL_VIEW_BUTTON
        if self._gui.Button(name) or self._gui.IsKeyPressed(self._gui.ImGuiKey_G):
            self._global_view = not self._global_view
            inv_pose = np.linalg.inv(self._last_pose)
            if self._global_view:
                self._ps.get_point_cloud("current_frame").set_transform(self._last_pose)
                self._ps.get_point_cloud("planar_points").set_transform(self._last_pose)
                self._ps.get_point_cloud("non_planar_points").set_transform(self._last_pose)
                self._ps.get_point_cloud("local_map").set_transform(np.eye(4))
                self._register_trajectory()
            else:
                self._ps.get_point_cloud("current_frame").set_transform(np.eye(4))
                self._ps.get_point_cloud("planar_points").set_transform(np.eye(4))
                self._ps.get_point_cloud("non_planar_points").set_transform(np.eye(4))
                self._ps.get_point_cloud("local_map").set_transform(inv_pose)
                self._unregister_trajectory()
            self._ps.reset_camera_to_home_view()

    def _quit_callback(self):
        self._gui.SetCursorPosX(self._gui.GetCursorPosX() + self._gui.GetContentRegionAvail()[0] - 50)
        if self._gui.Button(QUIT_BUTTON) or self._gui.IsKeyPressed(self._gui.ImGuiKey_Q):
            self._ps.unshow()
            os._exit(0)
# ___________________BO___________________________
# from __future__ import annotations

# import importlib
# from abc import ABC
# from typing import Optional

# import numpy as np


# class StubVisualizer(ABC):
#     def update(self, source: np.ndarray, local_map: np.ndarray, pose: np.ndarray):
#         return

#     def close(self):
#         return


# class RegistrationVisualizer(StubVisualizer):
#     def __init__(self):
#         try:
#             self.o3d = importlib.import_module("open3d")
#         except ModuleNotFoundError as exc:
#             raise ModuleNotFoundError(
#                 'open3d is required for visualization. Install with: pip install "genz-icp[visualizer]"'
#             ) from exc

#         self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
#         self.vis.create_window(window_name="GenZ-ICP Visualizer", width=1600, height=900)

#         self._map = self.o3d.geometry.PointCloud()
#         self._scan = self.o3d.geometry.PointCloud()
#         self._trajectory = self.o3d.geometry.LineSet()

#         self._poses_xyz = []
#         self._initialized = False

#     def _set_points(self, pcd, points: np.ndarray, color: Optional[np.ndarray] = None):
#         pcd.points = self.o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
#         if color is not None:
#             pcd.paint_uniform_color(np.asarray(color, dtype=np.float64))

#     def update(self, source: np.ndarray, local_map: np.ndarray, pose: np.ndarray):
#         source = np.asarray(source, dtype=np.float64)
#         local_map = np.asarray(local_map, dtype=np.float64)
#         pose = np.asarray(pose, dtype=np.float64)

#         self._set_points(self._map, local_map, color=np.array([1.0, 1.0, 0.0]))

#         if source.size:
#             scan_global = source @ pose[:3, :3].T + pose[:3, 3]
#         else:
#             scan_global = source
#         self._set_points(self._scan, scan_global, color=np.array([0.1, 0.8, 0.1]))

#         self._poses_xyz.append(pose[:3, 3].copy())
#         if len(self._poses_xyz) >= 2:
#             points = np.asarray(self._poses_xyz)
#             lines = [[idx, idx + 1] for idx in range(len(points) - 1)]
#             self._trajectory.points = self.o3d.utility.Vector3dVector(points)
#             self._trajectory.lines = self.o3d.utility.Vector2iVector(lines)
#             self._trajectory.colors = self.o3d.utility.Vector3dVector(
#                 np.tile(np.array([[1.0, 0.1, 0.1]]), (len(lines), 1))
#             )

#         if not self._initialized:
#             self.vis.add_geometry(self._map)
#             self.vis.add_geometry(self._scan)
#             self.vis.add_geometry(self._trajectory)
#             self.vis.get_render_option().background_color = [0.0, 0.0, 0.0]
#             self.vis.get_render_option().point_size = 2.0
#             self._initialized = True
#         else:
#             self.vis.update_geometry(self._map)
#             self.vis.update_geometry(self._scan)
#             self.vis.update_geometry(self._trajectory)

#         self.vis.poll_events()
#         self.vis.update_renderer()

#     def close(self):
#         if hasattr(self, "vis"):
#             self.vis.destroy_window()

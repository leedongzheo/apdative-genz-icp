#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <vector>

#include "genz_icp/core/Preprocessing.hpp"
#include "genz_icp/metrics/Metrics.hpp"
#include "genz_icp/pipeline/GenZICP.hpp"
#include "stl_vector_eigen.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace genz_icp {
PYBIND11_MODULE(genz_icp_pybind, m) {
    pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_Vector3dVector", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    py::class_<pipeline::GenZConfig>(m, "_GenZConfig")
        .def(py::init<>())
        .def_readwrite("max_range", &pipeline::GenZConfig::max_range)
        .def_readwrite("min_range", &pipeline::GenZConfig::min_range)
        .def_readwrite("map_cleanup_radius", &pipeline::GenZConfig::map_cleanup_radius)
        .def_readwrite("max_points_per_voxel", &pipeline::GenZConfig::max_points_per_voxel)
        .def_readwrite("voxel_size", &pipeline::GenZConfig::voxel_size)
        .def_readwrite("desired_num_voxelized_points", &pipeline::GenZConfig::desired_num_voxelized_points)
        .def_readwrite("min_motion_th", &pipeline::GenZConfig::min_motion_th)
        .def_readwrite("initial_threshold", &pipeline::GenZConfig::initial_threshold)
        .def_readwrite("planarity_threshold", &pipeline::GenZConfig::planarity_threshold)
        .def_readwrite("deskew", &pipeline::GenZConfig::deskew)
        .def_readwrite("max_num_iterations", &pipeline::GenZConfig::max_num_iterations)
        .def_readwrite("convergence_criterion", &pipeline::GenZConfig::convergence_criterion)
        // === [THÊM CÁC DÒNG NÀY] ===
        .def_readwrite("use_adaptive_planarity", &pipeline::GenZConfig::use_adaptive_planarity)
        .def_readwrite("adaptive_threshold_base", &pipeline::GenZConfig::adaptive_threshold_base)
        .def_readwrite("min_adaptive_threshold", &pipeline::GenZConfig::min_adaptive_threshold)
        .def_readwrite("max_adaptive_threshold", &pipeline::GenZConfig::max_adaptive_threshold);
        // ===========================

    py::class_<pipeline::GenZICP>(m, "_GenZICP")
        .def(py::init<const pipeline::GenZConfig &>(), "config"_a)
        .def(py::init<>())
        .def("_register_frame", py::overload_cast<const std::vector<Eigen::Vector3d> &>(&pipeline::GenZICP::RegisterFrame), "frame"_a)
        .def("_register_frame", py::overload_cast<const std::vector<Eigen::Vector3d> &, const std::vector<double> &>(&pipeline::GenZICP::RegisterFrame), "frame"_a, "timestamps"_a)
        .def("_local_map", &pipeline::GenZICP::LocalMap)
        .def("_poses", [](const pipeline::GenZICP &self) {
            std::vector<Eigen::Matrix4d> poses;
            for (const auto &pose : self.poses()) poses.emplace_back(pose.matrix());
            return poses;
        })
        .def("_last_pose", [](const pipeline::GenZICP &self) {
            const auto poses = self.poses();
            return poses.empty() ? Eigen::Matrix4d::Identity() : poses.back().matrix();
        })
        .def("_voxel_down_sample", &pipeline::GenZICP::Voxelize, "frame"_a, "voxel_size"_a);

    m.def("_voxel_down_sample", &VoxelDownsample, "frame"_a, "voxel_size"_a);
    m.def("_correct_kitti_scan", &CorrectKITTIScan, "frame"_a);
    m.def("_kitti_seq_error", &metrics::SeqError, "gt_poses"_a, "results_poses"_a);
    m.def("_absolute_trajectory_error", &metrics::AbsoluteTrajectoryError, "gt_poses"_a,
          "results_poses"_a);
}
}  // namespace genz_icp

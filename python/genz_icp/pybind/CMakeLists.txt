pybind11_add_module(genz_icp_pybind MODULE genz_icp_pybind.cpp)
target_link_libraries(genz_icp_pybind PRIVATE core metrics pipeline)
install(TARGETS genz_icp_pybind DESTINATION .)

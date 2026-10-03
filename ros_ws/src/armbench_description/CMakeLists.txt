cmake_minimum_required(VERSION 3.16)
project(armbench_description)

find_package(ament_cmake REQUIRED)

# World files: the template (ARMBENCH_SCENE marker intact) is installed as-is and also as
# `tabletop.sdf` (no cubes). `check_scene.sdf` = template + the fixed F1 self-check fixture.
file(READ worlds/tabletop.sdf.in _tpl)
file(READ worlds/check_scene.sdf.in _fixture)
string(REPLACE "    <!-- ARMBENCH_SCENE -->\n" "${_fixture}" _check_world "${_tpl}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/worlds/check_scene.sdf "${_check_world}")

install(DIRECTORY urdf config DESTINATION share/${PROJECT_NAME})
install(FILES worlds/tabletop.sdf.in worlds/check_scene.sdf.in
        ${CMAKE_CURRENT_BINARY_DIR}/worlds/check_scene.sdf
        DESTINATION share/${PROJECT_NAME}/worlds)
install(FILES worlds/tabletop.sdf.in RENAME tabletop.sdf DESTINATION share/${PROJECT_NAME}/worlds)

ament_package()

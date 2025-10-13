cmake_minimum_required(VERSION 3.16)
project(QtQuickGraphsApp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Quick Quick3D Graphs Qml)

add_executable(QtQuickGraphsApp
    src/main.cpp
    src/ChartManager.h
    src/ChartManager.cpp
    src/ChartManager2D.h
    src/ChartManager2D.cpp
    src/ChartManager3D.h
    src/ChartManager3D.cpp
    src/qml.qrc
)

target_link_libraries(QtQuickGraphsApp
    PRIVATE Qt6::Quick Qt6::Quick3D Qt6::Graphs Qt6::Qml
)

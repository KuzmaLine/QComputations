# 🎪 QtQuick Graph Viewer

You know what this is

---

1. **Open Folder**
   - Click **“Open Folder”** and choose a directory that contains your data.
   - If the folder name ends with **“3D”**, the app automatically switches to 3D mode.

2. **Adjust Settings**
   - Click **“Settings”** to open a pop-up where you can:
     - Change X and Y axis scale.
     - Toggle main grid and sub-ticks.
   - All changes apply instantly to the graph.

3. **Export Image**
   - The app exports both the graph and legend as one image.
   - The image is saved under:
     ```
     <selected folder>/export/<folder_name>_<timestamp>.png
     ```
   - The export directory is created automatically if it doesn’t exist.

---
## 🧱 Build Instructions

### Requirements
- **Qt 6.6+** installed (make sure these modules are available):
  - `Qt Quick`
  - `Qt Quick Controls`
  - `Qt Quick 3D`
  - `Qt Graphs`
- **Vulkan**
- **CMake 3.16+**
- **C++17 compiler**

### Steps
- Configure the project
`cmake -B build -S .`

- Build
`cmake --build build`

- Run
`./build/QtQuickGraphsApp`


### Build and Run script (tested only on MacOS)
```
#!/usr/bin/env bash

BUILD_DIR=build
QT_PREFIX=/opt/homebrew/opt/qt
VULKAN_SDK=/opt/homebrew/opt/molten-vk

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake -DCMAKE_PREFIX_PATH=$QT_PREFIX \
      -DVulkan_INCLUDE_DIR=$VULKAN_SDK/include \
      -DVulkan_LIBRARY=$VULKAN_SDK/lib/libMoltenVK.dylib \
      ../

cmake --build . --config Debug

cd ..

if [ -d "$BUILD_DIR/QtQuickGraphsApp.app" ]; then
    open "$BUILD_DIR/QtQuickGraphsApp.app"
else
    "$BUILD_DIR/QtQuickGraphsApp"
fi
```

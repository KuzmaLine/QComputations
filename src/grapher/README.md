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
- **CMake 3.16+**
- **C++17 compiler**

### Steps
# Configure the project
cmake -B build -S .

# Build
cmake --build build

# Run
./build/QtQuickGraphsApp


### Macos-Specific hack for Qt build 
```
cd build
cmake -DQt6Dir=~/opt/homebrew/Cellar/qt/<YOUR VERSION>/lib/cmake/Qt6 -DCMAKE_PREFIX_PATH=~/Qt/<YOUR VERSION>/macos/ ../
make
cd ..
./build/QtQuickGraphsApp
```

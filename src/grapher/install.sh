#!/usr/bin/env bash

BUILD_DIR=build
QT_PREFIX=~/Qt

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake -DCMAKE_PREFIX_PATH=$QT_PREFIX \
      ../

cmake --build . --config Debug

cd ..

if [ -d "$BUILD_DIR/QtQuickGraphsApp.app" ]; then
    open "$BUILD_DIR/QtQuickGraphsApp.app"
else
    "$BUILD_DIR/QtQuickGraphsApp"
fi
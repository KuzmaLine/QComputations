import QtQuick
import QtGraphs

Item {
    id: graph3DRoot
    anchors.top: parent.top
    anchors.left: parent.left
    anchors.bottom: parent.bottom
    anchors.right: parent.right
    visible: root.show3D

    Loader {
        id: surfaceLoader
        anchors.fill: parent
        active: root.show3D
        sourceComponent: Surface3D {
            axisX: Value3DAxis {
                labels: "X Axis"
            }
            axisY: Value3DAxis {
                labels: "Y Axis"
            }
            axisZ: Value3DAxis {
                labels: "Z Axis"
            }
        }

        onLoaded: {
            console.log("[graph3DRoot] Loader loaded Surface3D");
            update3DSurfaceSeries();
        }

        function update3DSurfaceSeries() {
            if (!surfaceLoader.item) {
                console.warn("[graph3DRoot] update3DSurfaceSeries: loader.item is null");
                return;
            }
            if (!chartManager.surfaceSeries) {
                console.warn("[graph3DRoot] update3DSurfaceSeries: chartManager.surfaceSeries is null");
                return;
            }

            var seriesList = surfaceLoader.item.seriesList;
            console.log("[graph3DRoot] Current seriesList length:", seriesList.length);

            // Hide all existing series
            for (var i = 0; i < seriesList.length; i++) {
                console.log("[graph3DRoot] Hiding series:", seriesList[i].name);
                seriesList[i].visible = false;
            }

            // Add our surfaceSeries if not present
            if (!seriesList.includes(chartManager.surfaceSeries)) {
                console.log("[graph3DRoot] Adding surfaceSeries:", chartManager.surfaceSeries.name);
                surfaceLoader.item.addSeries(chartManager.surfaceSeries);
            }

            chartManager.surfaceSeries.visible = true;
            console.log("[graph3DRoot] SurfaceSeries visible set to true");
        }
    }

    Connections {
        target: chartManager

        function onSurfaceSeriesChanged() {
            console.log("[graph3DRoot] surfaceSeriesChanged signal received");
            Qt.callLater(() => surfaceLoader.update3DSurfaceSeries());
        }
    }

    onVisibleChanged: console.log("[graph3DRoot] visible changed:", visible)
}

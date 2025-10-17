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

            theme: GraphsTheme {
                colorScheme: Theme.dark ? GraphsTheme.Theme.QtGreen : GraphsTheme.Theme.QtGreenNeon
            }
        }

        onLoaded: {
            console.log("[Graph2DView] Loader loaded Surface3D");
            update3DSurfaceSeries();
        }

        function update3DSurfaceSeries() {
            if (!surfaceLoader.item) {
                console.warn("[Graph2DView] update3DSurfaceSeries: loader.item is null");
                return;
            }
            if (!chartManager.surfaceSeries) {
                console.warn("[Graph2DView] update3DSurfaceSeries: chartManager.surfaceSeries is null");
                return;
            }

            var seriesList = surfaceLoader.item.seriesList;
            console.log("[Graph2DView] Current seriesList length:", seriesList.length);

            // Hide all existing series
            for (var i = 0; i < seriesList.length; i++) {
                console.log("[Graph2DView] Hiding series:", seriesList[i].name);
                seriesList[i].visible = false;
            }

            // Add our surfaceSeries if not present
            if (!seriesList.includes(chartManager.surfaceSeries)) {
                console.log("[Graph2DView] Adding surfaceSeries:", chartManager.surfaceSeries.name);
                surfaceLoader.item.addSeries(chartManager.surfaceSeries);
            }

            chartManager.surfaceSeries.visible = true;
            console.log("[Graph2DView] SurfaceSeries visible set to true");
        }
    }

    function updateTheme(palette) {
        if (!surfaceLoader.item)
            return;
        surfaceLoader.item.backgroundColor = palette.window;
        surfaceLoader.item.axisX.labelsColor = palette.windowText;
        surfaceLoader.item.axisY.labelsColor = palette.windowText;
        surfaceLoader.item.axisZ.labelsColor = palette.windowText;

        if (chartManager.surfaceSeries) {
            chartManager.surfaceSeries.baseColor = palette.highlight;
        }
    }
    Connections {
        target: chartManager

        function onSurfaceSeriesChanged() {
            console.log("[Graph2DView] surfaceSeriesChanged signal received");
            Qt.callLater(() => surfaceLoader.update3DSurfaceSeries());
        }
    }

    onVisibleChanged: console.log("[Graph2DView] visible changed:", visible)
}

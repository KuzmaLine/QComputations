import QtQuick
import QtGraphs

Item {
    id: graph2DRoot
    anchors.fill: parent
    visible: !root.show3D

    // 🔹 These are controlled externally (e.g. by a popup)
    property real xScale: 1.0
    property real yScale: 1.0
    property bool gridVisible: true
    property bool showSubTicks: false

    property real paddingFactor: 0.05

    GraphsView {
        id: graphView
        anchors.fill: parent

        theme: GraphsTheme {
            colorScheme: GraphsTheme.Theme.QtGreenNeon
        }

        axisX: ValueAxis {
            id: axisX
            titleText: "X Axis"
            min: chartManager.minX
            max: chartManager.maxX * graph2DRoot.xScale * (1 + graph2DRoot.paddingFactor)
            gridVisible: graph2DRoot.gridVisible
            subTickCount: graph2DRoot.showSubTicks ? 4 : 0
        }

        axisY: ValueAxis {
            id: axisY
            titleText: "Y Axis"
            min: chartManager.minY
            max: chartManager.maxY * graph2DRoot.yScale * (1 + graph2DRoot.paddingFactor)
            gridVisible: graph2DRoot.gridVisible
            subTickCount: graph2DRoot.showSubTicks ? 4 : 0
        }

        function updateAxisRanges() {
            axisX.min = chartManager.minX;
            axisX.max = chartManager.maxX * graph2DRoot.xScale * (1 + graph2DRoot.paddingFactor);
            axisY.min = chartManager.minY;
            axisY.max = chartManager.maxY * graph2DRoot.yScale * (1 + graph2DRoot.paddingFactor);
        }

        Component.onCompleted: {
            console.log("[Graph2DView] onCompleted: initializing 2D graph");

            var list = chartManager.lineSeriesList;
            if (!list) {
                console.log("[Graph2DView] WARNING: lineSeriesList is null or undefined");
                return;
            }

            console.log("[Graph2DView] list length:", list.length);
            for (var i = 0; i < list.length; i++) {
                var s = list[i];
                if (s) {
                    graphView.addSeries(s);
                    console.log("[Graph2DView] Added existing series:", s.name);
                }
            }
            updateAxisRanges();
        }

        Connections {
            target: chartManager
            function onLineSeriesAdded(series) {
                if (series)
                    graphView.addSeries(series);
                updateAxisRanges();
            }
            function onLineSeriesRemoved(series) {
                if (series)
                    graphView.removeSeries(series);
                updateAxisRanges();
            }
        }
    }
}

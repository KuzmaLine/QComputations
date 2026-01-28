import QtQuick
import QtGraphs

Item {
    id: graph2DRoot
    anchors.fill: parent
    visible: !root.show3D

    property real xScale: settingsPopup.xScale
    property real yScale: settingsPopup.yScale
    property bool gridVisible: settingsPopup.showGrid
    property bool showSubTicks: settingsPopup.showSubTicks

    property real paddingFactor: 0.05
    property bool darkTheme: false

    GraphsView {
        id: graphView
        anchors.fill: parent

        theme: GraphsTheme {
            colorScheme: darkTheme ? GraphsTheme.Theme.QtGreen : GraphsTheme.Theme.QtGreenNeon
        }

        axisX: ValueAxis {
            id: axisX
            titleText: settingsPopup.xAxisName
            min: chartManager.minX
            max: chartManager.maxX * graph2DRoot.xScale * (1 + graph2DRoot.paddingFactor)
            gridVisible: graph2DRoot.gridVisible
            subTickCount: graph2DRoot.showSubTicks ? 4 : 0
        }

        axisY: ValueAxis {
            id: axisY
            titleText: settingsPopup.yAxisName
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
            console.log("[Graph2DView] updateAxisRanges: X[" + axisX.min + ", " + axisX.max + "], Y[" + axisY.min + ", " + axisY.max + "]");
            console.log("[Graph2DView] xScale:", graph2DRoot.xScale, "yScale:", graph2DRoot.yScale);
        }

        Component.onCompleted: {
            console.log("[Graph2DView] onCompleted: initializing 2D graph");

            var list = chartManager.lineSeriesList;
            if (!list) {
                console.warn("[Graph2DView] lineSeriesList is null or undefined");
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
            }
            function onLineSeriesRemoved(series) {
                if (series)
                    graphView.removeSeries(series);
            }
            function onMinMaxValuesChanged() {
                graphView.updateAxisRanges();
            }
        }
        Connections {
            target: settingsPopup
            function onXScaleChanged() {
                graphView.updateAxisRanges();
            }
            function onYScaleChanged() {
                graphView.updateAxisRanges();
            }
            function onShowGridChanged() {
                axisX.gridVisible = settingsPopup.showGrid;
                axisY.gridVisible = settingsPopup.showGrid;
            }
            function onShowSubTicksChanged() {
                axisX.subTickCount = settingsPopup.showSubTicks ? 4 : 0;
                axisY.subTickCount = settingsPopup.showSubTicks ? 4 : 0;
            }
        }
    }
}

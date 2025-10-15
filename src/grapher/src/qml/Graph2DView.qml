import QtQuick
import QtGraphs
import QtQuick.Controls

Item {
    id: graph2DRoot
    anchors.fill: parent
    visible: !root.show3D

    property real xScale: 1
    property real yScale: 1
    property bool gridVisible: true
    property bool showSubTicks: true

    property real paddingFactor: 0.05
    property bool darkTheme: false

    onXScaleChanged: graphView.updateAxisRanges()
    onYScaleChanged: graphView.updateAxisRanges()

    GraphsView {
        id: graphView
        anchors.fill: parent

        theme: GraphsTheme {
            colorScheme: darkTheme ? GraphsTheme.Theme.QtGreen : GraphsTheme.Theme.QtGreenNeon
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
    }

    //MouseWheel - oriented zooming
    //Wheel for vertical zoom
    //Shift + Wheel for horizontal zoom
    //Ctrl + Wheel for both axes zoom
    //Horizontal wheel (or trackpad) for horizontal zoom
    //TODO: invert wheel direction?
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.NoButton
        hoverEnabled: true
        property real scrollIncrement: 0.05

        onWheel: function (event) {
            //for regular vertical wheel
            if (event.modifiers === Qt.ControlModifier) {
                console.log("Ctrl + Wheel: Zooming both axes");
                xScale *= (event.angleDelta.y > 0) ? (1 + scrollIncrement) : (1 - scrollIncrement);
                yScale *= (event.angleDelta.y > 0) ? (1 + scrollIncrement) : (1 - scrollIncrement);
            } else if (event.modifiers === Qt.ShiftModifier) {
                xScale *= (event.angleDelta.y > 0) ? (1 + scrollIncrement) : (1 - scrollIncrement);
            } else {
                yScale *= (event.angleDelta.y > 0) ? (1 + scrollIncrement) : (1 - scrollIncrement);
            }

            //for horizontal wheel (also rorks for touchpad horizontal scroll)
            if (event.modifiers === Qt.NoModifier & event.angleDelta.x !== 0) {
                xScale *= (event.angleDelta.x > 0) ? (1 + scrollIncrement) : (1 - scrollIncrement);
            }
        }
    }
}

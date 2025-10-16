import QtQuick
import QtQuick.Controls
import QtGraphs

Item {
    id: graph2DRoot
    anchors.fill: parent
    visible: !root.show3D
    property bool darkTheme: false

    GraphsView {
        id: graphView
        anchors.fill: parent

        theme: GraphsTheme {
            colorScheme: darkTheme ? GraphsTheme.Theme.QtGreen : GraphsTheme.Theme.QtGreenNeon
        }

        axisX: ValueAxis {
            id: axisX
            titleText: "X Axis"
        }
        axisY: ValueAxis {
            id: axisY
            titleText: "Y Axis"
        }

        function updateAxisRanges() {
            if (!Graph2DState.initialized)
                Graph2DState.setFromChart(chartManager);
            Graph2DState.applyToChart(graphView, chartManager.minX, chartManager.maxX, chartManager.minY, chartManager.maxY);
        }

        Component.onCompleted: {
            if (!chartManager || !chartManager.lineSeriesList)
                return;
            chartManager.lineSeriesList.forEach(s => {
                if (s)
                    graphView.addSeries(s);
            });
            Graph2DState.setFromChart(chartManager);
            updateAxisRanges();
        }
    }

    Timer {
        id: inertiaTimer
        interval: 16
        repeat: true
        running: false
        onTriggered: {
            Graph2DState.panOffsetX += dragHandler.velocityX;
            Graph2DState.panOffsetY += dragHandler.velocityY;
            Graph2DState.applyToChart(graphView, chartManager.minX, chartManager.maxX, chartManager.minY, chartManager.maxY);

            const friction = 0.92;
            dragHandler.velocityX *= friction;
            dragHandler.velocityY *= friction;

            if (Math.abs(dragHandler.velocityX) < 0.1 && Math.abs(dragHandler.velocityY) < 0.1) {
                inertiaTimer.stop();
            }
        }
    }

    // --- Mouse / touch panning ---
    DragHandler {
        id: dragHandler
        target: graphView
        acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad | PointerDevice.TouchScreen

        property real startPanX: 0
        property real startPanY: 0
        property real prevTranslationX: 0
        property real prevTranslationY: 0
        property real velocityX: 0
        property real velocityY: 0

        // Timer for deceleration

        onActiveChanged: {
            if (active) {
                startPanX = Graph2DState.panOffsetX;
                startPanY = Graph2DState.panOffsetY;
                prevTranslationX = 0;
                prevTranslationY = 0;
                inertiaTimer.stop();
            } else {
                // Start inertia
                inertiaTimer.start();
            }
        }

        onTranslationChanged: {
            const visibleX = (chartManager.maxX - chartManager.minX) * Graph2DState.xScale * (1 + Graph2DState.paddingFactor);
            const visibleY = (chartManager.maxY - chartManager.minY) * Graph2DState.yScale * (1 + Graph2DState.paddingFactor);

            // Move graph along with pointer
            Graph2DState.panOffsetX = startPanX + (-translation.x / graphView.width) * visibleX;
            Graph2DState.panOffsetY = startPanY + (translation.y / graphView.height) * visibleY;
            Graph2DState.applyToChart(graphView, chartManager.minX, chartManager.maxX, chartManager.minY, chartManager.maxY);

            // Calculate instantaneous velocity
            velocityX = ((translation.x - prevTranslationX) / graphView.width) * -visibleX;
            velocityY = ((translation.y - prevTranslationY) / graphView.height) * visibleY;

            prevTranslationX = translation.x;
            prevTranslationY = translation.y;
        }
    }

    // --- Mouse wheel zoom ---
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
        property real scrollIncrement: 0.05

        onWheel: function (event) {
            let xFactor = 1, yFactor = 1;
            if (event.modifiers === Qt.ControlModifier)
                xFactor = yFactor = event.angleDelta.y > 0 ? (1 + scrollIncrement) : (1 - scrollIncrement);
            else if (event.modifiers === Qt.ShiftModifier)
                xFactor = event.angleDelta.y > 0 ? (1 + scrollIncrement) : (1 - scrollIncrement);
            else
                yFactor = event.angleDelta.y > 0 ? (1 + scrollIncrement) : (1 - scrollIncrement);

            Graph2DState.applyScaleAndPan(xFactor, yFactor, graphView, chartManager.minX, chartManager.maxX, chartManager.minY, chartManager.maxY);
        }
    }

    // --- Update on chartManager events ---
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

    // --- Update axis ranges on state change ---
    Connections {
        target: Graph2DState
        function onXScaleChanged() {
            graphView.updateAxisRanges();
        }
        function onYScaleChanged() {
            graphView.updateAxisRanges();
        }
    }
}

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
            Graph2DState.applyToChart();
        }

        Component.onCompleted: {
            // Initialize Graph2DState with this chart
            Graph2DState.initialize(graphView);

            // TODO: remove ?
            if (!chartManager || !chartManager.lineSeriesList)
                return;

            chartManager.lineSeriesList.forEach(s => {
                if (s)
                    graphView.addSeries(s);
            });
        }
    }

    // --- Inertia for smooth panning ---
    Timer {
        id: inertiaTimer
        interval: 16
        repeat: true
        running: false
        onTriggered: {
            Graph2DState.panBy(dragHandler.velocityX, dragHandler.velocityY);
            dragHandler.velocityX *= 0.92;
            dragHandler.velocityY *= 0.92;
            if (Math.abs(dragHandler.velocityX) < 0.1 && Math.abs(dragHandler.velocityY) < 0.1)
                inertiaTimer.stop();
        }
    }

    // --- Mouse/touch drag panning ---
    DragHandler {
        id: dragHandler
        target: graphView
        acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad | PointerDevice.TouchScreen

        property real prevX: 0
        property real prevY: 0
        property real velocityX: 0
        property real velocityY: 0

        onActiveChanged: {
            if (active) {
                prevX = translation.x;
                prevY = translation.y;
                inertiaTimer.stop();
            } else {
                inertiaTimer.start();
            }
        }

        onTranslationChanged: {
            const dx = translation.x - prevX;
            const dy = translation.y - prevY;
            Graph2DState.panBy(dx, dy);
            velocityX = dx;
            velocityY = dy;
            prevX = translation.x;
            prevY = translation.y;
        }
    }

    // --- Mouse wheel zooming ---
    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
        property real scrollIncrement: 0.1

        onWheel: function (event) {
            let xFactor = 1, yFactor = 1;
            if (event.modifiers === Qt.ControlModifier)
                xFactor = yFactor = event.angleDelta.y > 0 ? (1 - scrollIncrement) : (1 + scrollIncrement);
            else if (event.modifiers === Qt.ShiftModifier)
                yFactor = event.angleDelta.y > 0 ? (1 - scrollIncrement) : (1 + scrollIncrement);
            else
                xFactor = event.angleDelta.y > 0 ? (1 - scrollIncrement) : (1 + scrollIncrement);

            Graph2DState.scaleBy(xFactor, yFactor);
        }
    }

    // --- Keep axes synced with Graph2DState ---
    Connections {
        target: Graph2DState
        function onXScaleChanged() {
            graphView.updateAxisRanges();
        }
        function onYScaleChanged() {
            graphView.updateAxisRanges();
        }
        function onPanOffsetXChanged() {
            graphView.updateAxisRanges();
        }
        function onPanOffsetYChanged() {
            graphView.updateAxisRanges();
        }
    }
}

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtGraphs 6.3

Item {
    id: graph2DRoot
    anchors.fill: parent
    visible: !root.show3D

    GraphsView {
        id: graphView
        anchors.fill: parent

        theme: GraphsTheme {
            colorScheme: Theme.dark ? GraphsTheme.Theme.QtGreen : GraphsTheme.Theme.QtGreenNeon
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
            axisX.min = Graph2DState.visibleMinX;
            axisX.max = Graph2DState.visibleMaxX;
            axisY.min = Graph2DState.visibleMinY;
            axisY.max = Graph2DState.visibleMaxY;
            axisX.gridVisible = Graph2DState.gridVisible;
            axisY.gridVisible = Graph2DState.gridVisible;
            axisX.subTickCount = Graph2DState.showSubTicks ? 4 : 0;
            axisY.subTickCount = Graph2DState.showSubTicks ? 4 : 0;
        }

        Component.onCompleted: Graph2DState.initialize(graphView)
    }

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

    DragHandler {
        id: dragHandler
        target: graphView
        acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad | PointerDevice.TouchScreen
        enabled: !Graph2DState.uiHovering

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

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
        enabled: !Graph2DState.uiHovering
        property real scrollIncrement: 0.1

        onWheel: function (event) {
            if (Graph2DState.uiHovering)
                return; // prevent zoom when UI is hovered

            // Normalize wheel delta
            let deltaX = event.angleDelta.x / 120;
            let deltaY = event.angleDelta.y / 120;

            const ctrl = (event.modifiers & Qt.ControlModifier) !== 0;
            const shift = (event.modifiers & Qt.ShiftModifier) !== 0;

            // panning (trackpads or horizontal scroll wheels)
            if (!ctrl && !shift && deltaX > deltaY) {
                const xRange = Graph2DState.visibleRangeX();
                const dx = deltaX * xRange * scrollIncrement;
                Graph2DState.panBy(dx, 0);
                event.accepted = true;
                return;
            }

            let xFactor = 1, yFactor = 1;
            if (ctrl && shift) {
                // Ctrl+Shift = zoom both X & Y
                xFactor = yFactor = Math.pow(1 + scrollIncrement, -deltaY);
            } else if (ctrl) {
                // Ctrl = zoom X only
                xFactor = Math.pow(1 + scrollIncrement, -deltaY);
            } else if (shift) {
                // Shift = zoom Y only
                yFactor = Math.pow(1 + scrollIncrement, -deltaY);
            } else {
                // No modifier = pan X
                const xRange = Graph2DState.visibleRangeX();
                const dx = -deltaY * xRange * scrollIncrement;
                Graph2DState.panBy(dx, 0);
                event.accepted = true;
                return;
            }

            Graph2DState.scaleBy(xFactor, yFactor);
            event.accepted = true;
        }
    }

    Connections {
        target: Graph2DState
        function onVisibleMinXChanged() {
            graphView.updateAxisRanges();
        }
        function onVisibleMaxXChanged() {
            graphView.updateAxisRanges();
        }
        function onVisibleMinYChanged() {
            graphView.updateAxisRanges();
        }
        function onVisibleMaxYChanged() {
            graphView.updateAxisRanges();
        }
        function onGridVisibleChanged() {
            graphView.updateAxisRanges();
        }
        function onShowSubTicksChanged() {
            graphView.updateAxisRanges();
        }
    }
}

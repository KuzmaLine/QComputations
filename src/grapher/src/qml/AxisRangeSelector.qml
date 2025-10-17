import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: axisRangeSelector
    property string axis: "x"
    property real currentMin: 0
    property real currentMax: 1
    property bool updatingHandles: false

    signal rangeChanged(real min, real max)

    width: 300
    height: 50

    function axisMin() {
        return axis === "x" ? chartManager.minX : chartManager.minY;
    }
    function axisMax() {
        return axis === "x" ? chartManager.maxX : chartManager.maxY;
    }

    RowLayout {
        anchors.fill: parent
        spacing: 6

        // --- Min field ---
        TextField {
            id: minField
            width: 60
            color: Theme.windowText
            validator: DoubleValidator {
                bottom: axisMin()
                top: currentMax
            }
            text: currentMin.toFixed(2)
            onEditingFinished: {
                focus = false;
                let val = parseFloat(text);
                val = Math.min(Math.max(val, axisMin()), currentMax);
                currentMin = val;
                updateHandles();
                rangeChanged(currentMin, currentMax);
                console.log(`[AxisRangeSelector ${axis}] Min edited →`, val);
            }
        }

        // --- Slider track ---
        Item {
            id: sliderTrack
            Layout.fillWidth: true
            height: 6

            Rectangle {
                anchors.fill: parent
                color: Theme.mid
                radius: 3
            }

            Rectangle {
                id: rangeHighlight
                height: parent.height
                y: 0
                color: Theme.highlight
                radius: 3
            }

            // --- Min handle ---
            Rectangle {
                id: minHandle
                width: 10
                height: 14
                color: Theme.highlight
                radius: 2
                anchors.verticalCenter: parent.verticalCenter

                MouseArea {
                    anchors.fill: parent
                    drag.target: parent
                    drag.axis: Drag.XAxis
                    drag.minimumX: 0
                    drag.maximumX: sliderTrack.width - maxHandle.width - minHandle.width
                    cursorShape: Qt.SizeHorCursor

                    onPressed: {
                        Graph2DState.uiHovering = true;
                        console.log(`[AxisRangeSelector ${axis}] Min handle press`);
                    }
                    onReleased: {
                        Graph2DState.uiHovering = false;
                        console.log(`[AxisRangeSelector ${axis}] Min handle release`);
                    }

                    onPositionChanged: {
                        if (axisRangeSelector.updatingHandles)
                            return;
                        let minV = axisMin();
                        let maxV = axisMax();
                        let trackWidth = sliderTrack.width - minHandle.width;
                        let fraction = minHandle.x / trackWidth;
                        currentMin = minV + fraction * (maxV - minV);
                        currentMin = Math.min(currentMin, currentMax);
                        minField.text = currentMin.toFixed(2);
                        updateRangeHighlight();
                        rangeChanged(currentMin, currentMax);
                    }
                }
            }

            // --- Max handle ---
            Rectangle {
                id: maxHandle
                width: 10
                height: 14
                color: Theme.highlight
                radius: 2
                anchors.verticalCenter: parent.verticalCenter

                MouseArea {
                    anchors.fill: parent
                    drag.target: parent
                    drag.axis: Drag.XAxis
                    drag.minimumX: minHandle.x + minHandle.width
                    drag.maximumX: sliderTrack.width - maxHandle.width
                    cursorShape: Qt.SizeHorCursor

                    onPressed: {
                        Graph2DState.uiHovering = true;
                        console.log(`[AxisRangeSelector ${axis}] Max handle press`);
                    }
                    onReleased: {
                        Graph2DState.uiHovering = false;
                        console.log(`[AxisRangeSelector ${axis}] Max handle release`);
                    }

                    onPositionChanged: {
                        if (axisRangeSelector.updatingHandles)
                            return;
                        let minV = axisMin();
                        let maxV = axisMax();
                        let trackWidth = sliderTrack.width - maxHandle.width;
                        let fraction = maxHandle.x / trackWidth;
                        currentMax = minV + fraction * (maxV - minV);
                        currentMax = Math.max(currentMax, currentMin);
                        maxField.text = currentMax.toFixed(2);
                        updateRangeHighlight();
                        rangeChanged(currentMin, currentMax);
                    }
                }
            }
        }

        // --- Max field ---
        TextField {
            id: maxField
            width: 60
            color: Theme.windowText
            validator: DoubleValidator {
                bottom: currentMin
                top: axisMax()
            }
            text: currentMax.toFixed(2)
            onEditingFinished: {
                focus = false;
                let val = parseFloat(text);
                val = Math.min(Math.max(val, currentMin), axisMax());
                currentMax = val;
                updateHandles();
                rangeChanged(currentMin, currentMax);
                console.log(`[AxisRangeSelector ${axis}] Max edited →`, val);
            }
        }
    }

    // --- helpers ---
    function updateHandles() {
        if (!sliderTrack.width)
            return;
        updatingHandles = true;
        let minV = axisMin();
        let maxV = axisMax();
        let trackWidth = sliderTrack.width - minHandle.width;
        minHandle.x = (currentMin - minV) / (maxV - minV) * trackWidth;
        maxHandle.x = (currentMax - minV) / (maxV - minV) * trackWidth;
        updateRangeHighlight();
        updatingHandles = false;
        console.log(`[AxisRangeSelector ${axis}] Handles updated →`, currentMin, currentMax);
    }

    function updateRangeHighlight() {
        rangeHighlight.x = minHandle.x + minHandle.width / 2;
        rangeHighlight.width = maxHandle.x - minHandle.x;
    }

    Component.onCompleted: {
        if (axis === "x") {
            currentMin = Graph2DState.visibleMinX;
            currentMax = Graph2DState.visibleMaxX;
        } else {
            currentMin = Graph2DState.visibleMinY;
            currentMax = Graph2DState.visibleMaxY;
        }
        console.log(`[AxisRangeSelector ${axis}] initialized with`, currentMin, currentMax);
        updateHandles();
    }

    Connections {
        target: Graph2DState
        function onVisibleMinXChanged() {
            if (axis === "x") {
                currentMin = Graph2DState.visibleMinX;
                minField.text = currentMin.toFixed(2);
                updateHandles();
            }
        }
        function onVisibleMaxXChanged() {
            if (axis === "x") {
                currentMax = Graph2DState.visibleMaxX;
                maxField.text = currentMax.toFixed(2);
                updateHandles();
            }
        }
        function onVisibleMinYChanged() {
            if (axis === "y") {
                currentMin = Graph2DState.visibleMinY;
                minField.text = currentMin.toFixed(2);
                updateHandles();
            }
        }
        function onVisibleMaxYChanged() {
            if (axis === "y") {
                currentMax = Graph2DState.visibleMaxY;
                maxField.text = currentMax.toFixed(2);
                updateHandles();
            }
        }
    }

    onCurrentMinChanged: {
        if (updatingHandles)
            return;
        if (axis === "x")
            Graph2DState.setVisibleRangeX(currentMin, currentMax);
        else
            Graph2DState.setVisibleRangeY(currentMin, currentMax);
    }

    onCurrentMaxChanged: {
        if (updatingHandles)
            return;
        if (axis === "x")
            Graph2DState.setVisibleRangeX(currentMin, currentMax);
        else
            Graph2DState.setVisibleRangeY(currentMin, currentMax);
    }
}

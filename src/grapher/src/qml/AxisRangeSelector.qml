import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: axisRangeSelector
    property string axis: "x"
    property real currentMin: 0
    property real currentMax: 1

    signal rangeChanged(real min, real max)

    width: 300
    height: 50
    property bool updatingHandles: false

    RowLayout {
        anchors.fill: parent
        spacing: 6

        TextField {
            id: minField
            width: 60
            color: Theme.windowText
            validator: DoubleValidator {
                bottom: 0
                top: currentMax
            }
            text: currentMin.toFixed(2)
            onEditingFinished: {
                let val = parseFloat(text);
                if (axis === "x") {
                    if (val > Graph2DState.maxX)
                        val = Graph2DState.maxX;
                    if (val < chartManager.minX)
                        val = chartManager.minX;
                } else {
                    if (val > Graph2DState.maxY)
                        val = Graph2DState.maxY;
                    if (val < chartManager.minY)
                        val = chartManager.minY;
                }
                currentMin = val;
                updateHandles();
                rangeChanged(currentMin, currentMax);
            }
        }

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
                    drag.maximumX: maxHandle.x - parent.width
                    cursorShape: Qt.SizeHorCursor
                    onPositionChanged: {
                        let trackWidth = sliderTrack.width - minHandle.width;
                        let minV, maxV;
                        if (axis === "x") {
                            minV = chartManager.minX;
                            maxV = chartManager.maxX;
                        } else {
                            minV = chartManager.minY;
                            maxV = chartManager.maxY;
                        }
                        currentMin = minV + (parent.x / trackWidth) * (maxV - minV);
                        minField.text = currentMin.toFixed(2);
                        updateRangeHighlight();
                        rangeChanged(currentMin, currentMax);
                    }
                }
            }

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
                    drag.minimumX: minHandle.x + parent.width
                    drag.maximumX: sliderTrack.width - parent.width
                    cursorShape: Qt.SizeHorCursor
                    onPositionChanged: {
                        let trackWidth = sliderTrack.width - minHandle.width;
                        let minV, maxV;
                        if (axis === "x") {
                            minV = chartManager.minX;
                            maxV = chartManager.maxX;
                        } else {
                            minV = chartManager.minY;
                            maxV = chartManager.maxY;
                        }
                        currentMax = minV + (parent.x / trackWidth) * (maxV - minV);
                        maxField.text = currentMax.toFixed(2);
                        updateRangeHighlight();
                        rangeChanged(currentMin, currentMax);
                    }
                }
            }
        }

        TextField {
            id: maxField
            width: 60
            color: Theme.windowText
            validator: DoubleValidator {
                bottom: currentMin
                top: 1e12
            }
            text: currentMax.toFixed(2)
            onEditingFinished: {
                let val = parseFloat(text);
                if (axis === "x") {
                    if (val < Graph2DState.minX)
                        val = Graph2DState.minX;
                    if (val > chartManager.maxX)
                        val = chartManager.maxX;
                } else {
                    if (val < Graph2DState.minY)
                        val = Graph2DState.minY;
                    if (val > chartManager.maxY)
                        val = chartManager.maxY;
                }
                currentMax = val;
                updateHandles();
                rangeChanged(currentMin, currentMax);
            }
        }
    }

    function updateHandles() {
        if (updatingHandles)
            return;
        updatingHandles = true;
        let trackWidth = sliderTrack.width - minHandle.width;
        let minV, maxV;
        if (axis === "x") {
            minV = chartManager.minX;
            maxV = chartManager.maxX;
        } else {
            minV = chartManager.minY;
            maxV = chartManager.maxY;
        }
        minHandle.x = (currentMin - minV) / (maxV - minV) * trackWidth;
        maxHandle.x = (currentMax - minV) / (maxV - minV) * trackWidth;
        updateRangeHighlight();
        updatingHandles = false;
    }

    function updateRangeHighlight() {
        rangeHighlight.x = minHandle.x + minHandle.width / 2;
        rangeHighlight.width = maxHandle.x - minHandle.x;
    }

    Component.onCompleted: updateHandles()

    Connections {
        target: Graph2DState
        function onMinXChanged() {
            if (axis === "x") {
                currentMin = Graph2DState.minX;
                minField.text = currentMin.toFixed(2);
                updateHandles();
            }
        }
        function onMaxXChanged() {
            if (axis === "x") {
                currentMax = Graph2DState.maxX;
                maxField.text = currentMax.toFixed(2);
                updateHandles();
            }
        }
        function onMinYChanged() {
            if (axis === "y") {
                currentMin = Graph2DState.minY;
                minField.text = currentMin.toFixed(2);
                updateHandles();
            }
        }
        function onMaxYChanged() {
            if (axis === "y") {
                currentMax = Graph2DState.maxY;
                maxField.text = currentMax.toFixed(2);
                updateHandles();
            }
        }
    }

    onCurrentMinChanged: {
        if (axis === "x")
            Graph2DState.setBorders(currentMin, Graph2DState.maxX, Graph2DState.minY, Graph2DState.maxY);
        else
            Graph2DState.setBorders(Graph2DState.minX, Graph2DState.maxX, currentMin, Graph2DState.maxY);
    }
    onCurrentMaxChanged: {
        if (axis === "x")
            Graph2DState.setBorders(Graph2DState.minX, currentMax, Graph2DState.minY, Graph2DState.maxY);
        else
            Graph2DState.setBorders(Graph2DState.minX, Graph2DState.maxX, Graph2DState.minY, currentMax);
    }
}

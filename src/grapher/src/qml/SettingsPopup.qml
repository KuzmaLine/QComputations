import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Popup {
    id: settingsPopup
    modal: true
    width: 360
    height: 420
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
    onOpened: {
        Graph2DState.uiHovering = true;
        console.log("[SettingsPopup] opened — initialized:", Graph2DState.initialized);
    }
    onClosed: {
        Graph2DState.uiHovering = false;
        console.log("[SettingsPopup] closed");
    }

    property bool show3D: false

    Rectangle {
        id: popupRect
        anchors.fill: parent
        color: Theme.base
        radius: 8
        border.color: Theme.highlight
        border.width: 1

        ColumnLayout {
            anchors.fill: parent
            spacing: 10
            anchors.margins: 12

            Text {
                text: show3D ? "3D Settings" : "2D Grid & Axis Settings"
                font.bold: true
                font.pointSize: 16
                color: Theme.windowText
                Layout.alignment: Qt.AlignHCenter
            }

            RowLayout {
                spacing: 6
                Label {
                    text: "X Zoom:"
                    width: 60
                    color: Theme.windowText
                }

                Slider {
                    id: xZoomSlider
                    from: 0.1
                    to: 1.2
                    stepSize: 0.001
                    value: 1.0
                    Layout.fillWidth: true

                    property bool updatingFromGraph: false

                    onMoved: {
                        if (!Graph2DState.initialized)
                            return;
                        updatingFromGraph = true;

                        let minX = Graph2DState.visibleMinX;
                        let maxX = Graph2DState.visibleMaxX;
                        let totalRange = Graph2DState.totalMaxX - Graph2DState.totalMinX;
                        let currentWidth = maxX - minX;
                        let newWidth = totalRange * value;

                        // Try expanding to the right
                        let newMax = minX + newWidth;
                        if (newMax > Graph2DState.totalMaxX) {
                            // Can't expand to right, shift to left
                            let shift = newMax - Graph2DState.totalMaxX;
                            minX = Math.max(Graph2DState.totalMinX, minX - shift);
                            newMax = minX + newWidth;
                        }

                        // Ensure we don't go past min bound
                        if (minX < Graph2DState.totalMinX) {
                            minX = Graph2DState.totalMinX;
                            newMax = minX + newWidth;
                        }

                        Graph2DState.setVisibleRangeX(minX, newMax);
                        updatingFromGraph = false;
                    }

                    function updateFromGraph() {
                        if (!Graph2DState.initialized || updatingFromGraph)
                            return;
                        const width = Graph2DState.visibleMaxX - Graph2DState.visibleMinX;
                        const totalRange = Graph2DState.totalMaxX - Graph2DState.totalMinX;
                        const newValue = width / totalRange;
                        if (Math.abs(value - newValue) > 0.001)
                            value = newValue;
                    }

                    Connections {
                        target: Graph2DState
                        function onVisibleMinXChanged() {
                            xZoomSlider.updateFromGraph();
                        }
                        function onVisibleMaxXChanged() {
                            xZoomSlider.updateFromGraph();
                        }
                    }
                }

                Label {
                    text: xZoomSlider.value.toFixed(2) + "×"
                    width: 50
                    color: Theme.windowText
                }
            }

            RowLayout {
                spacing: 6
                Label {
                    text: "Y Zoom:"
                    width: 60
                    color: Theme.windowText
                }

                Slider {
                    id: yZoomSlider
                    from: 0.1
                    to: 1.2
                    stepSize: 0.001
                    value: 1.0
                    Layout.fillWidth: true

                    property bool updatingFromGraph: false

                    onMoved: {
                        if (!Graph2DState.initialized)
                            return;
                        updatingFromGraph = true;

                        let minY = Graph2DState.visibleMinY;
                        let maxY = Graph2DState.visibleMaxY;
                        let totalRange = Graph2DState.totalMaxY - Graph2DState.totalMinY;
                        let currentHeight = maxY - minY;
                        let newHeight = totalRange * value;

                        let newMax = minY + newHeight;
                        if (newMax > Graph2DState.totalMaxY) {
                            let shift = newMax - Graph2DState.totalMaxY;
                            minY = Math.max(Graph2DState.totalMinY, minY - shift);
                            newMax = minY + newHeight;
                        }

                        if (minY < Graph2DState.totalMinY) {
                            minY = Graph2DState.totalMinY;
                            newMax = minY + newHeight;
                        }

                        Graph2DState.setVisibleRangeY(minY, newMax);
                        updatingFromGraph = false;
                    }

                    function updateFromGraph() {
                        if (!Graph2DState.initialized || updatingFromGraph)
                            return;
                        const height = Graph2DState.visibleMaxY - Graph2DState.visibleMinY;
                        const totalRange = Graph2DState.totalMaxY - Graph2DState.totalMinY;
                        const newValue = height / totalRange;
                        if (Math.abs(value - newValue) > 0.001)
                            value = newValue;
                    }

                    Connections {
                        target: Graph2DState
                        function onVisibleMinYChanged() {
                            yZoomSlider.updateFromGraph();
                        }
                        function onVisibleMaxYChanged() {
                            yZoomSlider.updateFromGraph();
                        }
                    }
                }

                Label {
                    text: yZoomSlider.value.toFixed(2) + "×"
                    width: 50
                    color: Theme.windowText
                }
            }

            RowLayout {
                spacing: 12
                CheckBox {
                    text: "Lock X"
                    checked: Graph2DState.lockX
                    onToggled: {
                        Graph2DState.lockX = checked;
                        Graph2DState.applyToChart();
                        console.log("[SettingsPopup] Lock X toggled:", checked);
                    }
                }
                CheckBox {
                    text: "Lock Y"
                    checked: Graph2DState.lockY
                    onToggled: {
                        Graph2DState.lockY = checked;
                        Graph2DState.applyToChart();
                        console.log("[SettingsPopup] Lock Y toggled:", checked);
                    }
                }
            }

            // --- Reset Buttons ---
            RowLayout {
                spacing: 10
                Button {
                    text: "Reset Scaling"
                    onClicked: {
                        console.log("[SettingsPopup] Reset Scaling");
                        Graph2DState.resetScaling();
                    }
                }
                Button {
                    text: "Reset Position"
                    onClicked: {
                        console.log("[SettingsPopup] Reset Position");
                        Graph2DState.resetPosition();
                    }
                }
                Button {
                    text: "Reset All"
                    onClicked: {
                        console.log("[SettingsPopup] Reset All");
                        Graph2DState.resetAll();
                    }
                }
            }

            // --- Grid & Sub-Ticks ---
            RowLayout {
                spacing: 12
                CheckBox {
                    text: "Show Grid"
                    checked: Graph2DState.gridVisible
                    onToggled: {
                        Graph2DState.gridVisible = checked;
                        Graph2DState.applyToChart();
                        console.log("[SettingsPopup] Show Grid:", checked);
                    }
                }
                CheckBox {
                    text: "Show Sub-Ticks"
                    checked: Graph2DState.showSubTicks
                    onToggled: {
                        Graph2DState.showSubTicks = checked;
                        Graph2DState.applyToChart();
                        console.log("[SettingsPopup] Show Sub-Ticks:", checked);
                    }
                }
            }

            // --- Axis range selectors ---
            AxisRangeSelector {
                axis: "x"
            }
            AxisRangeSelector {
                axis: "y"
            }
        }
    }

    Connections {
        target: Graph2DState
        function onInitializedChanged() {
            console.log("[SettingsPopup] Graph2DState initialized changed:", Graph2DState.initialized, "Range X:", Graph2DState.totalMinX, "→", Graph2DState.totalMaxX);
        }
    }
}

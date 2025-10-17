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

            // --- Header ---
            Text {
                text: show3D ? "3D Settings" : "2D Grid & Axis Settings"
                font.bold: true
                font.pointSize: 16
                color: Theme.windowText
                Layout.alignment: Qt.AlignHCenter
            }

            // --- X Zoom Slider ---
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
                    to: 2.0
                    stepSize: 0.001
                    value: 1.0
                    Layout.fillWidth: true

                    property bool updatingFromGraph: false

                    onMoved: {
                        if (!Graph2DState.initialized || updatingFromGraph)
                            return;
                        console.log(`[SettingsPopup] X zoom slider moved → ${value.toFixed(3)}`);

                        // Scale width by factor around visibleMinX
                        const minX = Graph2DState.visibleMinX;
                        const width = Graph2DState.visibleMaxX - minX;
                        const newWidth = width / value;
                        Graph2DState.setVisibleRangeX(minX, minX + newWidth);
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

                    function updateFromGraph() {
                        if (!Graph2DState.initialized)
                            return;
                        updatingFromGraph = true;
                        const minX = Graph2DState.visibleMinX;
                        const width = Graph2DState.visibleMaxX - minX;
                        const totalRange = Graph2DState.totalMaxX - Graph2DState.totalMinX;
                        const newValue = totalRange / width;
                        if (Math.abs(xZoomSlider.value - newValue) > 0.001) {
                            xZoomSlider.value = newValue;
                            console.log(`[SettingsPopup] X zoom updated from visible range → ${newValue.toFixed(3)}`);
                        }
                        updatingFromGraph = false;
                    }
                }

                Label {
                    text: xZoomSlider.value.toFixed(2) + "×"
                    width: 50
                    color: Theme.windowText
                }
            }

            // --- Y Zoom Slider ---
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
                    to: 2.0
                    stepSize: 0.001
                    value: 1.0
                    Layout.fillWidth: true

                    property bool updatingFromGraph: false

                    onMoved: {
                        if (!Graph2DState.initialized || updatingFromGraph)
                            return;
                        console.log(`[SettingsPopup] Y zoom slider moved → ${value.toFixed(3)}`);

                        const minY = Graph2DState.visibleMinY;
                        const height = Graph2DState.visibleMaxY - minY;
                        const newHeight = height / value;
                        Graph2DState.setVisibleRangeY(minY, minY + newHeight);
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

                    function updateFromGraph() {
                        if (!Graph2DState.initialized)
                            return;
                        updatingFromGraph = true;
                        const minY = Graph2DState.visibleMinY;
                        const height = Graph2DState.visibleMaxY - minY;
                        const totalRange = Graph2DState.totalMaxY - Graph2DState.totalMinY;
                        const newValue = totalRange / height;
                        if (Math.abs(yZoomSlider.value - newValue) > 0.001) {
                            yZoomSlider.value = newValue;
                            console.log(`[SettingsPopup] Y zoom updated from visible range → ${newValue.toFixed(3)}`);
                        }
                        updatingFromGraph = false;
                    }
                }

                Label {
                    text: yZoomSlider.value.toFixed(2) + "×"
                    width: 50
                    color: Theme.windowText
                }
            }

            // --- Lock axis ---
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

    // --- watch for state initialization ---
    Connections {
        target: Graph2DState
        function onInitializedChanged() {
            console.log("[SettingsPopup] Graph2DState initialized changed:", Graph2DState.initialized, "Range X:", Graph2DState.totalMinX, "→", Graph2DState.totalMaxX);
        }
    }
}

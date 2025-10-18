import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// TODO: move 3D-specific settings to a separate popup
Popup {
    id: settingsPopup
    modal: true
    width: 300
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
    background: Rectangle {
        color: Theme.base
        radius: 8
        border.color: Theme.highlight
        border.width: 1
    }

    property bool show3D: false

    function movePopup(dx, dy) {
        settingsPopup.x = Math.max(0, Math.min(root.width - settingsPopup.width, settingsPopup.x + dx));
        settingsPopup.y = Math.max(0, Math.min(root.height - settingsPopup.height, settingsPopup.y + dy));
    }
    GridLayout {
        id: mainGrid
        anchors.fill: parent
        anchors.margins: 8
        rowSpacing: 10
        columnSpacing: 8
        columns: 3

        // --- Title row ---
        Rectangle {
            id: titleBar
            color: Theme.mid
            radius: 6
            Layout.columnSpan: 3
            Layout.fillWidth: true
            height: 30

            property real dragStartX: 0
            property real dragStartY: 0

            Text {
                anchors.centerIn: parent
                text: show3D ? "3D Settings" : "2D Grid & Axis Settings"
                font.bold: true
                font.pointSize: 16
                color: Theme.windowText
            }

            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.SizeAllCursor

                onPressed: function (mouse) {
                    titleBar.dragStartX = mouse.x;
                    titleBar.dragStartY = mouse.y;
                }

                onPositionChanged: function (mouse) {
                    // move popup manually
                    settingsPopup.x += mouse.x - titleBar.dragStartX;
                    settingsPopup.y += mouse.y - titleBar.dragStartY;

                    // clamp inside main window
                    settingsPopup.x = Math.max(0, Math.min(root.width - settingsPopup.width, settingsPopup.x));
                    settingsPopup.y = Math.max(0, Math.min(root.height - settingsPopup.height, settingsPopup.y));
                }
            }
        }

        // --- X Zoom ---
        Label {
            text: "X Zoom:"
            color: Theme.windowText
            Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
        }
        Slider {
            id: xZoomSlider
            from: 0.1
            to: 1.05
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
                    let shift = newMax - Graph2DState.totalMaxX;
                    minX = Math.max(Graph2DState.totalMinX, minX - shift);
                    newMax = minX + newWidth;
                }

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
            color: Theme.windowText
            Layout.alignment: Qt.AlignVCenter
        }

        // --- Y Zoom ---
        Label {
            text: "Y Zoom:"
            color: Theme.windowText
            Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
        }
        Slider {
            id: yZoomSlider
            from: 0.1
            to: 1.05
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
            color: Theme.windowText
            Layout.alignment: Qt.AlignVCenter
        }

        // --- Lock toggles ---
        RowLayout {
            Layout.columnSpan: 3
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
            CheckBox {
                text: "Lock X"
                checked: Graph2DState.lockX
                onToggled: {
                    Graph2DState.lockX = checked;
                    Graph2DState.applyToChart();
                    console.log("[SettingsPopup] Lock X toggled:", checked);
                }
                Layout.columnSpan: 1
            }
            CheckBox {
                text: "Lock Y"
                checked: Graph2DState.lockY
                onToggled: {
                    Graph2DState.lockY = checked;
                    Graph2DState.applyToChart();
                    console.log("[SettingsPopup] Lock Y toggled:", checked);
                }
                Layout.columnSpan: 2
            }
        }

        // --- Grid/Sub-ticks checkboxes ---
        RowLayout {
            Layout.columnSpan: 3
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
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
                Layout.columnSpan: 2
            }
        }

        // --- Axis range selectors ---
        AxisRangeSelector {
            axis: "x"
            Layout.columnSpan: 3
            Layout.fillWidth: true
        }
        AxisRangeSelector {
            axis: "y"
            Layout.columnSpan: 3
            Layout.fillWidth: true
        }

        // --- Reset buttons ---
        RowLayout {
            Layout.columnSpan: 3
            Layout.fillWidth: true
            Button {
                text: "Reset Scaling"
                Layout.fillWidth: true
                onClicked: {
                    console.log("[SettingsPopup] Reset Scaling");
                    Graph2DState.resetScaling();
                }
            }
            Button {
                text: "Reset Position"
                Layout.fillWidth: true
                onClicked: {
                    console.log("[SettingsPopup] Reset Position");
                    Graph2DState.resetPosition();
                }
            }
            Button {
                text: "Reset All"
                Layout.fillWidth: true
                onClicked: {
                    console.log("[SettingsPopup] Reset All");
                    Graph2DState.resetAll();
                }
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

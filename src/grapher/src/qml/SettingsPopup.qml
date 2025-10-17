// SettingsPopup.qml
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Popup {
    id: settingsPopup
    modal: true
    width: 360
    height: 420
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

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

            // --- X Scale Slider ---
            RowLayout {
                spacing: 6
                Label {
                    text: "X Scale:"
                    width: 60
                    color: Theme.windowText
                }
                Slider {
                    id: xSlider
                    from: 0.1
                    to: 2.0
                    stepSize: 0.001
                    value: Graph2DState.xScale
                    Layout.fillWidth: true
                    onMoved: {
                        Graph2DState.xScale = value;
                        if (Math.abs(value - 1.0) < 0.005)
                            Graph2DState.xScale = 1.0;
                        Graph2DState.applyToChart();
                    }
                }
                Label {
                    text: Graph2DState.xScale.toFixed(2)
                    width: 40
                    color: Theme.windowText
                }
            }

            // --- Y Scale Slider ---
            RowLayout {
                spacing: 6
                Label {
                    text: "Y Scale:"
                    width: 60
                    color: Theme.windowText
                }
                Slider {
                    id: ySlider
                    from: 0.1
                    to: 2.0
                    stepSize: 0.001
                    value: Graph2DState.yScale
                    Layout.fillWidth: true
                    onMoved: {
                        Graph2DState.yScale = value;
                        if (Math.abs(value - 1.0) < 0.005)
                            Graph2DState.yScale = 1.0;
                        Graph2DState.applyToChart();
                    }
                }
                Label {
                    text: Graph2DState.yScale.toFixed(2)
                    width: 40
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
                    }
                }
                CheckBox {
                    text: "Lock Y"
                    checked: Graph2DState.lockY
                    onToggled: {
                        Graph2DState.lockY = checked;
                        Graph2DState.applyToChart();
                    }
                }
            }

            // --- Reset Buttons ---
            RowLayout {
                spacing: 10
                Button {
                    text: "Reset Scaling"
                    onClicked: Graph2DState.resetScaling()
                }
                Button {
                    text: "Reset Position"
                    onClicked: Graph2DState.resetPosition()
                }
                Button {
                    text: "Reset All"
                    onClicked: Graph2DState.resetAll()
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
                    }
                }
                CheckBox {
                    text: "Show Sub-Ticks"
                    checked: Graph2DState.showSubTicks
                    onToggled: {
                        Graph2DState.showSubTicks = checked;
                        Graph2DState.applyToChart();
                    }
                }
            }

            AxisRangeSelector {
                axis: "x"
            }

            AxisRangeSelector {
                axis: "y"
            }
        }
    }
}

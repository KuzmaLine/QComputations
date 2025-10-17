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
    property Theme theme

    Rectangle {
        id: popupRect
        anchors.fill: parent
        color: theme.base
        radius: 8
        border.color: theme.highlight
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
                color: theme.windowText
                Layout.alignment: Qt.AlignHCenter
            }

            // --- X Scale Slider ---
            RowLayout {
                spacing: 6
                Label {
                    text: "X Scale:"
                    width: 60
                    color: theme.windowText
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
                    }
                }
                Label {
                    text: Graph2DState.xScale.toFixed(2)
                    width: 40
                    color: theme.windowText
                }
            }

            // --- Y Scale Slider ---
            RowLayout {
                spacing: 6
                Label {
                    text: "Y Scale:"
                    width: 60
                    color: theme.windowText
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
                    }
                }
                Label {
                    text: Graph2DState.yScale.toFixed(2)
                    width: 40
                    color: theme.windowText
                }
            }

            // --- Lock axis ---
            RowLayout {
                spacing: 12
                CheckBox {
                    text: "Lock X"
                    checked: Graph2DState.lockX
                    onToggled: Graph2DState.lockX = checked
                }
                CheckBox {
                    text: "Lock Y"
                    checked: Graph2DState.lockY
                    onToggled: Graph2DState.lockY = checked
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
                    onClicked: Graph2DState.reset()
                }
            }

            // --- Grid & Sub-Ticks ---
            RowLayout {
                spacing: 12
                CheckBox {
                    text: "Show Grid"
                    checked: Graph2DState.gridVisible
                    onToggled: Graph2DState.gridVisible = checked
                }
                CheckBox {
                    text: "Show Sub-Ticks"
                    checked: Graph2DState.showSubTicks
                    onToggled: Graph2DState.showSubTicks = checked
                }
            }

            // --- Manual Borders using FloatSpinBox ---
            GroupBox {
                title: "Manual Borders"
                Layout.fillWidth: true
                GridLayout {
                    columns: 2

                    Label {
                        text: "X Min:"
                        color: theme.windowText
                    }
                    FloatSpinBox {
                        axis: "x"
                        isMin: true
                    }

                    Label {
                        text: "X Max:"
                        color: theme.windowText
                    }
                    FloatSpinBox {
                        axis: "x"
                        isMin: false
                    }

                    Label {
                        text: "Y Min:"
                        color: theme.windowText
                    }
                    FloatSpinBox {
                        axis: "y"
                        isMin: true
                    }

                    Label {
                        text: "Y Max:"
                        color: theme.windowText
                    }
                    FloatSpinBox {
                        axis: "y"
                        isMin: false
                    }
                }
            }
        }
    }
}

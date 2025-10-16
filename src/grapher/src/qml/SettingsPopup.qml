import QtQuick
import QtQuick.Controls

Popup {
    id: settingsPopup
    modal: true
    width: 300
    height: 280
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    property bool show3D: false
    property Theme theme

    property real defaultFrom: 0.1
    property real defaultTo: 2.0

    Rectangle {
        id: popupRect
        anchors.fill: parent
        color: theme.base

        Column {
            anchors.centerIn: parent
            spacing: 15

            Text {
                text: show3D ? "3D Settings" : "2D Grid & Axis Settings"
                font.bold: true
                font.pointSize: 14
                color: theme ? theme.windowText : "black"
            }

            // --- X Scale Slider ---
            Row {
                spacing: 8
                Text {
                    text: "X Scale:"
                    color: theme ? theme.windowText : "black"
                    width: 70
                }
                Slider {
                    id: xScaleSlider
                    from: 0.1
                    to: 2.0
                    value: Graph2DState.xScale
                    stepSize: 0.001
                    width: 150

                    function scale() {
                        Graph2DState.xScale = value;

                        if (value < defaultFrom)
                            from = value;
                        else if (value > defaultTo)
                            to = value;
                        else {
                            from = defaultFrom;
                            to = defaultTo;
                        }
                    }

                    onMoved: scale()
                }
                Text {
                    text: Graph2DState.xScale.toFixed(2)
                    color: theme ? theme.windowText : "black"
                    width: 40
                }
            }

            // --- Y Scale Slider ---
            Row {
                spacing: 8
                Text {
                    text: "Y Scale:"
                    color: theme ? theme.windowText : "black"
                    width: 70
                }
                Slider {
                    id: yScaleSlider
                    from: 0.1
                    to: 2.0
                    value: Graph2DState.yScale
                    stepSize: 0.001
                    width: 150

                    function scale() {
                        Graph2DState.yScale = value;

                        if (value < defaultFrom)
                            from = value;
                        else if (value > defaultTo)
                            to = value;
                        else {
                            from = defaultFrom;
                            to = defaultTo;
                        }
                    }

                    onMoved: scale()
                }
                Text {
                    text: Graph2DState.yScale.toFixed(2)
                    color: theme ? theme.windowText : "black"
                    width: 40
                }
            }

            Button {
                text: "Restore Axes"
                anchors.horizontalCenter: parent.horizontalCenter
                onClicked: {
                    Graph2DState.reset();
                }
            }

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
    }
}

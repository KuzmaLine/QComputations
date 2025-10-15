import QtQuick
import QtQuick.Controls

Popup {
    id: settingsPopup
    modal: true
    width: 300
    height: 280
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    property Graph2DView graph
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

            // Axis sliders
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
                    value: graph.xScale
                    stepSize: 0.001
                    width: 150

                    function scale() {
                        graph.xScale = value;

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
                    text: graph.xScale.toFixed(2)
                    color: theme ? theme.windowText : "black"
                    width: 40
                }
            }

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
                    value: graph.yScale
                    stepSize: 0.001
                    width: 150

                    function scale() {
                        graph.yScale = value;
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
                    text: graph.yScale.toFixed(2)
                    color: theme ? theme.windowText : "black"
                    width: 40
                }
            }

            Button {
                text: "Restore Axes"
                anchors.horizontalCenter: parent.horizontalCenter
                onClicked: {
                    graph.xScale = 1.0;
                    graph.yScale = 1.0;
                }
            }

            CheckBox {
                text: "Show Grid"
                checked: graph.gridVisible
                onToggled: graph.gridVisible = checked
            }

            CheckBox {
                text: "Show Sub-Ticks"
                checked: graph.showSubTicks
                onToggled: graph.showSubTicks = checked
            }
        }
    }
}

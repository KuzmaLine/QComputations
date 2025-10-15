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
                    stepSize: 0.05
                    onValueChanged: graph.xScale = value
                    width: 150
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
                    stepSize: 0.05
                    onValueChanged: graph.yScale = value
                    width: 150
                }
                Text {
                    text: graph.xScale.toFixed(2)
                    color: theme ? theme.windowText : "black"
                    width: 40
                }
            }

            // Show grid toggle
            CheckBox {
                text: "Show Grid"
                checked: graph.gridVisible
                onToggled: graph.gridVisible = checked
            }

            // Show sub-ticks toggle
            CheckBox {
                text: "Show Sub-Ticks"
                checked: graph.showSubTicks
                onToggled: graph.showSubTicks = checked
            }

            Button {
                text: "Close"
                anchors.horizontalCenter: parent.horizontalCenter
                onClicked: settingsPopup.close()
            }
        }
    }
}

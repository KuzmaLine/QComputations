import QtQuick
import QtQuick.Controls

Popup {
    id: settingsPopup
    modal: true
    width: 300
    height: 280
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    property string xAxisName: "X Axis"
    property string yAxisName: "Y Axis"
    property real xScale: 1.0
    property real yScale: 1.0
    property bool showGrid: true
    property bool showSubTicks: true
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

            // Axis names

            Row {
                spacing: 8
                Text {
                    text: "X Axis Name:"
                    color: theme ? theme.windowText : "black"
                    width: 100
                }
                TextField {
                    id: xAxisNameField
                    text: settingsPopup.xAxisName
                    color: theme ? theme.windowText : "black"
                    onTextChanged: settingsPopup.xAxisName = text
                    width: 150
                }
            }
            Row {
                spacing: 8
                Text {
                    text: "Y Axis Name:"
                    color: theme ? theme.windowText : "black"
                    width: 100
                }
                TextField {
                    id: yAxisNameField
                    text: settingsPopup.yAxisName
                    color: theme ? theme.windowText : "black"
                    onTextChanged: settingsPopup.yAxisName = text
                    width: 150
                }
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
                    value: settingsPopup.xScale
                    stepSize: 0.05
                    onValueChanged: settingsPopup.xScale = value
                    width: 150
                }
                Text {
                    text: xScaleSlider.value.toFixed(2)
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
                    value: settingsPopup.yScale
                    stepSize: 0.05
                    onValueChanged: settingsPopup.yScale = value
                    width: 150
                }
                Text {
                    text: yScaleSlider.value.toFixed(2)
                    color: theme ? theme.windowText : "black"
                    width: 40
                }
            }

            // Show grid toggle
            CheckBox {
                text: "Show Grid"
                checked: settingsPopup.showGrid
                onToggled: settingsPopup.showGrid = checked
            }

            // Show sub-ticks toggle
            CheckBox {
                text: "Show Sub-Ticks"
                checked: settingsPopup.showSubTicks
                onToggled: settingsPopup.showSubTicks = checked
            }

            Button {
                text: "Close"
                anchors.horizontalCenter: parent.horizontalCenter
                onClicked: settingsPopup.close()
            }
        }
    }
}

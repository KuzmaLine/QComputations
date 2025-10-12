import QtQuick
import QtQuick.Controls

Popup {
    id: popup
    modal: true
    dim: true
    width: 420
    height: 280
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    property bool show3D: false

    Rectangle {
        anchors.fill: parent
        radius: 16
        gradient: Gradient {
            GradientStop {
                position: 0
                color: "#2a2a2a"
            }
            GradientStop {
                position: 1
                color: "#1e1e1e"
            }
        }
        border.color: "#555"
        border.width: 1
    }

    Column {
        anchors.centerIn: parent
        spacing: 14

        Text {
            text: show3D ? "3D Style" : "2D Grid Settings"
            color: "#9cffb4"
            font.bold: true
            font.pointSize: 16
        }

        Loader {
            sourceComponent: show3D ? threeD : twoD
        }

        Button {
            text: "Close"
            onClicked: popup.close()
        }
    }

    Component {
        id: twoD
        Column {
            spacing: 10
            Row {
                spacing: 8
                Text {
                    text: "Grid visible:"
                    color: "white"
                }
                Switch {
                    id: gridSwitch
                    checked: true
                }
            }
            Row {
                spacing: 8
                Text {
                    text: "Tick count:"
                    color: "white"
                }
                Slider {
                    from: 2
                    to: 20
                    stepSize: 1
                    width: 150
                }
            }
        }
    }

    Component {
        id: threeD
        Column {
            spacing: 10
            Row {
                spacing: 8
                Text {
                    text: "Surface type:"
                    color: "white"
                }
                ComboBox {
                    model: ["Soft", "Pointed", "Gradient"]
                }
            }
            Row {
                spacing: 8
                Text {
                    text: "Shading intensity:"
                    color: "white"
                }
                Slider {
                    from: 0
                    to: 1
                    stepSize: 0.05
                    width: 150
                }
            }
        }
    }
}

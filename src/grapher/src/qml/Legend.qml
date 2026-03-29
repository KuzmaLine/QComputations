import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Rectangle {
    id: legendRoot
    height: parent.height
    radius: 5

    property bool show3D: false

    color: Theme ? Theme.window : "white"

    ScrollView {
        anchors.fill: parent
        anchors.margins: 10
        clip: true

        ColumnLayout {
            id: legendColumn
            spacing: 6

            Repeater { // desc for vars from list
                id: legendRepeater
                model: []

                delegate: RowLayout { // delegate - how to desc
                    Layout.fillWidth: true
                    spacing: 8
                    height: 32

                    Rectangle {
                        id: colorRect
                        width: 20
                        height: 20
                        radius: 3
                        color: modelData ? (show3D ? modelData.baseColor : modelData.color) : "lightgray"
                        border.color: Qt.darker(color, 1.2)

                        MouseArea {
                            anchors.fill: parent
                            enabled: modelData !== null
                            onClicked: {
                                if (!modelData)
                                    return;
                                console.log("[Legend.qml] ColorRect clicked:", modelData.name);
                                colorDialog.seriesRef = modelData;
                                colorDialog.is3D = show3D;
                                colorDialog.selectedColor = show3D ? modelData.baseColor : modelData.color;
                                colorDialog.open();
                            }
                        }
                    }

                    TextInput {
                        Layout.fillWidth: true
                        text: modelData ? modelData.name : ""
                        color: Theme ? Theme.windowText : "black"
                        verticalAlignment: Text.AlignVCenter
                        selectByMouse: true
                        onEditingFinished: {
                            if (modelData && text !== modelData.name) {
                                console.log("[Legend.qml] Updating series name:", modelData.name, "->", text);
                                modelData.name = text;
                            }
                            cursorVisible = false;
                        }
                        Keys.onReturnPressed: editingFinished()
                        enabled: modelData !== null
                    }

                    CheckBox { // galochka
                        checked: modelData ? modelData.visible : false
                        enabled: modelData !== null
                        onCheckedChanged: {
                            if (modelData) {
                                console.log("[Legend.qml] Series visibility changed:", modelData.name, "->", checked);
                                modelData.visible = checked;
                            }
                        }
                    }
                }
            }
        }
    }

    ColorDialog {
        id: colorDialog
        title: "Select Series Color"
        property var seriesRef: null
        property bool is3D: false

        onAccepted: {
            if (!seriesRef)
                return;
            console.log("[Legend.qml] ColorDialog accepted for series:", seriesRef.name, "is3D:", is3D, "color:", selectedColor);
            if (is3D)
                seriesRef.baseColor = selectedColor;
            else
                seriesRef.color = selectedColor;

            // HACK:
            // force visual refresh if needed
            if (seriesRef.visible !== undefined) {
                var wasVisible = seriesRef.visible;
                seriesRef.visible = false;
                seriesRef.visible = wasVisible;
            }

            seriesRef = null;
        }

        onRejected: {
            console.log("[Legend.qml] ColorDialog rejected");
            seriesRef = null;
        }
    }

    Connections {
        target: chartManager
        ignoreUnknownSignals: true
        function onSurfaceSeriesChanged() {
            console.log("[Legend.qml] surfaceSeriesChanged signal received");
            if (show3D)
                Qt.callLater(updateLegendModel);
        }
        function onLineSeriesAdded(series) {
            console.log("[Legend.qml] lineSeriesAdded signal received:", series ? series.name : "(null)");
            if (!show3D)
                Qt.callLater(updateLegendModel);
        }
        function onLineSeriesRemoved(series) {
            console.log("[Legend.qml] lineSeriesRemoved signal received:", series ? series.name : "(null)");
            if (!show3D)
                Qt.callLater(updateLegendModel);
        }
    }

    function updateLegendModel() { // with change of data
        if (!chartManager) {
            console.warn("[Legend.qml] updateLegendModel: chartManager is null");
            return;
        }

        legendRepeater.model = show3D ? (chartManager.surfaceSeries ? [chartManager.surfaceSeries] : []) : (chartManager.lineSeriesList ? chartManager.lineSeriesList : []);

        console.log("[Legend.qml] updateLegendModel called, show3D:", show3D, "model length:", legendRepeater.model.length);
    }

    onShow3DChanged: Qt.callLater(updateLegendModel)
    Component.onCompleted: Qt.callLater(updateLegendModel)
}

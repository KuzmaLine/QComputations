import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs

ApplicationWindow {
    id: root
    width: 1000
    height: 600
    visible: true
    color: "#f0f0f0"
    property bool show3D: false
    property string lastFolderSelected: ""
    property string exportDir: ""

    Row {
        id: selectorRow
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 20
        z: 2

        Button {
            background: Rectangle {
                color: "#a0a0a0"
                border.color: "#cccccc"
                radius: 4
            }
            text: "Open Folder"
            onClicked: folderDialog.open()
        }

        Button {
            background: Rectangle {
                color: "#a0a0a0"
                border.color: "#cccccc"
                radius: 4
            }
            text: "Settings"
            onClicked: settingsPopup.open()
        }

        Button {
            background: Rectangle {
                color: "#a0a0a0"
                border.color: "#cccccc"
                radius: 4
            }
            text: "Export PNG"
            onClicked: exportGraphAndLegend()
        }
    }

    Item {
        id: exportContainer
        anchors.margins: 10
        anchors.top: selectorRow.bottom
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right

        Item {
            id: graphContainer
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: legendPanel.left

            Graph2DView {
                id: graph2DView
                anchors.fill: parent
                visible: !root.show3D
                xScale: settingsPopup.xScale
                yScale: settingsPopup.yScale
                gridVisible: settingsPopup.showGrid
                showSubTicks: settingsPopup.showSubTicks
            }

            Graph3DView {
                id: graph3DView
                anchors.fill: parent
                visible: root.show3D
            }
        }

        Legend {
            id: legendPanel
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.right: parent.right
            width: 200
            show3D: root.show3D
        }
    }

    FolderDialog {
        id: folderDialog
        title: "Select qData folder"
        currentFolder: "file://" + CurDirPath
        onAccepted: {
            var path = selectedFolder.toString().replace(/^(file:\/{3})/, "/");
            console.log("[Main.qml] Folder selected:", path);
            root.show3D = path.endsWith("3D");
            lastFolderSelected = path.split("/").pop();
            exportDir = CurDirPath + "/export/";
            console.log("[Main.qml] show3D =", root.show3D);
            chartManager.loadFolder(path);
        }
    }

    SettingsPopup {
        id: settingsPopup
        show3D: root.show3D
    }

    function exportGraphAndLegend() {
        if (!lastFolderSelected) {
            console.log("[Export] No folder selected yet!");
            return;
        }

        var date = new Date();
        var timestamp = date.getFullYear() + "-" + (date.getMonth() + 1) + "-" + date.getDate() + "_" + date.getHours() + "-" + date.getMinutes() + "-" + date.getSeconds();

        var filename = exportDir + lastFolderSelected + "_" + timestamp + ".png";

        console.log("[Export] Trying to export Graph+Legend as PNG:", filename);
        console.warn("[Export] Make sure that export directory exists");

        exportContainer.grabToImage(function (img) {
            var hiRes = img;
            hiRes.saveToFile(filename);
            console.log("[Export] Graph+Legend exported as PNG:", filename);
        }, Qt.size(exportContainer.width * 2, exportContainer.height * 2));
    }
}

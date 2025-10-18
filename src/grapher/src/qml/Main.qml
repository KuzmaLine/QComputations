import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

ApplicationWindow {
    id: root
    width: 1000
    height: 600
    visible: true
    property bool show3D: false
    property string lastFolderSelected: ""
    property string exportDir: ""

    palette.window: Theme.window
    palette.button: Theme.button
    palette.base: Theme.base
    palette.mid: Theme.mid
    palette.dark: Theme.darkColor
    palette.light: Theme.light
    palette.windowText: Theme.windowText
    palette.buttonText: Theme.buttonText
    palette.highlight: Theme.highlight
    palette.highlightedText: Theme.highlightedText

    GridLayout {
        id: mainLayout
        anchors.fill: parent
        columns: 2
        rowSpacing: 10
        columnSpacing: 10
        anchors.margins: 10

        // --- Top row buttons ---
        RowLayout {
            id: selectorRow
            spacing: 20
            //Layout.columnSpan: 2   // occupy both columns
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignLeft

            Button {
                Layout.fillWidth: true
                text: "Toggle dark mode"
                onClicked: Theme.dark = !Theme.dark
            }
            Button {
                Layout.fillWidth: true
                text: "Open Folder"
                onClicked: folderDialog.open()
            }
            Button {
                Layout.fillWidth: true
                text: "Settings"
                onClicked: settingsPopup.open()
            }
            Button {
                Layout.fillWidth: true
                text: "Export PNG"
                onClicked: exportGraphAndLegend()
            }
        }

        // --- Graph + Legend ---
        Item {
            id: exportContainer
            Layout.row: 1
            Layout.column: 0
            Layout.fillWidth: true
            Layout.fillHeight: true

            GridLayout {
                anchors.fill: parent
                columns: 2
                columnSpacing: 10

                // Graph container
                Item {
                    id: graphContainer
                    Layout.column: 0
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    Graph2DView {
                        id: graph2DView
                        anchors.fill: parent
                        visible: !root.show3D
                    }
                    Graph3DView {
                        id: graph3DView
                        anchors.fill: parent
                        visible: root.show3D
                    }
                }

                // Legend panel
                Legend {
                    id: legendPanel
                    Layout.column: 1
                    Layout.fillHeight: true
                    width: 200
                    show3D: root.show3D
                }
            }
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
        x: root.width - settingsPopup.width - 10
        y: selectorRow.y + selectorRow.height + 20
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

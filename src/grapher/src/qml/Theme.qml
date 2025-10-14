import QtQuick

QtObject {
    id: theme

    property bool dark: false

    property color window: dark ? "#1c1c1e" : "#f2f2f5"
    property color button: dark ? "#232326" : "#e0e0e5"
    property color base: dark ? "#2a2a2d" : "#dcdce0"
    property color mid: dark ? "#1f1f21" : "#c8c8cc"
    property color darkColor: dark ? "#141416" : "#a0a0a5"
    property color light: dark ? "#3a3a3c" : "#ffffff"
    property color alternateBase: dark ? "#2e2e30" : "#ededf0"
    property color windowText: dark ? "#ffffff" : "#000000"
    property color buttonText: dark ? "#ffffff" : "#000000"
    property color highlight: "#0a84ff"
    property color highlightedText: dark ? "#000000" : "#ffffff"

    property color graphBackground: window
    property color graphAxis: windowText
    property color graphGrid: mid
}

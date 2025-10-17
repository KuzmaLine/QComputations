import QtQuick 2.15
import QtQuick.Controls 2.15

SpinBox {
    id: spinBox

    property alias axis: spinBox._axis
    property alias isMin: spinBox._isMin
    property bool updating: false

    property string _axis: "x"    // "x" or "y"
    property bool _isMin: true

    from: -100000
    to: 100000
    stepSize: 1

    textFromValue: function (v) {
        return (v / 1000).toFixed(2);
    }
    valueFromText: function (t) {
        return Math.round(parseFloat(t) * 1000);
    }

    Component.onCompleted: {
        value = Math.round((spinBox._isMin ? Graph2DState["min" + _axis.toUpperCase()] : Graph2DState["max" + _axis.toUpperCase()]) * 1000);
    }

    onValueChanged: {
        if (updating)
            return;
        updating = true;
        if (spinBox._axis === 'x') {
            Graph2DState.setManualBorder('x', spinBox._isMin ? value / 1000 : Graph2DState.minX, spinBox._isMin ? Graph2DState.maxX : value / 1000);
        } else {
            Graph2DState.setManualBorder('y', spinBox._isMin ? value / 1000 : Graph2DState.minY, spinBox._isMin ? Graph2DState.maxY : value / 1000);
        }
        updating = false;
    }

    Connections {
        target: Graph2DState
        onMinXChanged: {
            if (spinBox._axis === 'x' && spinBox._isMin && !updating)
                value = Math.round(Graph2DState.minX * 1000);
        }
        onMaxXChanged: {
            if (spinBox._axis === 'x' && !spinBox._isMin && !updating)
                value = Math.round(Graph2DState.maxX * 1000);
        }
        onMinYChanged: {
            if (spinBox._axis === 'y' && spinBox._isMin && !updating)
                value = Math.round(Graph2DState.minY * 1000);
        }
        onMaxYChanged: {
            if (spinBox._axis === 'y' && !spinBox._isMin && !updating)
                value = Math.round(Graph2DState.maxY * 1000);
        }
    }
}

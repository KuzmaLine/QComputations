pragma Singleton
import QtQuick 6.9
import QtGraphs 6.3

QtObject {
    id: graph2DState

    property var chartView: null
    property bool initialized: false

    property real minX: 0
    property real maxX: 1
    property real minY: 0
    property real maxY: 1
    property real xScale: 1
    property real yScale: 1
    property real panOffsetX: 0
    property real panOffsetY: 0
    property real paddingFactor: 0.05
    property bool gridVisible: true
    property bool showSubTicks: true
    property bool lockX: false
    property bool lockY: false

    function initialize(chartViewRef) {
        if (!chartViewRef && !chartManager)
            return;

        chartView = chartViewRef;

        if (chartManager) {
            minX = chartManager.minX;
            maxX = chartManager.maxX;
            minY = chartManager.minY;
            maxY = chartManager.maxY;
        } else if (chartView && chartView.axisX && chartView.axisY) {
            minX = chartView.axisX.min;
            maxX = chartView.axisX.max;
            minY = chartView.axisY.min;
            maxY = chartView.axisY.max;
        }

        xScale = 1;
        yScale = 1;
        panOffsetX = 0;
        panOffsetY = 0;

        initialized = true;

        if (chartView)
            applyToChart();
    }

    function panBy(dx_px, dy_px) {
        if (!initialized || !chartView)
            return;
        const xRange = (maxX - minX) * xScale * (1 + paddingFactor);
        const yRange = (maxY - minY) * yScale * (1 + paddingFactor);
        if (!lockX)
            panOffsetX -= dx_px / chartView.width * xRange;
        if (!lockY)
            panOffsetY += dy_px / chartView.height * yRange;
        applyToChart();
    }

    function scaleBy(xFactor, yFactor) {
        if (!initialized || !chartView)
            return;

        const xRange = maxX - minX;
        const yRange = maxY - minY;

        const leftEdge = minX + panOffsetX;
        const bottomEdge = minY + panOffsetY;

        if (!lockX) {
            xScale = Math.min(Math.max(xScale * xFactor, 0.1), 10);
            panOffsetX = leftEdge - minX;
        }
        if (!lockY) {
            yScale = Math.min(Math.max(yScale * yFactor, 0.1), 10);
            panOffsetY = bottomEdge - minY;
        }

        applyToChart();
    }

    function setBorders(newMinX, newMaxX, newMinY, newMaxY) {
        if (typeof newMinX !== "number" || typeof newMaxX !== "number" || typeof newMinY !== "number" || typeof newMaxY !== "number")
            return;

        if (newMaxX <= newMinX || newMaxY <= newMinY)
            return;

        minX = newMinX;
        maxX = newMaxX;
        minY = newMinY;
        maxY = newMaxY;

        xScale = 1;
        yScale = 1;
        panOffsetX = 0;
        panOffsetY = 0;

        applyToChart();
    }

    function applyToChart() {
        if (!initialized || !chartView)
            return;
        const xRange = (maxX - minX) * xScale * (1 + paddingFactor);
        const yRange = (maxY - minY) * yScale * (1 + paddingFactor);
        chartView.axisX.min = minX + panOffsetX;
        chartView.axisX.max = minX + panOffsetX + xRange;
        chartView.axisY.min = minY + panOffsetY;
        chartView.axisY.max = minY + panOffsetY + yRange;
        chartView.axisX.gridVisible = gridVisible;
        chartView.axisY.gridVisible = gridVisible;
        chartView.axisX.subTickCount = showSubTicks ? 4 : 0;
        chartView.axisY.subTickCount = showSubTicks ? 4 : 0;
    }

    function resetAll() {
        if (!chartManager || !chartView)
            return;

        minX = chartManager.minX;
        maxX = chartManager.maxX;
        minY = chartManager.minY;
        maxY = chartManager.maxY;

        xScale = 1;
        yScale = 1;
        panOffsetX = 0;
        panOffsetY = 0;

        applyToChart();
    }

    function resetScaling() {
        xScale = 1;
        yScale = 1;
        applyToChart();
    }
    function resetPosition() {
        panOffsetX = 0;
        panOffsetY = 0;
        applyToChart();
    }

    property var chartConnections: Connections {
        target: chartManager
        function onLineSeriesAdded(series) {
            if (series && graph2DState.chartView)
                graph2DState.chartView.addSeries(series);
        }
        function onLineSeriesRemoved(series) {
            if (series && graph2DState.chartView)
                graph2DState.chartView.removeSeries(series);
        }
        function onMinMaxValuesChanged() {
            if (graph2DState.chartView)
                graph2DState.initialize(graph2DState.chartView);
        }
    }
}

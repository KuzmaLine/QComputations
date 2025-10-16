pragma Singleton
import QtQuick 6.9
import QtGraphs 6.3

QtObject {
    id: graph2DState

    // --- Persistent state ---
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
    property bool initialized: false

    // --- Utilities ---
    function isValidNumber(v) {
        return typeof v === "number" && !isNaN(v) && isFinite(v);
    }
    function isValidRange(minVal, maxVal) {
        return isValidNumber(minVal) && isValidNumber(maxVal) && maxVal > minVal;
    }

    // --- Initialize from chart ---
    function setFromChart(chartManager) {
        if (!chartManager)
            return;
        if (!isValidRange(chartManager.minX, chartManager.maxX) || !isValidRange(chartManager.minY, chartManager.maxY))
            return;

        minX = chartManager.minX;
        maxX = chartManager.maxX;
        minY = chartManager.minY;
        maxY = chartManager.maxY;
        xScale = 1;
        yScale = 1;
        panOffsetX = 0;
        panOffsetY = 0;
        initialized = true;
    }

    // --- Pan by pixel delta from DragHandler ---
    function panBy(deltaX_px, deltaY_px, chartView, minXVal, maxXVal, minYVal, maxYVal, startX = panOffsetX, startY = panOffsetY) {
        if (!initialized || !chartView)
            return;

        const visibleX = (maxXVal - minXVal) * xScale * (1 + paddingFactor);
        const visibleY = (maxYVal - minYVal) * yScale * (1 + paddingFactor);

        panOffsetX = startX + deltaX_px / chartView.width * visibleX * -1; // natural drag
        panOffsetY = startY + deltaY_px / chartView.height * visibleY;

        // Clamp
        panOffsetX = Math.min(Math.max(panOffsetX, 0), Math.max(0, (maxXVal - minXVal) * xScale - visibleX));
        panOffsetY = Math.min(Math.max(panOffsetY, 0), Math.max(0, (maxYVal - minYVal) * yScale - visibleY));

        applyToChart(chartView, minXVal, maxXVal, minYVal, maxYVal);
    }

    // --- Apply scale and preserve center ---
    function applyScaleAndPan(xFactor, yFactor, chartView, minXVal, maxXVal, minYVal, maxYVal) {
        if (!initialized || !chartView) {
            setFromChart({
                minX: minXVal,
                maxX: maxXVal,
                minY: minYVal,
                maxY: maxYVal
            });
        }

        const centerX = panOffsetX + (maxXVal - minXVal) * xScale / 2;
        const centerY = panOffsetY + (maxYVal - minYVal) * yScale / 2;

        xScale = Math.min(Math.max(xScale * xFactor, 0.1), 10);
        yScale = Math.min(Math.max(yScale * yFactor, 0.1), 10);

        panOffsetX = centerX - (maxXVal - minXVal) * xScale / 2;
        panOffsetY = centerY - (maxYVal - minYVal) * yScale / 2;

        // Clamp
        const visibleX = (maxXVal - minXVal) * xScale * (1 + paddingFactor);
        const visibleY = (maxYVal - minYVal) * yScale * (1 + paddingFactor);
        panOffsetX = Math.min(Math.max(panOffsetX, 0), Math.max(0, (maxXVal - minXVal) * xScale - visibleX));
        panOffsetY = Math.min(Math.max(panOffsetY, 0), Math.max(0, (maxYVal - minYVal) * yScale - visibleY));

        applyToChart(chartView, minXVal, maxXVal, minYVal, maxYVal);
    }

    // --- Apply current state to chart ---
    function applyToChart(chartView, minXVal, maxXVal, minYVal, maxYVal) {
        if (!initialized || !chartView)
            return;

        const xRange = (maxXVal - minXVal) * xScale * (1 + paddingFactor);
        const yRange = (maxYVal - minYVal) * yScale * (1 + paddingFactor);

        chartView.axisX.min = minXVal + panOffsetX;
        chartView.axisX.max = minXVal + panOffsetX + xRange;
        chartView.axisY.min = minYVal + panOffsetY;
        chartView.axisY.max = minYVal + panOffsetY + yRange;

        chartView.axisX.gridVisible = gridVisible;
        chartView.axisX.subTickCount = showSubTicks ? 4 : 0;
        chartView.axisY.gridVisible = gridVisible;
        chartView.axisY.subTickCount = showSubTicks ? 4 : 0;
    }

    // --- Reset everything ---
    function reset(chartManager) {
        minX = 0;
        maxX = 1;
        minY = 0;
        maxY = 1;
        xScale = 1;
        yScale = 1;
        panOffsetX = 0;
        panOffsetY = 0;
        initialized = false;
        if (chartManager)
            setFromChart(chartManager);
    }
}

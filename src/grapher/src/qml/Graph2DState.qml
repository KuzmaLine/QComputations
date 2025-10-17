pragma Singleton
import QtQuick 6.9
import QtGraphs 6.3

QtObject {
    id: graph2DState

    property var chartView: null
    property bool initialized: false
    property bool uiHovering: false

    property real visibleMinX: 0
    property real visibleMaxX: 1
    property real visibleMinY: 0
    property real visibleMaxY: 1

    property bool lockX: false
    property bool lockY: true
    property bool gridVisible: true
    property bool showSubTicks: true

    // --- derived / convenience ---
    readonly property real totalMinX: chartManager ? chartManager.minX : 0
    readonly property real totalMaxX: chartManager ? chartManager.maxX : 1
    readonly property real totalMinY: chartManager ? chartManager.minY : 0
    readonly property real totalMaxY: chartManager ? chartManager.maxY : 1

    // --- initialization ---
    function initialize(chartViewRef) {
        console.log("[Graph2DState] initialize() called");

        if (!chartViewRef || !chartManager) {
            console.warn("[Graph2DState] Initialization failed — chartView or chartManager missing");
            return;
        }

        chartView = chartViewRef;

        visibleMinX = chartManager.minX;
        visibleMaxX = chartManager.maxX;
        visibleMinY = chartManager.minY;
        visibleMaxY = chartManager.maxY;

        initialized = true;

        console.log("[Graph2DState] Initialized OK:", "X", visibleMinX, "→", visibleMaxX, "Y", visibleMinY, "→", visibleMaxY);

        applyToChart();
    }

    // --- helpers ---
    function visibleRangeX() {
        return visibleMaxX - visibleMinX;
    }
    function visibleRangeY() {
        return visibleMaxY - visibleMinY;
    }

    // --- set visible range safely ---
    function setVisibleRangeX(newMin, newMax) {
        if (!initialized)
            return;
        if (Math.abs(newMin - visibleMinX) < 1e-8 && Math.abs(newMax - visibleMaxX) < 1e-8)
            return;
        if (newMax <= newMin)
            return;
        console.log("[Graph2DState] setVisibleRangeX:", newMin, "→", newMax);

        visibleMinX = newMin;
        visibleMaxX = newMax;
        applyToChart();
    }

    function setVisibleRangeY(newMin, newMax) {
        if (!initialized)
            return;
        if (Math.abs(newMin - visibleMinY) < 1e-8 && Math.abs(newMax - visibleMaxY) < 1e-8)
            return;
        if (newMax <= newMin)
            return;
        console.log("[Graph2DState] setVisibleRangeY:", newMin, "→", newMax);

        visibleMinY = newMin;
        visibleMaxY = newMax;
        applyToChart();
    }

    // --- pan ---
    function panBy(dx_px, dy_px) {
        if (!initialized || !chartView)
            return;
        let xRange = visibleRangeX();
        let yRange = visibleRangeY();

        if (!lockX) {
            let deltaX = -dx_px / chartView.width * xRange;
            setVisibleRangeX(visibleMinX + deltaX, visibleMaxX + deltaX);
        }
        if (!lockY) {
            let deltaY = dy_px / chartView.height * yRange;
            setVisibleRangeY(visibleMinY + deltaY, visibleMaxY + deltaY);
        }
    }

    // --- scale ---
    function scaleBy(xFactor, yFactor) {
        if (!initialized)
            return;
        let centerX = visibleMinX + visibleRangeX() / 2;
        let centerY = visibleMinY + visibleRangeY() / 2;

        if (!lockX) {
            let newRangeX = visibleRangeX() / xFactor;
            setVisibleRangeX(centerX - newRangeX / 2, centerX + newRangeX / 2);
        }
        if (!lockY) {
            let newRangeY = visibleRangeY() / yFactor;
            setVisibleRangeY(centerY - newRangeY / 2, centerY + newRangeY / 2);
        }
    }

    // --- apply to chart ---
    function applyToChart() {
        if (!initialized || !chartView)
            return;
        console.log("[Graph2DState] applyToChart:", "X", visibleMinX, "→", visibleMaxX, "Y", visibleMinY, "→", visibleMaxY);

        chartView.axisX.min = visibleMinX;
        chartView.axisX.max = visibleMaxX;
        chartView.axisY.min = visibleMinY;
        chartView.axisY.max = visibleMaxY;
        chartView.axisX.gridVisible = gridVisible;
        chartView.axisY.gridVisible = gridVisible;
        chartView.axisX.subTickCount = showSubTicks ? 4 : 0;
        chartView.axisY.subTickCount = showSubTicks ? 4 : 0;
    }

    // --- reset ---
    function resetAll() {
        if (!chartManager)
            return;
        console.log("[Graph2DState] resetAll()");
        setVisibleRangeX(chartManager.minX, chartManager.maxX);
        setVisibleRangeY(chartManager.minY, chartManager.maxY);
    }

    function resetScaling() {
        if (!initialized || !chartManager)
            return;

        console.log("[Graph2DState] resetScaling()");

        // Keep lower-left corner fixed
        const minX = visibleMinX;
        const minY = visibleMinY;

        // Compute 1:1 range width/height from the original chart
        const originalWidth = chartManager.maxX - chartManager.minX;
        const originalHeight = chartManager.maxY - chartManager.minY;

        // Set visible max based on 1:1 scaling
        let newMaxX = minX + originalWidth;
        let newMaxY = minY + originalHeight;

        setVisibleRangeX(minX, newMaxX);
        setVisibleRangeY(minY, newMaxY);

        console.log(`[Graph2DState] resetScaling() → X: ${minX}–${newMaxX}, Y: ${minY}–${newMaxY}`);
    }

    function resetPosition() {
        if (!initialized || !chartManager)
            return;

        console.log("[Graph2DState] resetPosition()");

        // Current visible ranges
        const xRange = visibleMaxX - visibleMinX;
        const yRange = visibleMaxY - visibleMinY;

        // New visible region starts from (0,0)
        const newMinX = chartManager.minX;
        const newMaxX = newMinX + xRange;

        const newMinY = chartManager.minY;
        const newMaxY = newMinY + yRange;

        // Clamp to total bounds in case the shifted area exceeds available data
        const clampedMaxX = Math.min(newMaxX, chartManager.maxX);
        const clampedMaxY = Math.min(newMaxY, chartManager.maxY);

        const clampedMinX = clampedMaxX - xRange;
        const clampedMinY = clampedMaxY - yRange;

        setVisibleRangeX(clampedMinX, clampedMaxX);
        setVisibleRangeY(clampedMinY, clampedMaxY);

        console.log(`[Graph2DState] resetPosition() done → X ${clampedMinX} → ${clampedMaxX}, Y ${clampedMinY} → ${clampedMaxY}`);
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

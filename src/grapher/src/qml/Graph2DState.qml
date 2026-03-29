pragma Singleton
import QtQuick
import QtGraphs

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
    readonly property real paddingFactor: 0.01
    readonly property real minZoomFactor: 0.01   // cannot zoom smaller than 1% of total range
    readonly property real maxZoomFactor: 1.05   // can zoom out slightly beyond total range

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

        visibleMinY = newMin;
        visibleMaxY = newMax;
        applyToChart();
    }

    function scaleBy(xFactor, yFactor) {
        if (!initialized)
            return;

        // --- X-axis scaling ---
        if (!lockX) {
            let minX = visibleMinX;
            let width = visibleRangeX();
            let newWidth = width / xFactor;

            const totalRangeX = totalMaxX - totalMinX;
            const minWidth = totalRangeX * minZoomFactor;
            const maxWidth = totalRangeX * maxZoomFactor;

            newWidth = Math.max(minWidth, Math.min(newWidth, maxWidth));

            // Expand to the right first
            let newMax = minX + newWidth;
            if (newMax > totalMaxX * maxZoomFactor) {
                let shift = newMax - totalMaxX;
                minX = Math.max(totalMinX, minX - shift);
                newMax = minX + newWidth;
            }

            if (minX < totalMinX) {
                minX = totalMinX;
                newMax = minX + newWidth;
            }

            setVisibleRangeX(minX, newMax);
        }

        // --- Y-axis scaling ---
        if (!lockY) {
            let minY = visibleMinY;
            let height = visibleRangeY();
            let newHeight = height / yFactor;

            const totalRangeY = totalMaxY - totalMinY;
            const minHeight = totalRangeY * minZoomFactor;
            const maxHeight = totalRangeY * maxZoomFactor;

            newHeight = Math.max(minHeight, Math.min(newHeight, maxHeight));

            // Expand upwards
            let newMax = minY + newHeight;
            if (newMax > totalMaxY * maxZoomFactor) {
                let shift = newMax - totalMaxY;
                minY = Math.max(totalMinY, minY - shift);
                newMax = minY + newHeight;
            }

            if (minY < totalMinY) {
                minY = totalMinY;
                newMax = minY + newHeight;
            }

            setVisibleRangeY(minY, newMax);
        }
    }

    function panBy(dx_px, dy_px) { // for move in zoomed picture
        if (!initialized || !chartView)
            return;

        let xRange = visibleRangeX();
        let yRange = visibleRangeY();

        const minXLimit = totalMinX - 0.1 * (totalMaxX - totalMinX);
        const maxXLimit = totalMaxX * 1.1;
        const minYLimit = totalMinY - 0.1 * (totalMaxY - totalMinY);
        const maxYLimit = totalMaxY * 1.1;

        // --- X-axis ---
        if (!lockX) {
            let deltaX = -dx_px / chartView.width * xRange;
            let newMinX = visibleMinX + deltaX;
            let newMaxX = visibleMaxX + deltaX;

            // Only clamp if we actually go beyond limits
            if (newMinX < minXLimit) {
                newMinX = minXLimit;
                newMaxX = minXLimit + xRange;
            }
            if (newMaxX > maxXLimit) {
                newMaxX = maxXLimit;
                newMinX = maxXLimit - xRange;
            }

            setVisibleRangeX(newMinX, newMaxX);
        }

        // --- Y-axis ---
        if (!lockY) {
            let deltaY = dy_px / chartView.height * yRange;
            let newMinY = visibleMinY + deltaY;
            let newMaxY = visibleMaxY + deltaY;

            if (newMinY < minYLimit) {
                newMinY = minYLimit;
                newMaxY = minYLimit + yRange;
            }
            if (newMaxY > maxYLimit) {
                newMaxY = maxYLimit;
                newMinY = maxYLimit - yRange;
            }

            setVisibleRangeY(newMinY, newMaxY);
        }
    }

    function applyToChart() {
        if (!initialized || !chartView)
            return;

        // NOTE: only add padding to maxY values to avoid cutting off top
        let bufY = (visibleMaxY - visibleMinY) * paddingFactor;

        chartView.axisX.min = visibleMinX;
        chartView.axisX.max = visibleMaxX;
        chartView.axisY.min = visibleMinY;
        chartView.axisY.max = visibleMaxY + bufY;
        chartView.axisX.gridVisible = gridVisible;
        chartView.axisY.gridVisible = gridVisible;
        chartView.axisX.subTickCount = showSubTicks ? 4 : 0;
        chartView.axisY.subTickCount = showSubTicks ? 4 : 0;
    }

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

        // Keep lower-left corner fixed
        const minX = visibleMinX;
        const minY = visibleMinY;

        const originalWidth = chartManager.maxX - chartManager.minX;
        const originalHeight = chartManager.maxY - chartManager.minY;

        let newMaxX = minX + originalWidth;
        let newMaxY = minY + originalHeight;

        setVisibleRangeX(minX, newMaxX);
        setVisibleRangeY(minY, newMaxY);

        console.log(`[Graph2DState] resetScaling() → X: ${minX}–${newMaxX}, Y: ${minY}–${newMaxY}`);
    }

    function resetPosition() {
        if (!initialized || !chartManager)
            return;

        const xRange = visibleMaxX - visibleMinX;
        const yRange = visibleMaxY - visibleMinY;

        const newMinX = chartManager.minX;
        const newMaxX = newMinX + xRange;

        const newMinY = chartManager.minY;
        const newMaxY = newMinY + yRange;

        const clampedMaxX = Math.min(newMaxX, chartManager.maxX);
        const clampedMaxY = Math.min(newMaxY, chartManager.maxY);

        const clampedMinX = clampedMaxX - xRange;
        const clampedMinY = clampedMaxY - yRange;

        setVisibleRangeX(clampedMinX, clampedMaxX);
        setVisibleRangeY(clampedMinY, clampedMaxY);

        console.log(`[Graph2DState] resetPosition() done → X ${clampedMinX} → ${clampedMaxX}, Y ${clampedMinY} → ${clampedMaxY}`);
    }

    property var chartConnections: Connections { // adding with serries
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

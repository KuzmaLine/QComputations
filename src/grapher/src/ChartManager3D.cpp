#include "ChartManager3D.h"

#include <QDebug>
#include <QFile>
#include <QRegularExpression>
#include <QTextStream>
#include <QtMath>
#include <limits>

ChartManager3D::ChartManager3D(QObject *parent) : QObject(parent) {
    m_surfaceSeries = new QSurface3DSeries(new QSurfaceDataProxy(), this);
    m_surfaceSeries->setDrawMode(QSurface3DSeries::DrawSurfaceAndWireframe);
    m_surfaceSeries->setBaseColor(Qt::blue);
    m_surfaceSeries->setItemLabelVisible(true);
    m_surfaceSeries->setName("Surface3D");
}

ChartManager3D::~ChartManager3D() { clearSurface(); }

// ------------------- Safe clear -------------------
void ChartManager3D::clearSurface() {
    m_surfaceArray.clear();
    m_maxX = m_maxY = m_maxZ = 0;
    m_minX = m_minY = m_minZ = 0;

    if (m_surfaceSeries && m_surfaceSeries->dataProxy()) m_surfaceSeries->dataProxy()->resetArray(m_surfaceArray);

    emit surfaceSeriesChanged();
    emit minMaxValuesChanged();
}

// ------------------- Folder loading -------------------
void ChartManager3D::loadFolder3D(const QString &folderPath) {
    qDebug() << "[ChartManager3D] loadFolder3D:" << folderPath;

    QFile xFile(folderPath + "/x.csv");
    QFile yFile(folderPath + "/y.csv");
    QFile zFile(folderPath + "/z.csv");
    if (!xFile.open(QIODevice::ReadOnly | QIODevice::Text) || !yFile.open(QIODevice::ReadOnly | QIODevice::Text) ||
        !zFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "[ChartManager3D] Failed to open CSV files in folder:" << folderPath;
        return;
    }

    QTextStream xs(&xFile), ys(&yFile), zs(&zFile);
    QVector<double> xVec, yVec;
    QVector<QVector<double>> zMat;

    auto parseCsvLine = [](const QString &line, QVector<double> &vec) {
        for (const QString &s : line.split(QRegularExpression("[,\\s]+"), Qt::SkipEmptyParts)) {
            bool ok = false;
            double val = s.toDouble(&ok);
            if (ok) vec.append(val);
        }
    };

    auto parseCsvMatrix = [](QTextStream &ts, QVector<QVector<double>> &mat) {
        while (!ts.atEnd()) {
            QString line = ts.readLine().trimmed();
            if (line.isEmpty()) continue;
            QVector<double> row;
            for (const QString &s : line.split(QRegularExpression("[,\\s]+"), Qt::SkipEmptyParts)) {
                bool ok = false;
                double val = s.toDouble(&ok);
                if (ok) row.append(val);
            }
            if (!row.isEmpty()) mat.append(row);
        }
    };

    while (!xs.atEnd()) parseCsvLine(xs.readLine(), xVec);
    while (!ys.atEnd()) parseCsvLine(ys.readLine(), yVec);
    parseCsvMatrix(zs, zMat);

    xFile.close();
    yFile.close();
    zFile.close();

    if (xVec.isEmpty() || yVec.isEmpty() || zMat.isEmpty()) {
        qWarning() << "[ChartManager3D] Empty data in one of the CSV files";
        return;
    }

    clearSurface();

    // --- Adaptive downsampling ---
    int totalPoints = xVec.size() * yVec.size();
    int step = 1;
    if (totalPoints > MAX_POINTS_3D) {
        double ratio = static_cast<double>(totalPoints) / MAX_POINTS_3D;
        step = qCeil(qSqrt(ratio));
    }
    step = qMax(step, m_samplingStep);

    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double minZ = std::numeric_limits<double>::infinity();

    for (int yi = 0; yi < yVec.size() && yi < zMat.size(); yi += step) {
        const QVector<double> &zRow = zMat[yi];
        QSurfaceDataRow row;
        for (int xi = 0; xi < xVec.size() && xi < zRow.size(); xi += step) {
            double x = xVec[xi], y = yVec[yi], z = zRow[xi];
            row.append(QSurfaceDataItem(QVector3D(x, z, y)));

            maxX = qMax(maxX, x);
            maxY = qMax(maxY, y);
            maxZ = qMax(maxZ, z);
            minX = qMin(minX, x);
            minY = qMin(minY, y);
            minZ = qMin(minZ, z);
        }
        m_surfaceArray.append(row);
    }

    m_maxX = maxX;
    m_maxY = maxY;
    m_maxZ = maxZ;
    m_minX = minX;
    m_minY = minY;
    m_minZ = minZ;

    if (m_surfaceSeries && m_surfaceSeries->dataProxy()) m_surfaceSeries->dataProxy()->resetArray(m_surfaceArray);

    emit minMaxValuesChanged();
    emit surfaceSeriesChanged();

    qDebug() << "[ChartManager3D] Finished loading 3D folder:" << folderPath
             << "total points (after downsampling):" << m_surfaceArray.size();
}

#pragma once

#include <QColor>
#include <QObject>
#include <QPointer>
#include <QSurface3DSeries>
#include <QSurfaceDataItem>
#include <QSurfaceDataProxy>
#include <QVector3D>

class ChartManager3D : public QObject {
    Q_OBJECT

   public:
    explicit ChartManager3D(QObject *parent = nullptr);
    ~ChartManager3D();

    void loadFolder3D(const QString &path);

    QSurface3DSeries *surfaceSeries() const { return m_surfaceSeries; }

    void setSamplingStep(int step) {
        m_samplingStep = qMax(1, step);
        emit samplingStepChanged();
    }

    double maxX() const { return m_maxX; }
    double maxY() const { return m_maxY; }
    double maxZ() const { return m_maxZ; }
    double minX() const { return m_minX; }
    double minY() const { return m_minY; }
    double minZ() const { return m_minZ; }

   signals:
    void surfaceSeriesChanged();
    void samplingStepChanged();
    void minMaxValuesChanged();

   private:
    void clearSurface();

   private:
    QPointer<QSurface3DSeries> m_surfaceSeries;
    QSurfaceDataArray m_surfaceArray;
    int m_samplingStep = 1;

    double m_maxX = 0, m_maxY = 0, m_maxZ = 0;
    double m_minX = 0, m_minY = 0, m_minZ = 0;

    const int MAX_POINTS_3D = 50000;
};

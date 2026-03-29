#pragma once

#include <QColor>
#include <QLineSeries>
#include <QObject>
#include <QPointer>

class ChartManager2D : public QObject {
    Q_OBJECT

   public:
    explicit ChartManager2D(QObject *parent = nullptr);
    ~ChartManager2D();

    void loadFolder2D(const QString &path);

    qsizetype count() const;
    QLineSeries *getSeries(qsizetype index) const;
    QColor getSeriesColor(QLineSeries *series) const;

    void updateSeriesColor(QLineSeries *series, const QColor &color);
    void updateSeriesVisibility(QLineSeries *series, bool visible);
    void updateSeriesName(QLineSeries *series, const QString &name);
    void clearSeriesList();

    void setSamplingStep(int step);
    double maxX() const { return m_maxX; }
    double maxY() const { return m_maxY; }
    double minX() const { return m_minX; }
    double minY() const { return m_minY; }

    QList<QLineSeries *> getSeriesRawList() const;

   signals:
    void lineSeriesAdded(QLineSeries *series);
    void lineSeriesRemoved(QLineSeries *series);
    void lineSeriesListChanged();
    void samplingStepChanged();
    void minMaxValuesChanged();
    void seriesColorChanged(QObject *series);
    void seriesVisibilityChanged(QObject *series);
    void seriesNameChanged(QObject *series);

   private:
    QList<QPointer<QLineSeries>> m_lineSeriesList;
    int m_samplingStep = 1;  //TODO: remove
    double m_maxX = 0, m_maxY = 0;
    double m_minX = 0, m_minY = 0;

    const int MAX_POINTS_2D = 100000;
};

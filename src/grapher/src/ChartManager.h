#pragma once

#include <QColor>
#include <QLineSeries>
#include <QObject>
#include <QQmlListProperty>

#include "ChartManager2D.h"
#include "ChartManager3D.h"

class ChartManager : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool show2D READ show2D WRITE setShow2D NOTIFY show2DChanged)
    Q_PROPERTY(bool show3D READ show3D WRITE setShow3D NOTIFY show3DChanged)
    Q_PROPERTY(double maxX READ maxX NOTIFY minMaxValuesChanged)
    Q_PROPERTY(double maxY READ maxY NOTIFY minMaxValuesChanged)
    Q_PROPERTY(double maxZ READ maxZ NOTIFY minMaxValuesChanged)
    Q_PROPERTY(double minX READ minX NOTIFY minMaxValuesChanged)
    Q_PROPERTY(double minY READ minY NOTIFY minMaxValuesChanged)
    Q_PROPERTY(double minZ READ minZ NOTIFY minMaxValuesChanged)

    Q_PROPERTY(QQmlListProperty<QLineSeries> lineSeriesList READ lineSeriesList NOTIFY lineSeriesListChanged)
    Q_PROPERTY(QObject *surfaceSeries READ surfaceSeries NOTIFY surfaceSeriesChanged)

   public:
    explicit ChartManager(QObject *parent = nullptr);
    ~ChartManager() override;

    bool show2D() const { return m_show2D; }
    bool show3D() const { return m_show3D; }
    QQmlListProperty<QLineSeries> lineSeriesList();
    QObject *surfaceSeries() const;

    Q_INVOKABLE void setShow2D(bool visible);
    Q_INVOKABLE void setShow3D(bool visible);

    Q_INVOKABLE void clearAllLineSeries();

    Q_INVOKABLE void loadFolder(const QString &path);
    Q_INVOKABLE void redraw();

    Q_INVOKABLE double maxX() const;
    Q_INVOKABLE double maxY() const;
    Q_INVOKABLE double maxZ() const;

    Q_INVOKABLE double minX() const;
    Q_INVOKABLE double minY() const;
    Q_INVOKABLE double minZ() const;
   signals:
    void minMaxValuesChanged();

    void show2DChanged();
    void lineSeriesAdded(QLineSeries *series);
    void lineSeriesRemoved(QLineSeries *series);
    void lineSeriesListChanged();

    void show3DChanged();
    void surfaceSeriesChanged();

   private:
    void setGraphType(bool is3D);

   private:
    ChartManager2D *m_2dManager{nullptr};
    ChartManager3D *m_3dManager{nullptr};

    bool m_show2D{true};
    bool m_show3D{false};

    QString m_lastLoadedPath;
};

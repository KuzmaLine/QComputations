#include "ChartManager.h"

#include <QDebug>
#include <QDir>

ChartManager::ChartManager(QObject *parent) : QObject(parent) {
    m_2dManager = new ChartManager2D(this);
    m_3dManager = new ChartManager3D(this);

    // Forward 2D signals
    connect(m_2dManager, &ChartManager2D::lineSeriesAdded, this, [this](QLineSeries *series) {
        emit lineSeriesAdded(series);
    });
    connect(m_2dManager, &ChartManager2D::lineSeriesRemoved, this, [this](QLineSeries *series) {
        emit lineSeriesRemoved(series);
    }); // signals probros to here
    connect(m_2dManager, &ChartManager2D::lineSeriesListChanged, this, &ChartManager::lineSeriesListChanged);
    connect(m_2dManager, &ChartManager2D::minMaxValuesChanged, this, &ChartManager::minMaxValuesChanged);

    // Forward 3D signals
    connect(m_3dManager, &ChartManager3D::surfaceSeriesChanged, this, &ChartManager::surfaceSeriesChanged);
    connect(m_3dManager, &ChartManager3D::minMaxValuesChanged, this, &ChartManager::minMaxValuesChanged);
    // signals probros to here
}

ChartManager::~ChartManager() = default;

// ------------------- QML property -------------------
QQmlListProperty<QLineSeries> ChartManager::lineSeriesList() {
    // Only for QML read access; modifications go through functions
    auto rawList = m_2dManager ? m_2dManager->getSeriesRawList() : QList<QLineSeries *>();
    return QQmlListProperty<QLineSeries>(this, new QList<QLineSeries *>(rawList));
}

// ------------------- Clear all 2D series safely -------------------
void ChartManager::clearAllLineSeries() {
    if (!m_2dManager) return;

    const auto seriesList = m_2dManager->getSeriesRawList();
    for (auto *s : seriesList) {
        if (s) emit lineSeriesRemoved(s);
    }

    m_2dManager->clearSeriesList();  // safe internal clear
}

// ------------------- Visibility -------------------
void ChartManager::setShow2D(bool visible) {
    if (m_show2D == visible) return;
    m_show2D = visible;
    emit show2DChanged(); // FIX 2D TO 3D
}

void ChartManager::setShow3D(bool visible) {
    if (m_show3D == visible) return;

    if (!m_show3D && visible) clearAllLineSeries();  // remove 2D series when switching to 3D

    m_show3D = visible;
    emit show3DChanged();
}

// ------------------- Folder loading -------------------
void ChartManager::loadFolder(const QString &path) {
    QString folderName = QDir(path).dirName();
    bool is3D = folderName.endsWith("3d", Qt::CaseInsensitive); // ????

    setGraphType(is3D); // ChartManager::setGraphType
    m_lastLoadedPath = path;

    if (is3D && m_3dManager)
        m_3dManager->loadFolder3D(path);
    else if (!is3D && m_2dManager) {
        m_2dManager->loadFolder2D(path);
    }
    emit minMaxValuesChanged();
}

void ChartManager::redraw() {
    if (!m_lastLoadedPath.isEmpty()) loadFolder(m_lastLoadedPath);
}

// ------------------- Max values -------------------
double ChartManager::maxX() const {
    if (m_show3D && m_3dManager) return m_3dManager->maxX();
    if (!m_show3D && m_2dManager) return m_2dManager->maxX();
    return 1.0;
}

double ChartManager::maxY() const {
    if (m_show3D && m_3dManager) return m_3dManager->maxY();
    if (!m_show3D && m_2dManager) return m_2dManager->maxY();
    return 1.0;
}

double ChartManager::maxZ() const { return (m_show3D && m_3dManager) ? m_3dManager->maxZ() : 1.0; }

double ChartManager::minX() const {
    if (m_show3D && m_3dManager) return m_3dManager->minX();
    if (!m_show3D && m_2dManager) return m_2dManager->minX();
    return 1.0;
}

double ChartManager::minY() const {
    if (m_show3D && m_3dManager) return m_3dManager->minY();
    if (!m_show3D && m_2dManager) return m_2dManager->minY();
    return 1.0;
}

double ChartManager::minZ() const { return (m_show3D && m_3dManager) ? m_3dManager->minZ() : 1.0; }

// ------------------- Internal graph type -------------------
void ChartManager::setGraphType(bool is3D) {
    if (m_show3D == is3D) return;

    if (!m_show3D && is3D) clearAllLineSeries();

    m_show3D = is3D;
    emit show3DChanged();
}

QObject *ChartManager::surfaceSeries() const { return m_3dManager ? m_3dManager->surfaceSeries() : nullptr; }
                                                                                                // death

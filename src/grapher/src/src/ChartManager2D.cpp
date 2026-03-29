#include "ChartManager2D.h"

#include <QDebug>
#include <QDir>
#include <QFile>
#include <QRandomGenerator>
#include <QTextStream>

ChartManager2D::ChartManager2D(QObject *parent) : QObject(parent) {}

ChartManager2D::~ChartManager2D() { clearSeriesList(); }

void ChartManager2D::loadFolder2D(const QString &path) {
    qDebug() << "[ChartManager2D] loadFolder2D(" << path << ")";
    if (!m_lineSeriesList.empty()) clearSeriesList();

    QFile basisFile(path + "/basis.csv");
    QFile probsFile(path + "/probs.csv");
    QFile timeFile(path + "/time.csv");
    if (!basisFile.open(QIODevice::ReadOnly | QIODevice::Text) ||
        !probsFile.open(QIODevice::ReadOnly | QIODevice::Text) ||
        !timeFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "[ChartManager2D] Could not open CSV files in folder:" << path;
        return;
    }

    QTextStream basisIn(&basisFile), probsIn(&probsFile), timeIn(&timeFile);
    QVector<QString> namesVec = basisIn.readLine().split(",");
    basisFile.close();

    QVector<double> timeVec;
    for (const auto &str : timeIn.readLine().split(",")) {
        bool ok = false;
        double val = str.toDouble(&ok);
        if (ok) timeVec.append(val);
    }
    timeFile.close();

    QVector<QVector<double>> probsVec(namesVec.size());
    while (!probsIn.atEnd()) {
        QStringList lineParts = probsIn.readLine().split(",");
        if (lineParts.size() != namesVec.size()) continue;
        for (int i = 0; i < lineParts.size(); ++i) {
            bool ok = false;
            double val = lineParts[i].toDouble(&ok);
            if (ok) probsVec[i].append(val);
        }
    }
    probsFile.close();

    int totalPoints = namesVec.size() * timeVec.size();
    int step = 1;
    if (totalPoints > MAX_POINTS_2D) {
        step = qCeil(static_cast<double>(totalPoints) / MAX_POINTS_2D);
    }
    step = qMax(step, m_samplingStep);

    double maxX = 0, maxY = 0;
    double minX = 0, minY = 0;

    QRandomGenerator rng(1);
    for (int i = 0; i < probsVec.size(); ++i) {
        auto *lineSeries = new QLineSeries(this);
        lineSeries->setName(namesVec[i]);
        lineSeries->setColor(QColor(rng.bounded(0, 255), rng.bounded(0, 255), rng.bounded(0, 255)));

        for (int j = 0; j < timeVec.size(); j += step) {
            lineSeries->append(timeVec[j], probsVec[i][j]);
            maxX = qMax(maxX, timeVec[j]);
            minX = qMin(minX, timeVec[j]);
            maxY = qMax(maxY, probsVec[i][j]);
            minY = qMin(minY, probsVec[i][j]);
        }

        m_lineSeriesList.append(QPointer<QLineSeries>(lineSeries));
        emit lineSeriesAdded(lineSeries);
        emit lineSeriesListChanged();
    }

    m_maxX = maxX;
    m_maxY = maxY;
    m_minX = minX;
    m_minY = minY;
    emit minMaxValuesChanged();

    qDebug() << "[ChartManager2D] minX:" << m_minX << "maxX:" << m_maxX << "minY:" << m_minY << "maxY:" << m_maxY;

    qDebug() << "[ChartManager2D] Finished loading 2D folder:" << path
             << "total points (after downsampling):" << totalPoints;
}

// ------------------- Accessors -------------------
qsizetype ChartManager2D::count() const { return m_lineSeriesList.size(); }

QLineSeries *ChartManager2D::getSeries(qsizetype index) const {
    if (index < 0 || index >= m_lineSeriesList.size()) return nullptr;
    return m_lineSeriesList.at(index);
}

QColor ChartManager2D::getSeriesColor(QLineSeries *series) const { return series ? series->color() : QColor(); }

void ChartManager2D::updateSeriesColor(QLineSeries *series, const QColor &color) {
    if (!series) return;
    series->setColor(color);
    emit seriesColorChanged(series);
}

void ChartManager2D::updateSeriesVisibility(QLineSeries *series, bool visible) {
    if (!series) return;
    series->setVisible(visible);
    emit seriesVisibilityChanged(series);
}

void ChartManager2D::updateSeriesName(QLineSeries *series, const QString &name) {
    if (!series) return;
    series->setName(name);
    emit seriesNameChanged(series);
}
// Safe raw pointer list for internal use
QList<QLineSeries *> ChartManager2D::getSeriesRawList() const {
    QList<QLineSeries *> rawList;
    for (auto &ptr : m_lineSeriesList) {
        if (ptr) rawList.append(ptr);
    }
    return rawList;
}

// ------------------- Clear series -------------------
void ChartManager2D::clearSeriesList() {
    for (auto &ptr : m_lineSeriesList) {
        if (ptr) emit lineSeriesRemoved(ptr);
    }
    // qDeleteAll(m_lineSeriesList);
    m_lineSeriesList.clear();
    emit lineSeriesListChanged();
}

// ------------------- Sampling -------------------
void ChartManager2D::setSamplingStep(int step) {
    if (m_samplingStep == step) return;
    m_samplingStep = step;
    emit samplingStepChanged();
}

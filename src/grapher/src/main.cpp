#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QDir>
#include <QIcon>

#include "ChartManager.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    app.setWindowIcon(QIcon(":/assets/icons/icon.png"));
    // Create engine (instead of QQuickView)
    QQmlApplicationEngine engine;

    // Create ChartManager instance and expose it to QML
    ChartManager chartManager;
    engine.rootContext()->setContextProperty("chartManager", &chartManager);
    engine.rootContext()->setContextProperty("CurDirPath", QDir::currentPath());

    // Add import path for qml folder
#ifdef Q_OS_WIN
    QString extraImportPath = QStringLiteral("%1/../../../../%2")
            .arg(QGuiApplication::applicationDirPath(), QString::fromLatin1("qml"));
#else
    QString extraImportPath = QStringLiteral("%1/../../../%2")
            .arg(QGuiApplication::applicationDirPath(), QString::fromLatin1("qml"));
#endif
    engine.addImportPath(extraImportPath);

    // Load the main QML file
    const QUrl url(QStringLiteral("qrc:/qml/Main.qml"));
    QObject::connect(
        &engine, &QQmlApplicationEngine::objectCreated,
        &app, [url](QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        }, Qt::QueuedConnection);

    engine.load(url);

    return app.exec();
}

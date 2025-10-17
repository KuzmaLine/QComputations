#include <QDir>
#include <QGuiApplication>
#include <QIcon>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>

#include "ChartManager.h"

void messageHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
    QByteArray localMsg = msg.toLocal8Bit();
    QString time = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
    QTextStream out(stdout);

    // ANSI color codes
    const char *colorReset = "\033[0m";
    const char *colorInfo = "\033[36m";   // cyan
    const char *colorWarn = "\033[33m";   // yellow
    const char *colorError = "\033[31m";  // red
    const char *colorDebug = "\033[32m";  // green

    switch (type) {
        case QtDebugMsg:
            out << colorDebug << "[DEBUG]" << colorReset << " " << time << " " << localMsg.constData() << "\n";
            break;
        case QtInfoMsg:
            out << colorInfo << "[INFO]" << colorReset << " " << time << " " << localMsg.constData() << "\n";
            break;
        case QtWarningMsg:
            out << colorWarn << "[WARN]" << colorReset << " " << time << " " << localMsg.constData() << "\n";
            break;
        case QtCriticalMsg:
            out << colorError << "[CRIT]" << colorReset << " " << time << " " << localMsg.constData() << "\n";
            break;
        case QtFatalMsg:
            out << colorError << "[FATAL]" << colorReset << " " << time << " " << localMsg.constData() << "\n";
            abort();
    }
    out.flush();
}

int main(int argc, char *argv[]) {
    qInstallMessageHandler(messageHandler);
    QQuickStyle::setStyle("Fusion");
    QGuiApplication app(argc, argv);

    app.setWindowIcon(QIcon(":/assets/icons/icon.png"));
    // Create engine (instead of QQuickView)
    QQmlApplicationEngine engine;

    // Create ChartManager instance and expose it to QML
    ChartManager chartManager;
    engine.rootContext()->setContextProperty("chartManager", &chartManager);
    engine.rootContext()->setContextProperty("CurDirPath", QDir::currentPath());

    // Register singleton types
    qmlRegisterSingletonType(QUrl("qrc:/qml/Graph2DState.qml"), "Graph2D", 1, 0, "Graph2DState");
    qmlRegisterSingletonType(QUrl("qrc:/qml/Theme.qml"), "Theme", 1, 0, "Theme");

    // Add import path for qml folder
#ifdef Q_OS_WIN
    QString extraImportPath =
        QStringLiteral("%1/../../../../%2").arg(QGuiApplication::applicationDirPath(), QString::fromLatin1("qml"));
#else
    QString extraImportPath =
        QStringLiteral("%1/../../../%2").arg(QGuiApplication::applicationDirPath(), QString::fromLatin1("qml"));
#endif
    engine.addImportPath(extraImportPath);

    // Load the main QML file
    const QUrl url(QStringLiteral("qrc:/qml/Main.qml"));
    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreated,
        &app,
        [url](QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl) QCoreApplication::exit(-1);
        },
        Qt::QueuedConnection);

    engine.load(url);

    return app.exec();
}

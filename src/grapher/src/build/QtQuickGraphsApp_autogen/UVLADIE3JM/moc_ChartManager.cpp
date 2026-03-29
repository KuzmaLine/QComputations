/****************************************************************************
** Meta object code from reading C++ file 'ChartManager.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.9.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/ChartManager.h"
#include <QtGraphs/qsurface3dseries.h>
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ChartManager.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 69
#error "This file was generated using the moc from 6.9.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {
struct qt_meta_tag_ZN12ChartManagerE_t {};
} // unnamed namespace

template <> constexpr inline auto ChartManager::qt_create_metaobjectdata<qt_meta_tag_ZN12ChartManagerE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "ChartManager",
        "minMaxValuesChanged",
        "",
        "show2DChanged",
        "lineSeriesAdded",
        "QLineSeries*",
        "series",
        "lineSeriesRemoved",
        "lineSeriesListChanged",
        "show3DChanged",
        "surfaceSeriesChanged",
        "setShow2D",
        "visible",
        "setShow3D",
        "clearAllLineSeries",
        "loadFolder",
        "path",
        "redraw",
        "maxX",
        "maxY",
        "maxZ",
        "minX",
        "minY",
        "minZ",
        "show2D",
        "show3D",
        "lineSeriesList",
        "QQmlListProperty<QLineSeries>",
        "surfaceSeries"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'minMaxValuesChanged'
        QtMocHelpers::SignalData<void()>(1, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'show2DChanged'
        QtMocHelpers::SignalData<void()>(3, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'lineSeriesAdded'
        QtMocHelpers::SignalData<void(QLineSeries *)>(4, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 5, 6 },
        }}),
        // Signal 'lineSeriesRemoved'
        QtMocHelpers::SignalData<void(QLineSeries *)>(7, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 5, 6 },
        }}),
        // Signal 'lineSeriesListChanged'
        QtMocHelpers::SignalData<void()>(8, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'show3DChanged'
        QtMocHelpers::SignalData<void()>(9, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'surfaceSeriesChanged'
        QtMocHelpers::SignalData<void()>(10, 2, QMC::AccessPublic, QMetaType::Void),
        // Method 'setShow2D'
        QtMocHelpers::MethodData<void(bool)>(11, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Bool, 12 },
        }}),
        // Method 'setShow3D'
        QtMocHelpers::MethodData<void(bool)>(13, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Bool, 12 },
        }}),
        // Method 'clearAllLineSeries'
        QtMocHelpers::MethodData<void()>(14, 2, QMC::AccessPublic, QMetaType::Void),
        // Method 'loadFolder'
        QtMocHelpers::MethodData<void(const QString &)>(15, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::QString, 16 },
        }}),
        // Method 'redraw'
        QtMocHelpers::MethodData<void()>(17, 2, QMC::AccessPublic, QMetaType::Void),
        // Method 'maxX'
        QtMocHelpers::MethodData<double() const>(18, 2, QMC::AccessPublic, QMetaType::Double),
        // Method 'maxY'
        QtMocHelpers::MethodData<double() const>(19, 2, QMC::AccessPublic, QMetaType::Double),
        // Method 'maxZ'
        QtMocHelpers::MethodData<double() const>(20, 2, QMC::AccessPublic, QMetaType::Double),
        // Method 'minX'
        QtMocHelpers::MethodData<double() const>(21, 2, QMC::AccessPublic, QMetaType::Double),
        // Method 'minY'
        QtMocHelpers::MethodData<double() const>(22, 2, QMC::AccessPublic, QMetaType::Double),
        // Method 'minZ'
        QtMocHelpers::MethodData<double() const>(23, 2, QMC::AccessPublic, QMetaType::Double),
    };
    QtMocHelpers::UintData qt_properties {
        // property 'show2D'
        QtMocHelpers::PropertyData<bool>(24, QMetaType::Bool, QMC::DefaultPropertyFlags | QMC::Writable | QMC::StdCppSet, 1),
        // property 'show3D'
        QtMocHelpers::PropertyData<bool>(25, QMetaType::Bool, QMC::DefaultPropertyFlags | QMC::Writable | QMC::StdCppSet, 5),
        // property 'maxX'
        QtMocHelpers::PropertyData<double>(18, QMetaType::Double, QMC::DefaultPropertyFlags, 0),
        // property 'maxY'
        QtMocHelpers::PropertyData<double>(19, QMetaType::Double, QMC::DefaultPropertyFlags, 0),
        // property 'maxZ'
        QtMocHelpers::PropertyData<double>(20, QMetaType::Double, QMC::DefaultPropertyFlags, 0),
        // property 'minX'
        QtMocHelpers::PropertyData<double>(21, QMetaType::Double, QMC::DefaultPropertyFlags, 0),
        // property 'minY'
        QtMocHelpers::PropertyData<double>(22, QMetaType::Double, QMC::DefaultPropertyFlags, 0),
        // property 'minZ'
        QtMocHelpers::PropertyData<double>(23, QMetaType::Double, QMC::DefaultPropertyFlags, 0),
        // property 'lineSeriesList'
        QtMocHelpers::PropertyData<QQmlListProperty<QLineSeries>>(26, 0x80000000 | 27, QMC::DefaultPropertyFlags | QMC::EnumOrFlag, 4),
        // property 'surfaceSeries'
        QtMocHelpers::PropertyData<QObject*>(28, QMetaType::QObjectStar, QMC::DefaultPropertyFlags, 6),
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<ChartManager, qt_meta_tag_ZN12ChartManagerE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject ChartManager::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12ChartManagerE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12ChartManagerE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN12ChartManagerE_t>.metaTypes,
    nullptr
} };

void ChartManager::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<ChartManager *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->minMaxValuesChanged(); break;
        case 1: _t->show2DChanged(); break;
        case 2: _t->lineSeriesAdded((*reinterpret_cast< std::add_pointer_t<QLineSeries*>>(_a[1]))); break;
        case 3: _t->lineSeriesRemoved((*reinterpret_cast< std::add_pointer_t<QLineSeries*>>(_a[1]))); break;
        case 4: _t->lineSeriesListChanged(); break;
        case 5: _t->show3DChanged(); break;
        case 6: _t->surfaceSeriesChanged(); break;
        case 7: _t->setShow2D((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 8: _t->setShow3D((*reinterpret_cast< std::add_pointer_t<bool>>(_a[1]))); break;
        case 9: _t->clearAllLineSeries(); break;
        case 10: _t->loadFolder((*reinterpret_cast< std::add_pointer_t<QString>>(_a[1]))); break;
        case 11: _t->redraw(); break;
        case 12: { double _r = _t->maxX();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 13: { double _r = _t->maxY();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 14: { double _r = _t->maxZ();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 15: { double _r = _t->minX();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 16: { double _r = _t->minY();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        case 17: { double _r = _t->minZ();
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = std::move(_r); }  break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 0:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QLineSeries* >(); break;
            }
            break;
        case 3:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 0:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QLineSeries* >(); break;
            }
            break;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (ChartManager::*)()>(_a, &ChartManager::minMaxValuesChanged, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager::*)()>(_a, &ChartManager::show2DChanged, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager::*)(QLineSeries * )>(_a, &ChartManager::lineSeriesAdded, 2))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager::*)(QLineSeries * )>(_a, &ChartManager::lineSeriesRemoved, 3))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager::*)()>(_a, &ChartManager::lineSeriesListChanged, 4))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager::*)()>(_a, &ChartManager::show3DChanged, 5))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager::*)()>(_a, &ChartManager::surfaceSeriesChanged, 6))
            return;
    }
    if (_c == QMetaObject::ReadProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast<bool*>(_v) = _t->show2D(); break;
        case 1: *reinterpret_cast<bool*>(_v) = _t->show3D(); break;
        case 2: *reinterpret_cast<double*>(_v) = _t->maxX(); break;
        case 3: *reinterpret_cast<double*>(_v) = _t->maxY(); break;
        case 4: *reinterpret_cast<double*>(_v) = _t->maxZ(); break;
        case 5: *reinterpret_cast<double*>(_v) = _t->minX(); break;
        case 6: *reinterpret_cast<double*>(_v) = _t->minY(); break;
        case 7: *reinterpret_cast<double*>(_v) = _t->minZ(); break;
        case 8: *reinterpret_cast<QQmlListProperty<QLineSeries>*>(_v) = _t->lineSeriesList(); break;
        case 9: *reinterpret_cast<QObject**>(_v) = _t->surfaceSeries(); break;
        default: break;
        }
    }
    if (_c == QMetaObject::WriteProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: _t->setShow2D(*reinterpret_cast<bool*>(_v)); break;
        case 1: _t->setShow3D(*reinterpret_cast<bool*>(_v)); break;
        default: break;
        }
    }
}

const QMetaObject *ChartManager::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ChartManager::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN12ChartManagerE_t>.strings))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int ChartManager::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 18)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 18;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 18)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 18;
    }
    if (_c == QMetaObject::ReadProperty || _c == QMetaObject::WriteProperty
            || _c == QMetaObject::ResetProperty || _c == QMetaObject::BindableProperty
            || _c == QMetaObject::RegisterPropertyMetaType) {
        qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    }
    return _id;
}

// SIGNAL 0
void ChartManager::minMaxValuesChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 0, nullptr);
}

// SIGNAL 1
void ChartManager::show2DChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void ChartManager::lineSeriesAdded(QLineSeries * _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 2, nullptr, _t1);
}

// SIGNAL 3
void ChartManager::lineSeriesRemoved(QLineSeries * _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 3, nullptr, _t1);
}

// SIGNAL 4
void ChartManager::lineSeriesListChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}

// SIGNAL 5
void ChartManager::show3DChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 5, nullptr);
}

// SIGNAL 6
void ChartManager::surfaceSeriesChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 6, nullptr);
}
QT_WARNING_POP

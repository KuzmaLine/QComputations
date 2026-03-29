/****************************************************************************
** Meta object code from reading C++ file 'ChartManager2D.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.9.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/ChartManager2D.h"
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ChartManager2D.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN14ChartManager2DE_t {};
} // unnamed namespace

template <> constexpr inline auto ChartManager2D::qt_create_metaobjectdata<qt_meta_tag_ZN14ChartManager2DE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "ChartManager2D",
        "lineSeriesAdded",
        "",
        "QLineSeries*",
        "series",
        "lineSeriesRemoved",
        "lineSeriesListChanged",
        "samplingStepChanged",
        "minMaxValuesChanged",
        "seriesColorChanged",
        "seriesVisibilityChanged",
        "seriesNameChanged"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'lineSeriesAdded'
        QtMocHelpers::SignalData<void(QLineSeries *)>(1, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 3, 4 },
        }}),
        // Signal 'lineSeriesRemoved'
        QtMocHelpers::SignalData<void(QLineSeries *)>(5, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 3, 4 },
        }}),
        // Signal 'lineSeriesListChanged'
        QtMocHelpers::SignalData<void()>(6, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'samplingStepChanged'
        QtMocHelpers::SignalData<void()>(7, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'minMaxValuesChanged'
        QtMocHelpers::SignalData<void()>(8, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'seriesColorChanged'
        QtMocHelpers::SignalData<void(QObject *)>(9, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::QObjectStar, 4 },
        }}),
        // Signal 'seriesVisibilityChanged'
        QtMocHelpers::SignalData<void(QObject *)>(10, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::QObjectStar, 4 },
        }}),
        // Signal 'seriesNameChanged'
        QtMocHelpers::SignalData<void(QObject *)>(11, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::QObjectStar, 4 },
        }}),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<ChartManager2D, qt_meta_tag_ZN14ChartManager2DE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject ChartManager2D::staticMetaObject = { {
    QMetaObject::SuperData::link<QObject::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN14ChartManager2DE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN14ChartManager2DE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN14ChartManager2DE_t>.metaTypes,
    nullptr
} };

void ChartManager2D::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<ChartManager2D *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->lineSeriesAdded((*reinterpret_cast< std::add_pointer_t<QLineSeries*>>(_a[1]))); break;
        case 1: _t->lineSeriesRemoved((*reinterpret_cast< std::add_pointer_t<QLineSeries*>>(_a[1]))); break;
        case 2: _t->lineSeriesListChanged(); break;
        case 3: _t->samplingStepChanged(); break;
        case 4: _t->minMaxValuesChanged(); break;
        case 5: _t->seriesColorChanged((*reinterpret_cast< std::add_pointer_t<QObject*>>(_a[1]))); break;
        case 6: _t->seriesVisibilityChanged((*reinterpret_cast< std::add_pointer_t<QObject*>>(_a[1]))); break;
        case 7: _t->seriesNameChanged((*reinterpret_cast< std::add_pointer_t<QObject*>>(_a[1]))); break;
        default: ;
        }
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
        case 0:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 0:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QLineSeries* >(); break;
            }
            break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType(); break;
            case 0:
                *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType::fromType< QLineSeries* >(); break;
            }
            break;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (ChartManager2D::*)(QLineSeries * )>(_a, &ChartManager2D::lineSeriesAdded, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager2D::*)(QLineSeries * )>(_a, &ChartManager2D::lineSeriesRemoved, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager2D::*)()>(_a, &ChartManager2D::lineSeriesListChanged, 2))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager2D::*)()>(_a, &ChartManager2D::samplingStepChanged, 3))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager2D::*)()>(_a, &ChartManager2D::minMaxValuesChanged, 4))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager2D::*)(QObject * )>(_a, &ChartManager2D::seriesColorChanged, 5))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager2D::*)(QObject * )>(_a, &ChartManager2D::seriesVisibilityChanged, 6))
            return;
        if (QtMocHelpers::indexOfMethod<void (ChartManager2D::*)(QObject * )>(_a, &ChartManager2D::seriesNameChanged, 7))
            return;
    }
}

const QMetaObject *ChartManager2D::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ChartManager2D::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN14ChartManager2DE_t>.strings))
        return static_cast<void*>(this);
    return QObject::qt_metacast(_clname);
}

int ChartManager2D::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void ChartManager2D::lineSeriesAdded(QLineSeries * _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1);
}

// SIGNAL 1
void ChartManager2D::lineSeriesRemoved(QLineSeries * _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1);
}

// SIGNAL 2
void ChartManager2D::lineSeriesListChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void ChartManager2D::samplingStepChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 3, nullptr);
}

// SIGNAL 4
void ChartManager2D::minMaxValuesChanged()
{
    QMetaObject::activate(this, &staticMetaObject, 4, nullptr);
}

// SIGNAL 5
void ChartManager2D::seriesColorChanged(QObject * _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 5, nullptr, _t1);
}

// SIGNAL 6
void ChartManager2D::seriesVisibilityChanged(QObject * _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 6, nullptr, _t1);
}

// SIGNAL 7
void ChartManager2D::seriesNameChanged(QObject * _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 7, nullptr, _t1);
}
QT_WARNING_POP

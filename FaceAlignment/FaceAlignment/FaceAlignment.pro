QT += core widgets
CONFIG += qt c++11
CONFIG -= app_bundle

SOURCES = \
    main.cpp \
    imagepreprocessor.cpp

HEADERS += \
    common.h \
    decisiontree.hpp \
    fernclassifier.hpp \
    fernregressor.hpp \
    naivebayesclassifier.hpp \
    randomforest.hpp \
    testdecisiontree.h \
    testfernregressor.h \
    imagepreprocessor.h

INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc

INCLUDEPATH += $$PWD/../../../../Utils/PhGLib/include
macx: CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../Utils/PhGLib/lib/debug/ -lPhGLib
else:max: CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../Utils/PhGLib/lib/release/ -lPhGLib


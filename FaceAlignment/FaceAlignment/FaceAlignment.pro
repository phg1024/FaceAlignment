QT += core widgets xml
CONFIG += qt c++11 console
CONFIG -= app_bundle

SOURCES = \
    main.cpp \
    imagepreprocessor.cpp \
    explicitshaperegressor.cpp \
    facedetector.cpp

HEADERS += \
    common.h \
    decisiontree.hpp \
    fernclassifier.hpp \
    fernregressor.hpp \
    naivebayesclassifier.hpp \
    randomforest.hpp \
    testdecisiontree.h \
    testfernregressor.h \
    imagepreprocessor.h \
    explicitshaperegressor.h \
    transform.hpp \
    testarmadillo.h \
    numerical.hpp \
    testtransform.h \
    facedetector.h

INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_video

macx: LIBS += -framework Accelerate

INCLUDEPATH += $$PWD/../../../../Utils/PhGLib/include
LIBS += -L$$PWD/../../../../Utils/PhGLib/lib/release -lPhGLib


win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../Utils/armadillo-4.450.2/release/ -larmadillo
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../Utils/armadillo-4.450.2/debug/ -larmadillo
else:unix: LIBS += -L$$PWD/../../../../Utils/armadillo-4.450.2/ -larmadillo

INCLUDEPATH += $$PWD/../../../../Utils/armadillo-4.450.2/include
DEPENDPATH += $$PWD/../../../../Utils/armadillo-4.450.2/include

/Users/phg/Utils/libface/build/src
win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../Utils/libface/lib/ -lface
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../Utils/libface/lib/ -lface
else:unix: LIBS += -L$$PWD/../../../../Utils/libface/lib/ -lface

INCLUDEPATH += $$PWD/../../../../Utils/libface/include
DEPENDPATH += $$PWD/../../../../Utils/libface/include

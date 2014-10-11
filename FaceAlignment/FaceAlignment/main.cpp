#include <QCoreApplication>
#include <QCommandLineParser>
#include <QLabel>
#include <QDebug>

#include <iostream>
using namespace std;

#include "testdecisiontree.h"
#include "testfernregressor.h"

#include "imagepreprocessor.h"

int main(int argc, char **argv)
{
  QCoreApplication app(argc, argv);
  QCoreApplication::setApplicationName("Face alignment program.");
  QCoreApplication::setApplicationVersion("1.0");

  QCommandLineParser parser;
  parser.setApplicationDescription("Email phg@tamu.edu for details.");
  parser.addHelpOption();
  parser.addVersionOption();

  QCommandLineOption preprocessOption(QStringList() << "preprocess", "Preprocess an image.");
  parser.addOption(preprocessOption);

  QCommandLineOption imagefileOption(QStringList() << "image_file", "Image to process.", "file");
  parser.addOption(imagefileOption);

  QCommandLineOption pointfileOption(QStringList() << "point_file", "Points file to process.", "file");
  parser.addOption(pointfileOption);

  // Process the actual command line arguments given by the user
  parser.process(app);
  qDebug() << "Running in command line mode.";

  if( parser.isSet(preprocessOption) ) {
    qDebug() << "Preprocessing image " << parser.value("image_file") << " with point file " << parser.value("point_file");
    ImagePreprocessor proc(parser.value("image_file").toStdString(), parser.value("point_file").toStdString());
  }

  return 0;// app.exec();
}

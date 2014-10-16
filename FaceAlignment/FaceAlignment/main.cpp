#include <QCoreApplication>
#include <QCommandLineParser>
#include <QLabel>
#include <QDebug>

#include <iostream>
using namespace std;

#include "imagepreprocessor.h"
#include "explicitshaperegressor.h"


#include "testdecisiontree.h"
#include "testfernregressor.h"
#include "testarmadillo.h"
#include "testtransform.h"

void runTests() {
  cout << "running tests..." << endl;
  FATest::AramdilloTest::testArmadillo();
  FATest::TransformTest::testTransform();
}


int main(int argc, char **argv)
{
  runTests();  

  QCoreApplication app(argc, argv);
  QCoreApplication::setApplicationName("Face alignment program.");
  QCoreApplication::setApplicationVersion("1.0");

  QCommandLineParser parser;
  parser.setApplicationDescription("Email phg@tamu.edu for details.");
  parser.addHelpOption();
  parser.addVersionOption();

  QCommandLineOption preprocessOption(QStringList() << "preprocess", "Preprocess an image.");
  parser.addOption(preprocessOption);

  QCommandLineOption trainOption(QStringList() << "train", "Train regressors.", "file");
  parser.addOption(trainOption);

  QCommandLineOption outputOption(QStringList() << "output", "Trained regressor output file.", "file");
  parser.addOption(outputOption);

  QCommandLineOption testOption(QStringList() << "test", "Test sample files.", "file");
  parser.addOption(testOption);

  QCommandLineOption regressorOption(QStringList() << "regressor", "Trained regressor file.", "file");
  parser.addOption(regressorOption);

  QCommandLineOption imagefileOption(QStringList() << "image_file", "Image to process.", "file");
  parser.addOption(imagefileOption);

  QCommandLineOption pointfileOption(QStringList() << "point_file", "Points file to process.", "file");
  parser.addOption(pointfileOption);

  // Process the actual command line arguments given by the user
  parser.process(app);
  qDebug() << "Running in command line mode.";

  if( parser.isSet(preprocessOption) ) {
    qDebug() << "Preprocessing image " << parser.value("image_file") << " with point file " << parser.value("point_file");
    ImagePreprocessor proc;
    proc.process(parser.value("image_file").toStdString(), parser.value("point_file").toStdString());
  }
  else if( parser.isSet(trainOption) ) {
    qDebug() << "Training regressor with samples file "  << parser.value("train") << " and output file " << parser.value("output");
    ExplicitShapeRegressor r;
    r.train(parser.value("train").toStdString());
    r.write(parser.value("output").toStdString());
  }
  else if( parser.isSet(testOption)) {
    qDebug() << "Evaluating test samples in file " << parser.value("test") << " with regressor " << parser.value("regressor");
    ExplicitShapeRegressor r;
    r.load(parser.value("regressor").toStdString());
    r.evaluate(parser.value("test").toStdString());
  }

  return 0;// app.exec();
}

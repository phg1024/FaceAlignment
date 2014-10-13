#ifndef IMAGEPREPROCESSOR_H
#define IMAGEPREPROCESSOR_H

#include "common.h"

class ImagePreprocessor
{
public:
  ImagePreprocessor();
  void process(const string &imgfile, const string &ptsfile);

private:
};

#endif // IMAGEPREPROCESSOR_H

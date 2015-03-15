////////////////////////////////////////////////////////////////////////////
//    File:        userDefinedKeypoints.cpp
//    Author:      Yi-Ling Chen
//    Description : A simple example shows how to compute SIFT descriptors
//                  of user-specified keypoint locations.
//
//
//    Copyright (c) 2015 National Taiwan University
//    All Rights Reserved
//
//    Permission to use, copy, modify and distribute this software and its
//    documentation for educational, research and non-profit purposes, without
//    fee, and without a written agreement is hereby granted, provided that the
//    above copyright notice and the following paragraph appear in all copies.
//
//    The National Taiwan University make no representations
//    about the suitability of this software for any purpose. It is provided
//    'as is' without express or implied warranty.
//
//    Please send BUG REPORTS to yiling.chen.ntu@gmail.com
//
////////////////////////////////////////////////////////////////////////////

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <GL/glew.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

#include "SiftGPU.h"

using namespace cv;
using std::vector;
using std::iostream;
using std::cout;
using std::endl;

int main()
{
    SiftGPU *sift = new SiftGPU;

    vector<float> descriptors, my_descriptors;
    vector<SiftGPU::SiftKeypoint> keys, mykeys;
    int num = 0;

    char * argv[] = {"-fo", "-1", "-v", "1"};//

    int argc = sizeof(argv)/sizeof(char*);
    sift->ParseParam(argc, argv);

    // Create a context for computation, and SiftGPU will be initialized automatically
    // The same context can be used by SiftMatchGPU
    if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return 0;

    //Mat img = imread("../data/800-1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat img = imread("../data/640-1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    unsigned char* data = (unsigned char*) img.data;

    if(sift->RunSIFT(img.cols, img.rows, data, GL_LUMINANCE, GL_UNSIGNED_BYTE)) {
        sift->SaveSIFT("../data/640-1.sift");
        num = sift->GetFeatureNum();
        keys.resize(num);
        descriptors.resize(128*num);
        sift->GetFeatureVector(&keys[0], &descriptors[0]);
    }

    //Method2, set keypoints for the next coming image
    //The difference of with method 1 is that method 1 skips gaussian filtering
    mykeys.resize(num);
    my_descriptors.resize(128*num);
    for(int i = 0; i < num; ++i){
        mykeys[i].s = 1.0f;
        mykeys[i].o = 0.0f;
        mykeys[i].x = keys[i].x;
        mykeys[i].y = keys[i].y;
    }
    sift->SetKeypointList(num, &mykeys[0], 0);

    // run this to ensure clean results
    sift->RunSIFT("../data/800-1.jpg");
    sift->RunSIFT("../data/640-1.jpg");

    sift->RunSIFT(num, &mykeys[0], 0);
    sift->SaveSIFT("../data/640-1.sift.1");
    sift->GetFeatureVector(&mykeys[0], &my_descriptors[0]);

    float avg_err = 0;
    for(int i = 0; i < num; ++i) {
        float err = 0;
        for (int j = 0; j < 128; j++) {
            float diff = fabs(descriptors[i*128+j] - my_descriptors[i*128+j]);
            err += diff * diff;
        }
        err = sqrt(err);
        avg_err += err;

        cout << "feature #" << i << " error: " << err << endl;
    }
    cout << "average error :" << avg_err/num << endl;

    // clean up..
    delete sift;

    return 0;
}


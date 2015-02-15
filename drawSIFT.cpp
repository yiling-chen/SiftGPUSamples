////////////////////////////////////////////////////////////////////////////
//    File:        drawSIFT.cpp
//    Author:      Yi-Ling Chen
//    Description : A simple example shows how to use SiftGPU and SiftMatchGPU
//
//
//    Copyright (c) 2007 University of North Carolina at Chapel Hill
//    All Rights Reserved
//
//    Permission to use, copy, modify and distribute this software and its
//    documentation for educational, research and non-profit purposes, without
//    fee, and without a written agreement is hereby granted, provided that the
//    above copyright notice and the following paragraph appear in all copies.
//
//    The University of North Carolina at Chapel Hill make no representations
//    about the suitability of this software for any purpose. It is provided
//    'as is' without express or implied warranty.
//
//    Please send BUG REPORTS to ccwu@cs.unc.edu
//
//    ------------------------------------------------------------------
//    This is a even simpler version of the original SimpleSIFT example.
//    Removed all the platform dependent codes for the ease of reading.
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

using std::vector;
using std::iostream;
using namespace cv;

int main()
{
    SiftGPU *sift = new SiftGPU;
    SiftMatchGPU *matcher = new SiftMatchGPU(4096);

    vector<float> descriptors1(1), descriptors2(1);
    vector<SiftGPU::SiftKeypoint> keys1(1), keys2(1);
    int num1 = 0, num2 = 0;

    char * argv[] = {"-fo", "-1", "-v", "1"};//

    int argc = sizeof(argv)/sizeof(char*);
    sift->ParseParam(argc, argv);

    // Create a context for computation, and SiftGPU will be initialized automatically
    // The same context can be used by SiftMatchGPU
    if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return 0;

    Mat img1 = imread("../data/800-1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread("../data/640-1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    unsigned char* data1 = (unsigned char*) img1.data;
    unsigned char* data2 = (unsigned char*) img2.data;

    if(sift->RunSIFT(img1.cols, img1.rows, data1, GL_LUMINANCE, GL_UNSIGNED_BYTE)) {
        num1 = sift->GetFeatureNum();
        keys1.resize(num1);
        descriptors1.resize(128*num1);
        sift->GetFeatureVector(&keys1[0], &descriptors1[0]);
    }

    if(sift->RunSIFT(img2.cols, img2.rows, data2, GL_LUMINANCE, GL_UNSIGNED_BYTE)) {
        num2 = sift->GetFeatureNum();
        keys2.resize(num2);
        descriptors2.resize(128*num2);
        sift->GetFeatureVector(&keys2[0], &descriptors2[0]);
    }

    // Convert SiftGPU into OpenCV KeyPoint structures
    vector<KeyPoint> kpList1;
    vector<KeyPoint> kpList2;
    kpList1.resize(num1);
    kpList2.resize(num2);

    for (int i = 0; i < num1; i++) {
        Point2f pt(keys1[i].x, keys1[i].y);
        kpList1[i].pt = pt;
        //kpList1[i].size = keys1[i].s;
        //kpList1[i].angle = keys1[i].o;
    }

    for (int i = 0; i < num2; i++) {
        Point2f pt(keys2[i].x, keys2[i].y);
        kpList2[i].pt = pt;
        //kpList2[i].size = keys2[i].s;
        //kpList2[i].angle = keys2[i].o;
    }

    Mat img_keypoints_1;
    Mat img_keypoints_2;
    drawKeypoints( img1, kpList1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    drawKeypoints( img2, kpList2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    //-- Show detected (drawn) keypoints
    imshow("Keypoints 1", img_keypoints_1 );
    imshow("Keypoints 2", img_keypoints_2 );

    waitKey(0);

    //**********************GPU SIFT MATCHING*********************************
    //**************************select shader language*************************
    //SiftMatchGPU will use the same shader lanaguage as SiftGPU by default
    //Before initialization, you can choose between glsl, and CUDA(if compiled).
    //matcher->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA); // +i for the (i+1)-th device

    //Verify current OpenGL Context and initialize the Matcher;
    //If you don't have an OpenGL Context, call matcher->CreateContextGL instead;
    matcher->VerifyContextGL(); //must call once

    //Set descriptors to match, the first argument must be either 0 or 1
    //if you want to use more than 4096 or less than 4096
    //call matcher->SetMaxSift() to change the limit before calling setdescriptor
    matcher->SetDescriptors(0, num1, &descriptors1[0]); //image 1
    matcher->SetDescriptors(1, num2, &descriptors2[0]); //image 2

    //match and get result.
    int (*match_buf)[2] = new int[num1][2];
    //use the default thresholds. Check the declaration in SiftGPU.h
    int num_match = matcher->GetSiftMatch(num1, match_buf);
    std::cout << num_match << " sift matches were found;\n";

    vector<DMatch> matches;
    matches.resize(num_match);

    //enumerate all the feature matches
    for(int i  = 0; i < num_match; ++i) {
        //How to get the feature matches:
        //SiftGPU::SiftKeypoint & key1 = keys1[match_buf[i][0]];
        //SiftGPU::SiftKeypoint & key2 = keys2[match_buf[i][1]];
        //key1 in the first image matches with key2 in the second image
        matches[i].queryIdx = match_buf[i][0];
        matches[i].trainIdx = match_buf[i][1];
    }

    Mat img_matches;
    drawMatches(img1, kpList1, img2, kpList2, matches, img_matches);
    imshow("matches", img_matches);
    waitKey(0);

    // clean up..
    delete[] match_buf;
    delete sift;
    delete matcher;

    return 1;
}




#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include <cstdio>

using namespace cv;

IplImage* DrawCorrespondences(IplImage* img1, const vector<KeyPoint>& features1, IplImage* img2,
                              const vector<KeyPoint>& features2, const vector<DMatch>& desc_idx);
 
void help() {
    printf("Use the SURF descriptor to match keypoints between 2 images, show the correspondences, and show the stitched images\n");
    printf("Format: \n./panograph <image1> <image2> <algorithm> <XML params>\n");
    printf("For example: ./panograph ../testimages/112_1298.JPG ../testimages/112_1299.JPG FERN fern_params.xml\n");
}

int main(int argc, char** argv) {
    if (argc != 5) {
        help();
        return 0;
    }

    std::string img1_name = std::string(argv[1]);
    std::string img2_name = std::string(argv[2]);
    std::string alg_name = std::string(argv[3]);
    std::string params_filename = std::string(argv[4]);

    Ptr<GenericDescriptorMatcher> descriptorMatcher = GenericDescriptorMatcher::create(alg_name, params_filename);
    if (descriptorMatcher == 0) {
        printf ("Could not create descriptor\n");
        return 0;
    }

    printf("Reading images...\n");
    IplImage* img1 = cvLoadImage(img1_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    IplImage* img1_rgb = cvLoadImage(img1_name.c_str(), CV_LOAD_IMAGE_COLOR);
    IplImage* img2 = cvLoadImage(img2_name.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    IplImage* img2_rgb = cvLoadImage(img2_name.c_str(), CV_LOAD_IMAGE_COLOR);
    
    SURF surf_extractor(5.0e3);

    printf("Extracting keypoints...\n");
    vector<KeyPoint> keypoints1;
    surf_extractor(img1, Mat(), keypoints1);
    printf("Extracted %d keypoints from the first image\n", (int)keypoints1.size());

    vector<KeyPoint> keypoints2;
    surf_extractor(img2, Mat(), keypoints2);
    printf("Extracted %d keypoints from the second image\n", (int)keypoints2.size());

    printf("Finding nearest neighbors... \n");
    vector<DMatch> matches2to1;
    descriptorMatcher->match(img2, keypoints2, img1, keypoints1, matches2to1);

    printf("Drawing correspondences... \n");
    IplImage* img_corr = DrawCorrespondences(img1_rgb, keypoints1, img2_rgb, keypoints2, matches2to1);

    cvNamedWindow("correspondences", 1);
    cvShowImage("correspondences", img_corr);
    cvWaitKey(0);

    cvReleaseImage(&img1);
    cvReleaseImage(&img1_rgb);
    cvReleaseImage(&img2);
    cvReleaseImage(&img2_rgb);
    cvReleaseImage(&img_corr);
}


IplImage* DrawCorrespondences(IplImage* img1, const vector<KeyPoint>& features1, IplImage* img2,
                              const vector<KeyPoint>& features2, const vector<DMatch>& desc_idx) {
    //Create image large enough to fit both
    IplImage* img_corr = cvCreateImage(cvSize(img1->width + img2->width, MAX(img1->height, img2->height)), IPL_DEPTH_8U, 3);

    //Copy image 1 over. Use ROI (region of interest) to position the image on the left
    cvSetImageROI(img_corr, cvRect(0, 0, img1->width, img1->height));
    cvCopy(img1, img_corr);

    //Copy image 2 over. Use ROI (region of interest) to position the image on the right
    cvSetImageROI(img_corr, cvRect(img1->width, 0, img2->width, img2->height));
    cvCopy(img2, img_corr);

    cvResetImageROI(img_corr);

    //Draw red circles on feature points of 1st image
    for (size_t i = 0; i < features1.size(); i++)
    {
        cvCircle(img_corr, features1[i].pt, 3, CV_RGB(255, 0, 0));
    }

    //Draw green circles on feature points of 2nd image and a line to join with the corresponding feature in 1st image
    for (size_t i = 0; i < features2.size(); i++)
    {
        //Shift the feature point across to where the image is actually located
        CvPoint pt = cvPoint(cvRound(features2[i].pt.x + img1->width), cvRound(features2[i].pt.y));
        cvCircle(img_corr, pt, 3, CV_RGB(255, 0, 0));
        cvLine(img_corr, features1[desc_idx[i].trainIdx].pt, pt, CV_RGB(0, 255, 0));
    }

    return img_corr;
}

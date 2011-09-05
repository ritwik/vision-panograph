#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdio>

using namespace cv;

void help() {
    printf("Use the SURF descriptor to match keypoints between 2 images, show the correspondences, and show the stitched images\n");
    printf("Format: \n./panograph <image1> <image2> <algorithm> <XML params>\n");
    printf("For example: ./panograph testimages/112_1298.JPG testimages/112_1299.JPG FERN samples/fern_params.xml\n");
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
    Mat img1 = imread(img1_name.c_str(), 0);
    Mat img1_rgb = imread(img1_name.c_str(), -1);
    Mat img2 = imread(img2_name.c_str(), 0);
    Mat img2_rgb = imread(img2_name.c_str(), -1);

    printf("Setting up result image...\n");
    Size size1 = img1.size();
    Size size2 = img2.size();
    Mat result(Size(size2.width * 3, size2.height * 3), CV_8UC1);
    Rect img2ROI = Rect(size2.width, size2.height, size2.width, size2.height);
    Mat result2 = result(img2ROI);
    img2.copyTo(result2);

    SURF surf_extractor(10.0e3);

    printf("Extracting keypoints...\n");
    vector<KeyPoint> keypoints1;
    surf_extractor(img1, Mat(), keypoints1);
    printf("Extracted %d keypoints from the first image\n", (int)keypoints1.size());

    vector<KeyPoint> keypoints2;
    surf_extractor(img2, Mat(), keypoints2);
    printf("Extracted %d keypoints from the second image\n", (int)keypoints2.size());

    printf("Finding nearest neighbors... \n");
    vector<DMatch> matches1to2;
    descriptorMatcher->match(img1, keypoints1, img2, keypoints2, matches1to2);

    printf("Drawing correspondences... \n");
    Mat img_corr;
    drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, img_corr);
    imwrite("blah.jpg", img_corr);

    printf("Finding homography...\n");
    vector<Point2f> points1, points2;
    for(size_t q = 0; q < MIN(100, matches1to2.size()); q++)
    {
        const DMatch & dmatch = matches1to2[q];

        points1.push_back(keypoints1[dmatch.queryIdx].pt);

        Point2f translated = keypoints2[dmatch.trainIdx].pt;
        translated.x = translated.x + size2.width;
        translated.y = translated.y + size2.height;

        points2.push_back(translated);
    }
    Mat H = findHomography(Mat(points1), Mat(points2), RANSAC, 50);

    printf("Applying perspective warp...\n");
    Mat warped(Size(size1.width * 3, size1.height * 3), CV_8UC1);
    warpPerspective(img1, warped, H, warped.size(), INTER_LINEAR, BORDER_TRANSPARENT);
    imwrite("blah4.jpg", warped);
    //warped.copyTo(result2, warped);

    warped.copyTo(result, warped);

    imwrite("blah2.jpg", result);

    //So the problem at the moment is;
    //Getting an image that fits in image1 + transformed image 2
    //The problem is that I can't figure out how to do this transformation!
    //

    //Alright, slightly better now
    //Just getting this clipping issue
    //So need to translate first?

}

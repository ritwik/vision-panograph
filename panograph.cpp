#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdio>

using namespace cv;

Mat stitchImages(Mat img1, Mat img2, Ptr<GenericDescriptorMatcher> descriptorMatcher);
Mat cropBlack(Mat toCrop);

void help() {
    printf("Use the SURF descriptor to match keypoints between 2 images, show the correspondences, and show the stitched images\n");
    printf("Format: \n./panograph <algorithm> <XML params> <image1> <image2> ...\n");
    printf("For example: ./panograph FERN samples/fern_params.xml testimages/horizontal/IMG_1457.jpg testimages/horizontal/IMG_1456.jpg \n");
}

int main(int argc, char** argv) {
    if (argc < 5) {
        help();
        return 0;
    }

    //Get image names from args
    std::string alg_name = std::string(argv[1]);
    std::string params_filename = std::string(argv[2]);
    std::string imgNames[argc - 3];
    int imgCount = argc - 3;
    for (int i = 3; i < argc; i++) {
        imgNames[i - 3] = std::string(argv[i]);
    }

    //Set up descriptor matcher from args
    Ptr<GenericDescriptorMatcher> descriptorMatcher = GenericDescriptorMatcher::create(alg_name, params_filename);
    if (descriptorMatcher == 0) {
        printf ("Could not create descriptor\n");
        return 0;
    }

    printf("Reading images...\n");
    Mat imgs[imgCount];
    Mat imgs_rgb[imgCount];
    for (int i = 0; i < imgCount; i++) {
        imgs[i] = imread(imgNames[i].c_str(), 0);
        imgs_rgb[i] = imread(imgNames[i].c_str(), -1);
    }

    //Need a pair of likely matches
    //So take the first image and find the one it best compares to
    //So we find numMatches or something? Or just get the max? Yep.
    //Just need to return one number: the index of the best match
    //while (imgCount > 1) {
    //    find first non NULL
    //    send it to the bestMatch function
    //    then stitch it with the best match
    //    then set the first to NULL
    //    and add the result to the array where the best match was
    //    problem is the result image has black everywhere and is too LARGE_INTEGER
    //    how to prevent that?
    //}

    Mat img = stitchImages(imgs[0], imgs[1], descriptorMatcher); //Need to swap these around or something so that the bigger image is what determines the canvas size!
    Mat img2 = stitchImages(img, imgs[2], descriptorMatcher);
    Mat img3 = stitchImages(img2, imgs[3], descriptorMatcher);
    Mat img4 = stitchImages(img3, imgs[4], descriptorMatcher);
    Mat img5 = stitchImages(img4, imgs[5], descriptorMatcher);
    imwrite("result.jpg", img5);
}

Mat stitchImages(Mat img1, Mat img2, Ptr<GenericDescriptorMatcher> descriptorMatcher) {
    Size size1 = img1.size();
    Size size2 = img2.size();

    //Create a large image with image2 in the center so we can
    //draw image1 in
    printf("Setting up result image...\n");
    Mat result(Size(size2.width * 3, size2.height * 3), CV_8UC1, Scalar(0));
    Rect img2ROI = Rect(size2.width, size2.height, size2.width, size2.height);
    Mat centreOfResult = result(img2ROI);
    img2.copyTo(centreOfResult);

    printf("Result rows %d, result cols %d\n", result.rows, result.cols);
    printf("Size width %d, size height %d\n", size2.width, size2.height);

    SURF surf_extractor(8.0e3);

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
    imwrite("correspondences.jpg", img_corr);

    printf("Finding homography...\n");
    vector<Point2f> points1, points2;
    for(size_t q = 0; q < matches1to2.size(); q++)
    {
        const DMatch & dmatch = matches1to2[q];

        points1.push_back(keypoints1[dmatch.queryIdx].pt);

        //Since we want to draw the transformed image1 onto the result image
        //we translate the points so they are where image2 currently is
        //This also helps avoid the clipping that occurs if image 1 is transformed
        //while it is at (0,0)
        Point2f translated = keypoints2[dmatch.trainIdx].pt;
        translated.x = translated.x + size2.width;
        translated.y = translated.y + size2.height;
        points2.push_back(translated);
    }
    Mat H = findHomography(Mat(points1), Mat(points2), RANSAC, 85);

    printf("Applying perspective warp...\n");
    warpPerspective(img1, result, H, result.size(), INTER_LINEAR, BORDER_TRANSPARENT);

    printf("Cropping image...\n");
    imwrite("resultUncropped.jpg", result);
    Mat croppedResult = cropBlack(result);

    return croppedResult;
}

Mat cropBlack(Mat toCrop) {
    int minCol = toCrop.cols;
    int minRow = toCrop.rows;
    int maxCol = 0;
    int maxRow = 0;
    for (int i = 0; i < toCrop.rows - 3; i++) {
        for (int j = 0; j < toCrop.cols; j++) {
            if (toCrop.at<char>(i, j) != 0) {
                if (i < minRow) {minRow = i;}
                if (j < minCol) {minCol = j;}
                if (i > maxRow) {maxRow = i;}
                if (j > maxCol) {maxCol = j;}
            }
        }
    }

    printf("minRow: %d, minCol: %d, maxRow: %d, maxCol: %d\n", minRow, minCol, maxRow, maxCol);
    Rect cropRect = Rect(minCol, minRow, maxCol - minCol, maxRow - minRow);
    Mat cropped = toCrop(cropRect);

    return cropped;
}

//Todo:
//- Get it working on colour images
//- Figure out why I'm still getting explosions
//-

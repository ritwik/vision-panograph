#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdio>

using namespace cv;

Mat stitchImages(Mat img1, Mat img2, Mat img1rgb, Mat img2rgb, Ptr<GenericDescriptorMatcher> descriptorMatcher);
Mat cropBlack(Mat toCrop, Mat toCropGray);
vector<int> findBestMatch(vector<Mat> imgs, Ptr<GenericDescriptorMatcher> descriptorMatcher);
Mat avgMatchDistances;

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

    //For demo:
    //horizontalAndVertical: first 4 images, 1298 - 1301
    //blurring: 456, 457-2.9, 458
    //horizontal: first 4

    //Patch sizes
    //For demo:
    //blurring: 91
    //everythingElse: 31

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
    vector<Mat> imgs;
    vector<Mat> imgs_rgb;
    for (int i = 0; i < imgCount; i++) {
        imgs.push_back(imread(imgNames[i].c_str(), 0));
        imgs_rgb.push_back(imread(imgNames[i].c_str(), 1));
    }

    avgMatchDistances = Mat(imgs.size(), imgs.size(), CV_32FC1, Scalar(-1));

    while (imgCount > 1) {
        printf("Finding best match...\n");
        vector<int> bestMatches = findBestMatch(imgs, descriptorMatcher);

        printf("Stitching images %d and %d\n", bestMatches[0], bestMatches[1]);
        imwrite("stitching1.jpg", imgs_rgb[bestMatches[0]]);
        imwrite("stitching2.jpg", imgs_rgb[bestMatches[1]]);
        imgs_rgb[bestMatches[0]] = stitchImages(imgs[bestMatches[0]], imgs[bestMatches[1]], imgs_rgb[bestMatches[0]], imgs_rgb[bestMatches[1]], descriptorMatcher);

        Mat stitchedGray;
        cvtColor(imgs_rgb[bestMatches[0]], stitchedGray, CV_RGB2GRAY);
        imgs[bestMatches[0]] = stitchedGray;

        //Newly stitched image is stored in bestMatches[0], so we erase image at bestMatches[1]
        imgs[bestMatches[1]].release();

        imwrite("result.jpg", imgs_rgb[bestMatches[0]]);
        imgCount--;
    }
}

Mat stitchImages(Mat img1, Mat img2, Mat img1rgb, Mat img2rgb, Ptr<GenericDescriptorMatcher> descriptorMatcher) {
    Size size1 = img1.size();
    Size size2 = img2.size();

    //Create a large image with image2 in the center so we can draw image1 in
    printf("Setting up result image...\n");
    Mat result(Size(size2.width * 3, size2.height * 3), img2rgb.type(), Scalar(0,0,0));
    Rect img2ROI = Rect(size2.width, size2.height, size2.width, size2.height);
    Mat centreOfResult = result(img2ROI);
    img2rgb.copyTo(centreOfResult);

    printf("Result rows %d, result cols %d\n", result.rows, result.cols);
    printf("Size width %d, size height %d\n", size2.width, size2.height);

    //Tweak this value, lower values detects more keypoints
    //For demo:
    //horizontalAndVertical: 5.0e3
    //blurring: 4.0e3
    //horizontal: 5.0e3 slow but nice
    SURF surf_extractor(5.0e3);

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
    drawMatches(img1rgb, keypoints1, img2rgb, keypoints2, matches1to2, img_corr);
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
    //Higher values seems to accept worse transformations
    //For demo:
    //horizontalAndVertical: 50
    //blurring: 90
    //horizontal: 90
    Mat H = findHomography(Mat(points1), Mat(points2), RANSAC, 90);

    printf("Applying perspective warp...\n");
    warpPerspective(img1rgb, result, H, result.size(), INTER_LINEAR, BORDER_TRANSPARENT);
    Mat resultGray;
    cvtColor(result, resultGray, CV_RGB2GRAY);

    printf("Cropping image...\n");
    imwrite("resultUncropped.jpg", result);
    Mat croppedResult = cropBlack(result, resultGray);

    //Attempt to detect when a correct transform could not be found and return the first input image
    if ((croppedResult.rows > 0.98 * result.rows) && (croppedResult.cols > 0.98 * result.cols)) {
        printf("Image explosion detected! Dropping stitched image and returning larger input image.\n");
        if ((img1rgb.rows * img1rgb.cols) > (img2rgb.rows * img2rgb.cols)) {
            return img1rgb;
        } else {
            return img2rgb;
        }
    } else {
        return croppedResult;
    }
}

//Crops toCrop to a rectangular image by looking at the first and last non-zero pixels
Mat cropBlack(Mat toCrop, Mat toCropGray) {
    int minCol = toCropGray.cols;
    int minRow = toCropGray.rows;
    int maxCol = 0;
    int maxRow = 0;
    for (int i = 0; i < toCropGray.rows - 3; i++) {
        for (int j = 0; j < toCropGray.cols; j++) {
            if (toCropGray.at<char>(i, j) != 0) {
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

//Looks for the next images to stitch together by "quickly" testing all permutations
vector<int> findBestMatch(vector<Mat> imgs, Ptr<GenericDescriptorMatcher> descriptorMatcher) {
    //Use a high threshold so as to get fewer and stronger keypoints
    //Tweak this value, lower values detects more keypoints
    //If you get a failure on this step, it's probably because 0 matches were found, so reduce the threshold
    //20.0e3 worked well for horizontal I think
    //For demo:
    //horizontalAndVertical: 9.0e3
    //blurring: 14.0e3
    //horizontal: 20.0e3
    SURF surf_extractor(20.0e3);

    vector<KeyPoint> keypoints[imgs.size()];
    bool keypointsCalced[imgs.size()];
    for (int i = 0; i < imgs.size(); i++) {
        keypointsCalced[i] = false;
    }

    for (int i = 0; i < imgs.size(); i++) {
        for (int j = i + 1; j < imgs.size(); j++) {
            if (avgMatchDistances.at<float>(i, j) == -1) {
                printf("Must recalculate match between %d and %d\n", i, j);
                if (!keypointsCalced[i]) {surf_extractor(imgs[i], Mat(), keypoints[i]); keypointsCalced[i] = true;}
                if (!keypointsCalced[j]) {surf_extractor(imgs[j], Mat(), keypoints[j]); keypointsCalced[j] = true;}

                vector<DMatch> matches1to2;
                descriptorMatcher->match(imgs[i], keypoints[i], imgs[j], keypoints[j], matches1to2);

                float sum = 0;
                for (int k = 0; k < matches1to2.size(); k++) {sum += matches1to2[k].distance;}
                avgMatchDistances.at<float>(i, j) = sum / matches1to2.size();
                printf("Got %d matches and average match distance %f\n", matches1to2.size(), avgMatchDistances.at<float>(i, j));
            }
        }
    }

    float minDistance = 9001;
    int minIndex1 = 0;
    int minIndex2 = 0;
    for (int i = 0; i < imgs.size(); i++) {
        for (int j = i + 1; j < imgs.size(); j++) {
            if (avgMatchDistances.at<float>(i, j) > 0) {
                printf("Average match distance between %d and %d was %f\n", i, j, avgMatchDistances.at<float>(i, j));
                if (avgMatchDistances.at<float>(i, j) < minDistance) {
                    minDistance = avgMatchDistances.at<float>(i, j);
                    minIndex1 = i;
                    minIndex2 = j;
                }
            }
        }
    }
    printf("Best match was between %d and %d: %f\n", minIndex1, minIndex2, minDistance);

    //Now anything involving minIndex1 will need to be recalculated
    for (int i = 0; i < imgs.size(); i++) {
        if (avgMatchDistances.at<float>(i, minIndex1) != -2) {
            avgMatchDistances.at<float>(i, minIndex1) = -1;
            avgMatchDistances.at<float>(minIndex1, i) = -1;
        }
    }

    //And since minIndex2 will be deleted, we must indicate this using -2
    for (int i = 0; i < imgs.size(); i++) {
        avgMatchDistances.at<float>(i, minIndex2) = -2;
        avgMatchDistances.at<float>(minIndex2, i) = -2;
    }

    vector<int> minIndexes;
    minIndexes.push_back(minIndex1);
    minIndexes.push_back(minIndex2);
    return minIndexes;
}

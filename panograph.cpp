#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdio>

using namespace cv;

Mat stitchImages(Mat img1, Mat img2, Mat img1rgb, Mat img2rgb, Ptr<GenericDescriptorMatcher> descriptorMatcher);
Mat cropBlack(Mat toCrop, Mat toCropGray);
vector<int> findBestMatch(Mat img, vector<Mat> imgs, Ptr<GenericDescriptorMatcher> descriptorMatcher);

vector<float> minAvg;
vector<int> minIndex;
bool firstTime;

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
    vector<Mat> imgs;
    vector<Mat> imgs_rgb;
    for (int i = 0; i < imgCount; i++) {
        imgs.push_back(imread(imgNames[i].c_str(), 0));
        imgs_rgb.push_back(imread(imgNames[i].c_str(), 1));
    }

    //Set up the min avg and index vectors
    for (int i = 0; i < imgs.size(); i++) {
        minAvg.push_back(0);
        minIndex.push_back(-1);
    }
    //Indicate that this is the first time we're checking matches
    firstTime = true;

    while (imgs.size() > 1) {
        vector<int> bestMatches = findBestMatch(imgs.front(), imgs, descriptorMatcher);

        printf("Stitching images %d and %d\n", bestMatches[0], bestMatches[1]);
        imwrite("blah1.jpg", imgs_rgb[bestMatches[0]]);
        imwrite("blah2.jpg", imgs_rgb[bestMatches[1]]);
        imgs_rgb[bestMatches[0]] = stitchImages(imgs[bestMatches[0]], imgs[bestMatches[1]], imgs_rgb[bestMatches[0]], imgs_rgb[bestMatches[1]], descriptorMatcher);

        Mat stitchedGray;
        cvtColor(imgs_rgb[bestMatches[0]], stitchedGray, CV_RGB2GRAY);
        imgs[bestMatches[0]] = stitchedGray;

        //Newly stitched image is stored in bestMatches[0], so we erase image at bestMatches[1]
        printf("Erasing image %d\n", bestMatches[1]);
        imgs.erase(imgs.begin() + bestMatches[1]);
        imgs_rgb.erase(imgs_rgb.begin() + bestMatches[1]);
    }
    imwrite("result.jpg", imgs_rgb[0]);
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
    Mat H = findHomography(Mat(points1), Mat(points2), RANSAC, 50);

    printf("Applying perspective warp...\n");
    warpPerspective(img1rgb, result, H, result.size(), INTER_LINEAR, BORDER_TRANSPARENT);
    Mat resultGray;
    cvtColor(result, resultGray, CV_RGB2GRAY);

    printf("Cropping image...\n");
    imwrite("resultUncropped.jpg", result);
    Mat croppedResult = cropBlack(result, resultGray);

    return croppedResult;
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
vector<int> findBestMatch(Mat img, vector<Mat> imgs, Ptr<GenericDescriptorMatcher> descriptorMatcher) {
    //Use a high threshold so as to get fewer and stronger keypoints
    SURF surf_extractor(20.0e3);

    //We don't need to check the last image, since it would have already been matched against all the others
    for (int i = 0; i < imgs.size(); i++) {
        //if (minIndex[i] < 0) {
            printf("Image %d must be recalculated\n", i);

            //Calculate keypoints of current image
            vector<KeyPoint> keypoints1;
            surf_extractor(imgs[i], Mat(), keypoints1);

            float currAvg[imgs.size()];

            //The first time we do matches, we check all images, so we don't need to redo any of the earlier ones
            //Later times, we do need to go back and recheck them
            int j = firstTime ? i + 1 : 0;
            for (; j < imgs.size(); j++) {
                //Calculate keypoints of other image
                vector<KeyPoint> keypoints2;
                surf_extractor(imgs[j], Mat(), keypoints2);
                vector<DMatch> matches1to2;

                //Find matches between current image and other image
                descriptorMatcher->match(imgs[i], keypoints1, imgs[j], keypoints2, matches1to2);

                //Find average of distances of matches
                for (int k = 0; k < matches1to2.size(); k++) {
                    currAvg[j] = currAvg[j] + matches1to2[k].distance;
                }
                currAvg[j] = currAvg[j] / matches1to2.size();
            }

            //Find the lowest average between current image and other images
            float currMinAvg = 90001;
            int currMinIndex = -1;
            for (int j = i + 1; j < imgs.size(); j++) {
                if (currAvg[j] < currMinAvg) {
                    currMinAvg = currAvg[j];
                    currMinIndex = j;
                }
            }
            minAvg[i] = currMinAvg;
            minIndex[i] = currMinIndex;
        //} else {
        //    printf("Image %d had already been calculated\n", i);
        //}

        //firstTime = false;
    }

    //Find lowest average between all images
    float finalMinAvg = 90001;
    int finalMinIndex1 = 0;
    int finalMinIndex2 = 0;
    for (int j = 0; j < imgs.size(); j++) {
        printf("Image %d had highest matches with %d, got %f\n", j, minIndex[j], minAvg[j]);
        if (minAvg[j] < finalMinAvg) {
            finalMinAvg = minAvg[j];
            finalMinIndex1 = j;
            finalMinIndex2 = minIndex[j];
        }
    }

    printf("Highest number of matches was between %d and %d, there were %f\n", finalMinIndex1, finalMinIndex2, finalMinAvg);

    //Return the indexes of the images that had the lowest average distance
    vector<int> minIndexes;
    minIndexes.push_back(finalMinIndex2);
    minIndexes.push_back(finalMinIndex1);

    //Now reset the average match values for the first image we are about to stitch
    //minIndex[minIndexes[0]] = -1;

    //And erase the values for the second one
    //minIndex.erase(minIndex.begin() + minIndexes[1]);


    return minIndexes;
}

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main()
{
	//Loading the Images
	//In this part you can change the image you want to use as an object and scene image
	Mat image_obj = imread("dataset1/obj1.jpg", IMREAD_GRAYSCALE);
	Mat image_scene = imread("dataset1/scene1.jpg", IMREAD_GRAYSCALE);

	//If the images are not loaded correctly, it shows an error message.
	if (!image_obj.data)
	{
		cout << " --(!) Error reading image1 " << endl;
		return -1;
	}
	if (!image_scene.data)
	{
		cout << " --(!) Error reading image2 " << endl;
		return -1;
	}


	//Detecting the keypoints using SIFT Detector
	Ptr<SIFT> detector = SIFT::create();
	vector<KeyPoint> keypoints_obj, keypoints_scene;
	detector->detect(image_obj, keypoints_obj);
	detector->detect(image_scene, keypoints_scene);

	//Calculating descriptors 
	Mat descriptors_obj, descriptors_scene;
	detector->compute(image_obj, keypoints_obj, descriptors_obj);
	detector->compute(image_scene, keypoints_scene, descriptors_scene);

	//Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_obj, descriptors_scene, matches);

	Mat img_matches;
	drawMatches(image_obj, keypoints_obj, image_scene, keypoints_scene,
		matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//Localizing the object
	vector<Point2f> obj;
	vector<Point2f> scene;

	for (int i = 0; i < matches.size(); i++)
	{
		//Getting the keypoints from the  matches
		obj.push_back(keypoints_obj[matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[matches[i].trainIdx].pt);
	}


	//FindHomography
	Mat H = findHomography(obj, scene, RANSAC);
	//Get the corners of the object which needs to be detected.
	vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(image_obj.cols, 0);
	obj_corners[2] = Point(image_obj.cols, image_obj.rows);
	obj_corners[3] = Point(0, image_obj.rows);

	//Getting the corners of the object from the scene image
	std::vector<Point2f> scene_corners(4);

	//Getting the perspectiveTransform
	perspectiveTransform(obj_corners, scene_corners, H);

	//Drawing lines between the corners 
	line(img_matches, scene_corners[0] + Point2f(image_obj.cols, 0), scene_corners[1] + Point2f(image_obj.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(image_obj.cols, 0), scene_corners[2] + Point2f(image_obj.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(image_obj.cols, 0), scene_corners[3] + Point2f(image_obj.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(image_obj.cols, 0), scene_corners[0] + Point2f(image_obj.cols, 0), Scalar(0, 255, 0), 4);

	//Marking and Showing detected image from the scene 
	namedWindow("DetectedImage", WINDOW_NORMAL);
	imshow("DetectedImage", img_matches);
	waitKey(0);

}
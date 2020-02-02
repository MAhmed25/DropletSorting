#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  
using namespace std;
using namespace cv;


Mat backgroundMask(Mat theImage);	//returns a matrix which you can add to original image to remove
									//the background

void fourierFunction(Mat& imgToTransform);

Mat shiftImage(Mat& img, int xShift, int yShift);

int findCircles(vector<vector<Point>>& contours, Mat &referenceImage);

int main(int argc, char* argv[])
{
	const char* inputVideoName = "images/cell2.avi";
	const char* frontCellCurve = "images/frontCellCurve.png";
	const char* backCellCurve = "images/backCellCurve.png";


	cout << inputVideoName << endl;
	VideoCapture inputVideo(inputVideoName);
	Mat inputFrames, thresholded, sobel, borderMask, labelled, backCurve, frontCurve, circles, sobelX, summed,
		edgeSummed,
		g1,g2, diff, dt, dots, finale;
	
	

	inputVideo >> inputFrames;

	cvtColor(inputFrames, inputFrames, COLOR_BGR2GRAY);

	// borderMask = backgroundMask(inputFrames);

	//All windows for the curve of the droplet
	namedWindow("input", WINDOW_AUTOSIZE);

	//namedWindow("Guassian_Blurred", WINDOW_AUTOSIZE);
	//namedWindow("SobelX", WINDOW_AUTOSIZE);
	namedWindow("Threshold", WINDOW_AUTOSIZE);
	//namedWindow("difference", WINDOW_AUTOSIZE);

	//namedWindow("G1", WINDOW_AUTOSIZE);
	//namedWindow("G2", WINDOW_AUTOSIZE);
	//namedWindow("ThresholdDifference", WINDOW_AUTOSIZE);
	namedWindow("justDots", WINDOW_AUTOSIZE);

	namedWindow("edgeSummed", WINDOW_AUTOSIZE);

	namedWindow("finale", WINDOW_AUTOSIZE);

	//namedWindow("lined", WINDOW_AUTOSIZE);

	int numberOfCircles;

	/*
	cv::Mat morphCurves1 = (cv::Mat_<uchar>(9, 5) << 
		1, 1, 1, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 1, 1);

	cv::Mat morphCurves2 = (cv::Mat_<uchar>(9, 5) <<
		0, 0, 0, 0, 0, 0, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 1, 1, 1,
		0, 0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 1, 0, 0, 0, 0,
		1, 1, 1, 0, 0, 0, 0, 0, 0);

	cv::Mat morphCurves3 = (cv::Mat_<uchar>(9, 5) <<
		0, 0, 0, 0, 0, 0, 1, 1, 1,
		0, 0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 1, 1, 1, 0, 0, 0, 0,
		1, 1, 1, 0, 0, 0, 0, 0, 0,
		1, 1, 1, 0, 0, 0, 0, 0, 0);

	cv::Mat morphCurves4 = (cv::Mat_<uchar>(9, 5) <<
		1, 1, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 1, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 1, 1,
		0, 0, 0, 0, 0, 0, 1, 1, 1);
	*/

	cv::Mat frontCurveMorph = imread(frontCellCurve, 0);
	threshold(frontCurveMorph, frontCurveMorph, 250, 255, THRESH_BINARY);

	cv::Mat backCurveMorph = imread(backCellCurve, 0);
	threshold(backCurveMorph, backCurveMorph, 250, 255, THRESH_BINARY);

	cv::Mat squareMorph = cv::Mat::ones(cv::Size(3, 3), CV_8U);

	cv::Mat morphLineErosion = (cv::Mat_<uchar>(5, 5) <<
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		1, 1, 1, 1, 1,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0);

	Mat_<int> ellipses(7, 7);
	ellipses <<
		  0,  0,  0, -1,  0,  0,  0,
		  0,  0, -1,  1, -1,  0,  0,
		  0, -1,  1,  1,  1, -1,  0,
		 -1,  1,  1,  0,  1,  1, -1,
		  0, -1,  1,  1,  1, -1,  0,
		  0,  0, -1,  1, -1,  0,  0,
		  0,  0,  0, -1,  0,  0,  0;



	Mat ellipseMorph = getStructuringElement(MORPH_ELLIPSE, Size(5,5));

	Mat_<float> sobelMinusX(3, 3);
	sobelMinusX << 1, 0, -1, 2, 0, -2, 1, 0, -1;

	vector<vector<Point>> contours;
	waitKey();

	while (!inputFrames.empty())
	{

		imshow("input", inputFrames);


		// for detecting the larger circles
		// First construction of a DoG filter
		GaussianBlur(inputFrames, g1, Size(9, 9), 1, 1); 
		//imshow("G1", g1);

		GaussianBlur(inputFrames, g2, Size(15, 15), 6, 6);
		//imshow("G2", g2);

		diff = (g1 - g2);
		//imshow("difference", diff);
		// end of DoG filterintg

		// Threshold the result for the circles as they will have higher intensity
		threshold(diff, dt, 40, 255, THRESH_BINARY);
		//imshow("ThresholdDifference", dt);

		// remove non-elliptic shapes
		morphologyEx(dt, dots, MORPH_OPEN, ellipseMorph, Point(-1,-1), 1);
		imshow("justDots", dots);



		Sobel(inputFrames, sobelX, CV_8U, 1, 0, 3); //standard Sobel in X direction
		//imshow("SobelX", sobelX);

		// Threshold result
		threshold(sobelX, sobelX, 80, 255, THRESH_BINARY);
		imshow("Threshold", sobelX);
		waitKey();
		frontCurve = cv::Mat::zeros(sobelX.size(), sobelX.type());
		backCurve = cv::Mat::zeros(sobelX.size(), sobelX.type());

		//attempt to remove curves which aren't correct needs to be made better with a bigger morphCurve kernel
		//morphologyEx(sobelX, frontCurve, MORPH_OPEN, frontCurveMorph, Point(-1, -1), 1);
		erode(sobelX, frontCurve, frontCurveMorph);
		dilate(frontCurve, frontCurve, backCurveMorph);

		erode(sobelX, backCurve, backCurveMorph);
		dilate(backCurve, backCurve, frontCurveMorph);

		filter2D(inputFrames, edgeSummed, CV_8U, ellipses);
		morphologyEx(edgeSummed, edgeSummed, MORPH_OPEN, squareMorph);
		imshow("edgeSummedpre", edgeSummed);

		borderMask = cv::Mat::zeros(cv::Size(256,256), CV_8U);

		line(borderMask, Point(0, 60), Point(255, 60), Scalar(255), 2);
		line(borderMask, Point(0, 209), Point(255, 209), Scalar(255), 2);
		floodFill(borderMask, Point(0, 0), Scalar(255));
		floodFill(borderMask, Point(0, 255), Scalar(255));
		imshow("lined", borderMask);

		threshold(edgeSummed, edgeSummed, 150, 255, THRESH_BINARY);
		edgeSummed = edgeSummed - borderMask;
		imshow("edgeSummed", edgeSummed);

		finale = edgeSummed + dots + frontCurve + backCurve;
		imshow("finale", finale);

		waitKey(4); //perform these operations once every 4s
		inputVideo >> inputFrames;
		cvtColor(inputFrames, inputFrames, COLOR_BGR2GRAY);
	}

	waitKey();
}

 int findCircles(vector<vector<Point>> &contours, Mat & referenceImage)
{
	 Rect box;
	 int numberOfCurves = 0;
	 int numberOfPoints = 0;
	 int numberOfNoise = 0;
	 int ignored = 0;
	 int maxRadius = 15;
	 int xLim = 0;

	 struct curveContainer {
		 bool type; //front facing = true, backwards facing = false;
		 int x, y, width, height; //Left most coords
	 };
	 vector<curveContainer> curves;
	 curves.reserve(3); // generally wont see more than 3 curves

	 struct pointContainer {
		 int centerX, centerY, height;
	 };
	 vector<pointContainer> points;
	 points.reserve(10);

	 for (size_t i = 0; i < contours.size(); i++)
	 {
		 box = boundingRect(contours[i]);
		 // First check if its on the edges if it is, just exit the for loop
		 if (box.x + box.width >= 254 || box.x <= 1 ) 
		 { 
			 ignored++;
		 }
		 else 
		 {

		 int topLeft = (int)referenceImage.at<uchar>(box.y, box.x);
		 int bottomLeft = (int)referenceImage.at<uchar>(box.y + box.height - 1, box.x);

		 // Curves are really big in general
			if (box.area() >= 3000)
			{
				if (topLeft || bottomLeft >= 255) // front facing
			 {
				 curves.push_back(curveContainer());
				 curves[numberOfCurves].type = true;
				 curves[numberOfCurves].x = box.x;
				 curves[numberOfCurves].y = box.y;
				 curves[numberOfCurves].width = box.width;
				 curves[numberOfCurves].height = box.height;
				 numberOfCurves++;
			 }
				else // if not front facing, must be backwards facing
				{
					curves.push_back(curveContainer());
					curves[numberOfCurves].type = false;
					curves[numberOfCurves].x = box.x;
					curves[numberOfCurves].y = box.y;
					curves[numberOfCurves].width = box.width;
					curves[numberOfCurves].height = box.height;
					numberOfCurves++;
				}
			}
			else // if the area is not greater than 4000, it isn't a curve
			{
				{
					points.push_back(pointContainer());
					points[numberOfPoints].centerX = box.x + box.width / 2;
					points[numberOfPoints].centerY = box.y + box.height / 2;
					points[numberOfPoints].height = box.height;
					numberOfPoints++;
				}
			}
		 }
	 }

	 for (int i = 0; i < numberOfCurves; i++)
	 {
		 int YBoundaryTop = curves[i].y;
		 int YBoundaryBottom = curves[i].y + curves[i].height;
		 int XBoundary = curves[i].x;
		 if (!curves[i].type) // if backwards facing
		 {
			 xLim = XBoundary;
			 XBoundary = curves[i].x + curves[i].width;
		 }
		 if (numberOfNoise >= points.size()) { break; }
		 for (int j = 0; j < numberOfPoints;  j++)
		 {
			 if ((points[j].centerX >= XBoundary - 30 && points[j].centerX <= XBoundary + 30 && (points[j].centerY >= YBoundaryTop - 30 && points[j].centerY <= YBoundaryTop + 30
				 || points[j].centerY >= YBoundaryBottom - 30 && points[j].centerY <= YBoundaryBottom + 30)) || points[j].height >= maxRadius)
			 {
				 numberOfNoise++;
			 }
			 else if (points[j].centerX < xLim)
			 {
				 numberOfNoise++;
			 }

		 }
	 }

	 cout << " NOC: " << numberOfCurves << " NON: " << numberOfNoise << " IG: " << ignored << " TOT: " << contours.size() << endl;
	 return contours.size() - numberOfCurves - numberOfNoise - ignored;
}

/* creates the background mask */
// No longer actually used needs more work
Mat backgroundMask(Mat theImage)
{
	Mat_<float> kernelsobelY(3, 3), negKernelSobelY(3, 3);
	kernelsobelY << -3, -1, -3, 0, 0, 0, 3, 1, 3;
	negKernelSobelY << 3, 1, 3, 0, 0, 0, -3, -1, -3;

	Mat sobelYEdge, negSobelYEdge, backGroundMask, floodFilledTop, floodFilledBottom, inverted;

	filter2D(theImage, sobelYEdge, CV_8U, kernelsobelY);
	filter2D(theImage, negSobelYEdge, CV_8U, negKernelSobelY);

	namedWindow("posSobelY", WINDOW_AUTOSIZE);
	imshow("posSobelY", sobelYEdge);

	namedWindow("negSobelY", WINDOW_AUTOSIZE);
	imshow("negSobelY", negSobelYEdge);

	floodFilledTop = negSobelYEdge.clone();
	floodFilledBottom = sobelYEdge.clone();

	threshold(floodFilledTop, floodFilledTop, 60, 255, THRESH_BINARY);
	inverted = floodFilledTop.clone();
	inverted = ~inverted;
	floodFill(floodFilledTop, cv::Point(128, 0), Scalar(255));
	bitwise_and(floodFilledTop, inverted, floodFilledTop);
	namedWindow("TopFF", WINDOW_AUTOSIZE);
	imshow("TopFF", floodFilledTop);

	threshold(floodFilledBottom, floodFilledBottom, 60, 255, THRESH_BINARY);
	inverted = floodFilledBottom.clone();
	inverted = ~inverted;
	floodFill(floodFilledBottom, cv::Point(128, 255), Scalar(255));
	bitwise_and(floodFilledBottom, inverted, floodFilledBottom);
	namedWindow("BottomFF", WINDOW_AUTOSIZE);
	imshow("BottomFF", floodFilledBottom);

	backGroundMask = floodFilledBottom + floodFilledTop;
	namedWindow("final", WINDOW_AUTOSIZE);
	//imshow("final", backGroundMask);

	return backGroundMask;
};

// Does not wrap around
Mat shiftImage(Mat& img, int xShift, int yShift) 
{
	Mat shiftedImg = cv::Mat::zeros(img.size(), img.type());
	
	// Bound xShift and yShift to img limits.

	xShift = std::max(-img.cols, std::min(xShift, img.cols));
	yShift = std::max(-img.rows, std::min(yShift, img.rows));

	for (int i = 0; i < (img.cols - (int)abs(xShift)); i++)
	{
		xShift < 0 ? img.col(img.cols - i - 1).copyTo(shiftedImg.col(shiftedImg.cols + xShift - i - 1)) // if negative, copy right to left
			: img.col(i).copyTo(shiftedImg.col(xShift + i)); // if positive, copy left to right
	}
	
	// to store the intermediate xShifted image
	// in preparation for yShift
	Mat shiftedXImg = cv::Mat::zeros(img.size(), img.type());

	for (int i = 0; i < (shiftedImg.rows - (int)abs(yShift)); i++)
	{
		yShift < 0 ? shiftedImg.row(shiftedImg.rows - i - 1).copyTo(shiftedXImg.row(shiftedXImg.rows + yShift - i - 1)) // if negative, copy bottom to top
			: shiftedImg.row(i).copyTo(shiftedXImg.row(yShift + i)); // if positive, copy top to bottom
	}

	return shiftedXImg;
}
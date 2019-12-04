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

int findCircles(vector<vector<Point>>& contours, Mat &referenceImage);

int main(int argc, char* argv[])
{
	const char* inputVideoName = "videos/cell2.avi";

	cout << inputVideoName << endl;
	VideoCapture inputVideo(inputVideoName);
	Mat inputFrames, thresholded, sobel, borderMask, labelled, Curves, circles, sobelX, summed, edge1Mat,edge2Mat, edge3Mat, edge4Mat,
		edgeSummed,
		g1,g2, diff, dt, dots, finale;

	inputVideo >> inputFrames;

	int width = inputFrames.cols;
	int height = inputFrames.rows;

	cvtColor(inputFrames, inputFrames, COLOR_BGR2GRAY);

	// borderMask = backgroundMask(inputFrames);

	//All windows for the curve of the droplet
	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("Guassian_Blurred", WINDOW_AUTOSIZE);
	namedWindow("SobelX", WINDOW_AUTOSIZE);
	namedWindow("Threshold", WINDOW_AUTOSIZE);
	namedWindow("difference", WINDOW_AUTOSIZE);

	namedWindow("G1", WINDOW_AUTOSIZE);
	namedWindow("G2", WINDOW_AUTOSIZE);
	namedWindow("ThresholdDifference", WINDOW_AUTOSIZE);
	namedWindow("justDots", WINDOW_AUTOSIZE);


	namedWindow("edge1", WINDOW_AUTOSIZE);
	namedWindow("edge2", WINDOW_AUTOSIZE);
	namedWindow("edge3", WINDOW_AUTOSIZE);
	namedWindow("edge4", WINDOW_AUTOSIZE);
	namedWindow("edgeSummed", WINDOW_AUTOSIZE);


	namedWindow("finale", WINDOW_AUTOSIZE);

	//namedWindow("lined", WINDOW_AUTOSIZE);

	int numberOfCircles;

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

	cv::Mat morphLineErosion = (cv::Mat_<uchar>(5, 5) <<
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		1, 1, 1, 1, 1,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0);

	Mat_<int> edge1(3, 3);
	edge1 << 2, 2, -1, 
			 2, -1, -1, 
			-1, -1, -1;

	Mat_<int> edge2(3, 3);
	edge2 << -1, 2, 2,
			-1, -1, 2,
			-1, -1, -1;

	Mat_<int> edge3(3, 3);
	edge3 << -1, -1, -1,
			 -1, -1, 2,
			 -1, 2, 2;

	Mat_<int> edge4(3, 3);
	edge4 << -1, -1, -1,
			 2, -1, -1,
			 2, 2, -1;

	Mat_<int> ellipsess(7, 7);
	ellipsess <<
		0, 0, 0, -1, 0, 0, 0,
		0, 0, -1, 1, -1, 0, 0,
		0, -1, 1, 1, 1, -1, 0,
		-1, 1, 1, 0, 1, 1, -1,
		0, -1, 1, 1, 1, -1, 0,
		0, 0, -1, 1, -1, 0, 0,
		0, 0, 0, -1, 0, 0, 0;



	Mat ellipseMorph = getStructuringElement(MORPH_ELLIPSE, Size(5,5));

	Mat_<float> sobelMinusX(3, 3);
	sobelMinusX << 1, 0, -1, 2, 0, -2, 1, 0, -1;

	vector<vector<Point>> contours;

	while (!inputFrames.empty())
	{
		imshow("input", inputFrames);


		// for detecting the larger circles
		// First construction of a DoG filter
		GaussianBlur(inputFrames, g1, Size(9, 9), 1, 1); 
		imshow("G1", g1);

		GaussianBlur(inputFrames, g2, Size(15, 15), 6, 6);
		imshow("G2", g2);

		diff = (g1 - g2);
		imshow("difference", diff);
		// end of DoG filterintg

		// Threshold the result for the circles as they will have higher intensity
		threshold(diff, dt, 40, 255, THRESH_BINARY);
		imshow("ThresholdDifference", dt);

		// remove non-elliptic shapes
		morphologyEx(dt, dots, MORPH_OPEN, ellipseMorph, Point(-1,-1), 1);
		imshow("justDots", dots);


		// For the edges of the circle
		GaussianBlur(inputFrames, inputFrames, Size(9, 9), 1,1);
		imshow("Guassian_Blurred", inputFrames);

		Sobel(inputFrames, sobelX, CV_8U, 1, 0, 3); //standard Sobel in X direction
		imshow("SobelX", sobelX);

		// Threshold result
		threshold(sobelX, Curves, 80, 255, THRESH_BINARY);
		imshow("Threshold", Curves);

		//attempt to remove curves which aren't correct needs to be made better with a bigger morphCurve kernel
		morphologyEx(Curves, Curves, MORPH_CLOSE, morphCurves1, Point(-1, -1), 8);
		morphologyEx(Curves, Curves, MORPH_CLOSE, morphCurves2, Point(-1,-1), 8);
		morphologyEx(Curves, Curves, MORPH_CLOSE, morphCurves3, Point(-1, -1), 8);
		morphologyEx(Curves, Curves, MORPH_CLOSE, morphCurves4, Point(-1, -1), 8);
		imshow("closed", Curves);


		// For the smaller circles.
		filter2D(inputFrames, edge1Mat, CV_8U, edge1);
		imshow("edge1", edge1Mat);

		filter2D(inputFrames, edge2Mat, CV_8U, edge2);
		imshow("edge2", edge2Mat);

		filter2D(inputFrames, edge3Mat, CV_8U, edge3);
		imshow("edge3", edge3Mat);

		filter2D(inputFrames, edge4Mat, CV_8U, edge4);
		imshow("edge4", edge4Mat);

		filter2D(inputFrames, edgeSummed, CV_8U, ellipsess);
		
		borderMask = cv::Mat::zeros(cv::Size(256,256), CV_8U);

		line(borderMask, Point(0, 60), Point(255, 60), Scalar(255), 2);
		line(borderMask, Point(0, 209), Point(255, 209), Scalar(255), 2);
		floodFill(borderMask, Point(0, 0), Scalar(255));
		floodFill(borderMask, Point(0, 255), Scalar(255));
		imshow("lined", borderMask);

		threshold(edgeSummed, edgeSummed, 160, 255, THRESH_BINARY);
		edgeSummed = edgeSummed - borderMask;
		imshow("edgeSummed", edgeSummed);

		finale = edgeSummed + dots - Curves;
		imshow("finale", finale);

		/*
		findContours(sobel, contours, RETR_LIST, CHAIN_APPROX_NONE);

		numberOfCircles = findCircles(contours, sobel);

		cout << "Number of circles = " << numberOfCircles << endl;
		*/


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


void fourierFunction(Mat& imgToTransform)
{
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(imgToTransform.rows);
	int n = getOptimalDFTSize(imgToTransform.cols); // on the border add zero values
	copyMakeBorder(imgToTransform, padded, 0, m - imgToTransform.rows, 0, n - imgToTransform.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
											// viewable image form (float between values 0 and 1).

	imshow("Input Image", imgToTransform);    // Show the result
	imshow("spectrum magnitude", magI);

}
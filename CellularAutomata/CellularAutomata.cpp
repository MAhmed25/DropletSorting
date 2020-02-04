#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  
using namespace std;
using namespace cv;


const int cellVelocity = 26;

Mat backgroundMask(Mat theImage);	//returns a matrix which you can add to original image to remove
									//the background

bool curveDetect(Mat& cellCurve, int span, int start, int threshold, int morphCols);

int findPosFirstWhite(Mat& frontCurve);

Mat largeDotDetection(Mat& input);

Mat shiftImage(Mat& img, int xShift, int yShift);

Mat integrateMatrices(Mat& accumalator, Mat& next);

int findCircles(vector<vector<Point>>& contours, Mat &referenceImage);

int main(int argc, char* argv[])
{
	const char* inputVideoName = "images/cell2.avi";
	const char* frontCellCurve = "images/frontCellCurve.png";
	const char* backCellCurve = "images/backCellCurve.png";

	cout << inputVideoName << endl;
	VideoCapture inputVideo(inputVideoName);
	Mat inputFrames, thresholded, sobel, borderMask, backCurve, frontCurve, sobelX, summed,
		smallDots, finale, largeDots;

	Mat test;
	

	inputVideo >> inputFrames;

	cvtColor(inputFrames, inputFrames, COLOR_BGR2GRAY);

	// borderMask = backgroundMask(inputFrames);

	//All windows for the curve of the droplet
	namedWindow("input", WINDOW_AUTOSIZE); //Stores the input frame

	namedWindow("finale", WINDOW_AUTOSIZE);

	cv::Mat frontCurveMorph = imread(frontCellCurve, 0);
	threshold(frontCurveMorph, frontCurveMorph, 250, 255, THRESH_BINARY);

	int curveMorphSum = (int)sum(frontCurveMorph)[0];

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

	waitKey(NULL);

	/******* Temoprarily create the border mask by manually created lines to remove anything not in the *******/
	/******************************************* Droplet Channel **********************************************/

	borderMask = cv::Mat::zeros(cv::Size(256, 256), CV_8U);
	// Creates the north border
	line(borderMask, Point(0, 47), Point(255, 47), Scalar(255), 2);
	// Creates the south border
	line(borderMask, Point(0, 214), Point(255, 214), Scalar(255), 2);
	// Fills the north and south border to then later subtract from finale
	floodFill(borderMask, Point(0, 0), Scalar(255));
	floodFill(borderMask, Point(0, 255), Scalar(255));
	imshow("Border Mask", borderMask);

	/**********************************************************************************************************/

	Mat divider = cv::Mat::zeros(inputFrames.size(), inputFrames.type());
	int dividerXPos = 0;

	Mat accumalator = cv::Mat::zeros(inputFrames.size(), inputFrames.type());

	while (!inputFrames.empty())
	{

		imshow("input", inputFrames);
		threshold(inputFrames, test, 70, 255, THRESH_BINARY_INV);
		imshow("threshbin", test);

		/**********************************  Large Dot Detection  *************************************************/
		// Detect the larger dots using a difference of gaussian filter and store in largeDots matrix
		largeDots = largeDotDetection(inputFrames);
		imshow("large circles dog detection", largeDots); // Displays the result

		/**********************************************************************************************************/


		/************************** Cell front and back curve detection********************************************/

		Sobel(inputFrames, sobelX, CV_8U, 1, 0, 3); //standard Sobel in X direction
		// Threshold to convert to a binary format as well as remove small noise
		threshold(sobelX, sobelX, 80, 255, THRESH_BINARY);

		// Initialize the two matrices which will store the result of the morphological open operations
		// We will perform to single out the front and back of the cell
		frontCurve = cv::Mat::zeros(sobelX.size(), sobelX.type());
		backCurve = cv::Mat::zeros(sobelX.size(), sobelX.type());

		// Performs a morpholigical open, as well as using the opposite curve, i.e eroding with front, dilating with back
		// To get the correct orientation on the reconstructed curve
		// Front curve open operation
		erode(sobelX, frontCurve, frontCurveMorph);
		dilate(frontCurve, frontCurve, backCurveMorph);
		imshow("Front Cell", frontCurve);
		// Back curve open operation
		erode(sobelX, backCurve, backCurveMorph);
		dilate(backCurve, backCurve, frontCurveMorph);
		imshow("Back Cell", backCurve);
		// End result is 2 matrices which contain only the front and back curves of droplets

		/**********************************************************************************************************/


		/********************* Detection of smaller dots using a elliptical edge detection kernel *****************/

		// Filter with custom ellipse edge kernel
		filter2D(inputFrames, smallDots, CV_8U, ellipses);
		// Morph to remove any noise and keep significant gradients only (those which are elliptical in nature
		// such as the small dots
		morphologyEx(smallDots, smallDots, MORPH_OPEN, squareMorph);
		// Convert to binary to remove noise and leave just the dots
		smallDots = largeDotDetection(smallDots);

		threshold(smallDots, smallDots, 155, 255, THRESH_BINARY);

		/********************************************************************************************************/

		test = test - borderMask;
		floodFill(test, Point(0, 0), Scalar(255));
		floodFill(test, Point(0, 255), Scalar(255));
		threshold(test, test, 1, 255, THRESH_BINARY_INV);
		imshow("test2", test);

		/*************************************** Create the final matrix ****************************************/

		if (!finale.empty()){ accumalator = finale; }
		imshow("accumalator", accumalator);

		finale = smallDots + largeDots;
		bitwise_and(finale, test, finale);
		finale += frontCurve + backCurve;

		imshow("finale", finale);

		// New cell detected

		if (curveDetect(frontCurve, frontCurveMorph.cols * 1.5, 0, curveMorphSum, frontCurveMorph.cols))
		{
			int x = findPosFirstWhite(frontCurve);
			dividerXPos = x;
			line(divider, Point(x, 0), Point(x, 255), 255);
			floodFill(divider, Point(0, 0), 255);
			bitwise_and(divider, finale, finale);
			accumalator = cv::Mat::zeros(accumalator.size(), accumalator.type());
		}

		imshow("divider", divider);
		dividerXPos += cellVelocity;


		waitKey(); //perform these operations once every 4s
		inputVideo >> inputFrames;
		cvtColor(inputFrames, inputFrames, COLOR_BGR2GRAY);
	}

	waitKey();
}

 
// Does not wrap around shifts matrix in x and y direction
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

// Returns the result of the Difference-of-guassian filter for large dot detection
Mat largeDotDetection(Mat& input)
{
	Mat g1, g2, diff;

	GaussianBlur(input, g1, Size(9, 9), 1, 1);
	GaussianBlur(input, g2, Size(15, 15), 6, 6);

	diff = g1 - g2;
	// Threshold the result for the circles as they will have higher intensity
	threshold(diff, diff, 40, 255, THRESH_BINARY);
	// Remove any non-elliptical shapes through a morph open 
	morphologyEx(diff, diff, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)), Point(-1, -1), 1);

	// Return the large dot
	return diff;
}

bool curveDetect(Mat& cellCurve, int span, int start, int threshold, int morphCols)
{
	// Detecting if a new droplet is incoming will be done by calculating average of
	// the span width and see if it greater than some value in which case
	// We know a new cell is incoming
	// Consider that worst case scenario, value returned by front curve morph
	// will be just a thin curve assuming the morph only finds 1 matching type
	// Actual worst will be 0 but in that case you're out anyway
	// Can expand/use some predictive algos by basing it on previous time between new cells
	if (span < morphCols)
	{ 
		cout << "Warning; your span is shorter than"
		"your morph matrix column width will lead to errors" << endl; 
	}
	
	// Calculate sum of span columns starting from left
	int spanSum = 0;
	for (int i = start; i < start + span; i++)
	{
		for (int j = 0; j < cellCurve.rows; j++)
		{
			spanSum += (int)cellCurve.at<uchar>(j, i);
		}
	}

	return spanSum > threshold ? true : false;
}

// Can probably encode this with curveDetect fn above but alas i am lazy
int findPosFirstWhite(Mat& frontCurve)
{
	// Will return the position, as an int, of the first column to contain
	// any white pixel
	// start from middle of matrix array as to reduce chance of detecting wrong curve
	// not a general purpose fn
	for (int i = ( frontCurve.cols / 2 ); i >= 0; i--)
	{
		for (int j = 0; j < frontCurve.rows; j++)
		{
			if ((int)frontCurve.at<uchar>(j, i) != 0) { return i; }
		}
	}
	return -1; // None found should not happen....
}

Mat integrateMatrices(Mat& accumalator, Mat& next)
{
	return next + shiftImage(accumalator, cellVelocity, 0);
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

int findCircles(vector<vector<Point>>& contours, Mat& referenceImage)
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
		if (box.x + box.width >= 254 || box.x <= 1)
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
		for (int j = 0; j < numberOfPoints; j++)
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

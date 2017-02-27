#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std; 

Mat src, src_gray;
/*void cornerHarris_demo()
{
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);
	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;
	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);
	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			//if ((int)dst_norm.at<float>(j, i) > 90)
			//{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
				circle(src, Point(i, j), 5, Scalar(255, 0, 0), -1, 8, 0);
			//}
		}
	}
	/// Showing the result
	cv::imshow("corner", dst_norm_scaled);
	cv::imshow("src", src);
}*/
Point tcircle(Point pt1, Point pt2, Point pt3, double &radius)
{
	double x1 = pt1.x, x2 = pt2.x, x3 = pt3.x;
	double y1 = pt1.y, y2 = pt2.y, y3 = pt3.y;
	double a = x1 - x2;
	double b = y1 - y2;
	double c = x1 - x3;
	double d = y1 - y3;
	double e = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0;
	double f = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0;
	double det = b * c - a * d;
	if (fabs(det) < 1e-5)
	{
		radius = -1;
		return Point(0, 0);
	}
	double x0 = -(d * e - b * f) / det;
	double y0 = -(a * f - c * e) / det;
	radius = hypot(x1 - x0, y1 - y0);
	return Point(x0, y0);
}
vector<double> getKB(Point p1,Point p2)
{
	vector<double> kb;
	double k, b;
	double dx = p1.x - p2.x;
	double dy = p1.y - p2.y;
	if (dx != 0)
		k = (double)(dy / dx);
	else
		k = (double)(dy / (dx + 0.0001));
	b = p1.y - p1.x*k;
	kb.push_back(k);
	kb.push_back(b);
	return kb;
}
int main(int argc, char** argv)
{
	/// Load source image and convert it to gray

	int thresh = 100;
	int max_thresh = 255;
	RNG rng(12345);
	src = imread("1.bmp");
	int w = src.cols;
	int h = src.rows;
	double t = (double)getTickCount();
	cvtColor(src, src_gray, CV_BGR2GRAY);

	Mat src_copy = src.clone();

	//cornerHarris_demo();
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);

	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));
	vector<double> area;
	for (vector<vector<Point> >::iterator it = contours.begin(); it != contours.end(); it++)
	{
		area.push_back(contourArea(*it));
	}
	vector<vector<Point> >poly(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), poly[i], 5, true);
	}

	/// Find the convex hull object for each contour
	vector<vector<Point> >hull(contours.size());
	// Int type hull
	vector<vector<int>> hullsI(contours.size());
	// Convexity defects
	vector<vector<Vec4i>> defects(contours.size());

	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
		// find int type hull
		convexHull(Mat(contours[i]), hullsI[i], false);

		// get convexity defects
		convexityDefects(Mat(contours[i]), hullsI[i], defects[i]);
	}
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	vector<Point> convexPoints;
	int countourId;
	vector<int> convexPointsIdx;
	for (size_t i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		if (area[i] < h * w * 0.9)
		{
			drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			//drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());

			// draw defects
			size_t count = contours[i].size();

			if (count < 50)
				continue;

			for (int j = 0; j < defects[i].size();j++)
			{
					Vec4i& v = defects[i][j];
					int startidx = v[0];
					Point ptStart(contours[i][startidx]); // point of the contour where the defect begins
					int endidx = v[1];
					Point ptEnd(contours[i][endidx]); // point of the contour where the defect ends
					int faridx = v[2];
					Point ptFar(contours[i][faridx]);// the farthest from the convex hull point within the defect
					int depth = v[3] / 256; // distance between the farthest point and the convex hull

					if (depth > 10 && depth < 200)
					{
						circle(drawing, ptFar, 4, Scalar(100, 0, 255), 2);
						convexPoints.push_back(ptFar);
						countourId = i;
						convexPointsIdx.push_back(faridx);
					}
			}
		}
	}
	if (convexPoints.size()==2)
		line(drawing, convexPoints[0], convexPoints[1], CV_RGB(0, 255, 0), 2);
	int start=999, end=-1;
	for (int i = 0; i < convexPointsIdx.size(); i++)
	{
		if (convexPointsIdx[i] < start)
			start = convexPointsIdx[i];
		if (convexPointsIdx[i] > end)
			end = convexPointsIdx[i];
	}
	Point center1, center2;
	//for (int i = start; i <= end; i++)
	{
		Point p(contours[countourId][(start+end)/2]);
		double r1,r2;
		int midPointId1 = (start + end) / 2;
		int mid2 = ((contours[countourId].size() - end) + start - 0) / 2;
		int midPointId2 = ((mid2 + end) < contours[countourId].size()) ? mid2 + end : start - mid2;
		center1 = tcircle(contours[countourId][start], contours[countourId][end], contours[countourId][midPointId1], r1);
		center2 = tcircle(contours[countourId][start], contours[countourId][end], contours[countourId][midPointId2], r1);
		
		
		circle(drawing, center1, 2, Scalar(100, 0, 255), 2);
		circle(drawing, center2, 2, Scalar(100, 0, 255), 2);
	}
	vector<double> convexKB = getKB(contours[countourId][start], contours[countourId][end]);
	vector<vector<Point> > resultPoint;
	vector<double> dist;
	resultPoint.resize(6);
	dist.resize(6);

	int best = 0;
	int best2 = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		best = best2 = 0;
		if (area[i] < h * w * 0.9)
		{
			double minKBError1 = 999999;
			for (int j = 0; j < contours[i].size(); j++)
			{
				if (contours[i][j].y < center1.y)
				{
					double k = getKB(center1,contours[i][j])[0];
					if (abs(k - convexKB[0]) < minKBError1)
					{
						best = j;
						minKBError1 = abs(k - convexKB[0]);
					}
				}
			}
			double minKBError2 = 999999;
			for (int j = 0; j < contours[i].size(); j++)
			{
				if (contours[i][j].y > center1.y)
				{
					double k = getKB(center1,contours[i][j])[0];
					if (abs(k - convexKB[0]) < minKBError2)
					{
						best2 = j;
						minKBError2 = abs(k - convexKB[0]);
					}
				}
			}
			resultPoint[0].push_back(contours[i][best]);
			resultPoint[1].push_back(contours[i][best2]);
		}
	}
	for (int i = 0; i < contours.size(); i++)
	{
		best = best2 = 0;
		if (area[i] < h * w * 0.9)
		{
			double minKBError1 = 999999;
			for (int j = 0; j < contours[i].size(); j++)
			{
				if (contours[i][j].y < center2.y)
				{
					double k = getKB(center2, contours[i][j])[0];
					if (abs(k - convexKB[0]) < minKBError1)
					{
						best = j;
						minKBError1 = abs(k - convexKB[0]);
					}
				}
			}
			double minKBError2 = 999999;
			for (int j = 0; j < contours[i].size(); j++)
			{
				if (contours[i][j].y > center2.y)
				{
					double k = getKB(contours[i][j], center2)[0];
					if (abs(k - convexKB[0]) < minKBError2)
					{
						best2 = j;
						minKBError2 = abs(k - convexKB[0]);
					}
				}
			}
			resultPoint[2].push_back(contours[i][best]);
			resultPoint[3].push_back(contours[i][best2]);
		}
	}
	double verticalK = getKB(center1, center2)[0];// 1.0 / convexKB[0];
	for (int i = 0; i < contours.size(); i++)
	{
		best = best2 = 0;
		if (area[i] < h * w * 0.9)
		{
			double minKBError1 = 999999;
			for (int j = 0; j < contours[i].size(); j++)
			{
				if (contours[i][j].x < center1.x)
				{
					double k = getKB(center1, contours[i][j])[0];
					if (abs(k - verticalK) < minKBError1)
					{
						cout << k << " ";
						best = j;
						minKBError1 = abs(k - verticalK);
					}
				}
			}
			resultPoint[4].push_back(contours[i][best]);
		}
	}
	for (int i = 0; i < contours.size(); i++)
	{
		best = best2 = 0;
		if (area[i] < h * w * 0.9)
		{
			double minKBError1 = 999999;
			for (int j = 0; j < contours[i].size(); j++)
			{
				if (contours[i][j].x > center2.x)
				{
					double k = getKB(center2, contours[i][j])[0];
					if (abs(k - verticalK) < minKBError1)
					{
						best = j;
						minKBError1 = abs(k - verticalK);
					}
				}
			}
			resultPoint[5].push_back(contours[i][best]);
		}
	}

	for (int i = 0; i < 6; i++)
	{
		circle(drawing, resultPoint[i][0], 2, Scalar(100, 0, 255), 2);
		circle(drawing, resultPoint[i][1], 2, Scalar(100, 0, 255), 2);
		line(drawing, resultPoint[i][0], resultPoint[i][1], CV_RGB(0, 255, 0), 2);
	}
	for (int i = 0; i < 6; i++)
	{
		dist[i] = sqrt((resultPoint[i][0].x - resultPoint[i][1].x)*(resultPoint[i][0].x - resultPoint[i][1].x) +
			(resultPoint[i][0].y - resultPoint[i][1].y)*(resultPoint[i][0].y - resultPoint[i][1].y));
		cout << dist[i] << " ";
	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "t: " << t / 1000.0 << " ms" << endl;
	resize(src, src, Size(640, 480));
	resize(drawing, drawing, Size(640, 480));
	/// Show in a window
	namedWindow("Hull demo", CV_WINDOW_AUTOSIZE);
	cv::imshow("Hull demo", drawing);
	cv::waitKey(0);
	return 0;
}
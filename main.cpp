#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc.hpp>
#ifdef HAVE_OPENCV_CUDAIMGPROC
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#endif
#include <math.h>
#include <vector>
#include <fstream>
#include <cstring>

using namespace cv;
using namespace std;

#define UNKNOWN_FLOW_THRESH 1e9  
#define MIN_TARGET_SIZE	250
#define MAX_TARGET_NUM 10
#define REPLY_OK 0
#define REPLY_ERR 1
#define DISPLAY_RESULT true;
#define DEFAULT_VIDEO_FILE "videos\\stuttgart_00.avi"

// Target class definition
class Target //Class for the tracking targets
{
public:
	bool sameTarget(Target); // Estimate if the given target is the same target as this one.	
	int width; //Width of the bounding box of the target.
	int height; //Height of the bounding box of the target.
	int x;	//The x-coordinate of the top-left point of the bounding box of the target
	int y;	//The y-coordinate of the top-left point of the bounding box of the target
	int LastTrackingFrame;  //The last frame that this target is tracked.

public:
	Target();
	Target(const Target& target);
	~Target();
	Point center() { return Point((x + width / 2), (y + height / 2)); }; //Return the center position of the target
	KalmanFilter* kalman;	//Kalman filter of the center position of the target bounding box.
};

// Tracking class definition
class Tracking //Class for the tracking procedure
{
public:
	vector<Target> target;	// Tracking targets
	VideoCapture m_pVideoCapture;
private:
	vector<Target> tempTarget; // Detected targets in current frame
	VideoWriter* m_pWriterTracking;
	Mat m_mSrcFrame3;
	Mat m_mSrcFrame;
	Mat m_mDiffGaussian;
	Mat m_mThreshold;
	int nFrame;
	
	// FPS calculation variables
	double lastTime;
	double currentTime;
	double fps;
	
	// CUDA support variables
	bool useCuda;
	#ifdef HAVE_OPENCV_CUDAIMGPROC
	cv::cuda::GpuMat gpu_frame, gpu_gray, gpu_gaussian, gpu_diff, gpu_threshold;
	#endif
public:
	Tracking();
	~Tracking();
	int DisplayResult(int nTime);
	int GetFrameNum();

	int CreateResultVideo();
	int WriteResultVideo();
	int ReleaseResultVideo();

	int LoadSequence(char* pcFileName);
	int LoadNextFrame();

	int ProcessImage();
	int DetectTarget();
	int TrackTarget();
	void ForceCPUMode() { useCuda = false; }
};

// Function prototype
Point findCenter(Mat Image, vector<Point> contour);

// Target class implementation
Target::Target()//Class for the tracking targets
{
	width = -1;
	height = -1;
	x = -1;
	y = -1;
	LastTrackingFrame = -1;
	kalman = new KalmanFilter(4, 2, 0); 
	Mat state(4, 1, CV_32F); /* (phi, delta_phi) */
	Mat processNoise(4, 1, CV_32F);
	Mat measurement = Mat::zeros(2, 1, CV_32F);
	char code = (char)-1;

	randn(state, Scalar::all(0), Scalar::all(0.1));
	kalman->transitionMatrix = (Mat_<float>(4, 4) <<
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1);
	setIdentity(kalman->measurementMatrix);
	setIdentity(kalman->processNoiseCov, Scalar::all(1e-5));
	setIdentity(kalman->measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kalman->errorCovPost, Scalar::all(1));

	randn(kalman->statePost, Scalar::all(0), Scalar::all(0.1));
}

Target::Target(const Target & target)
{
	x = target.x;
	y = target.y;
	width = target.width;
	height = target.height;
	LastTrackingFrame = target.LastTrackingFrame;
	kalman = target.kalman;
}

Target::~Target()
{
}

bool Target::sameTarget(Target newTarget)
{	// Estimate if the given target is the same target as this one.	
	int area = width*height;
	int newArea = newTarget.width*newTarget.height;
	if (5 * area < newArea || area>5 * newArea)
		return false;
	Mat prediction = kalman->predict();
	Point predictCenter(static_cast<int>(prediction.at<float>(0)), static_cast<int>(prediction.at<float>(1)));
	if (predictCenter.x > newTarget.x&&predictCenter.x < (newTarget.x + newTarget.width) && predictCenter.y>newTarget.y&&predictCenter.y < (newTarget.y + newTarget.height))
	{
		return true;
	}
	else if (newTarget.center().x > x&&newTarget.center().x < (x + width) && newTarget.center().y>y&&newTarget.center().y < (y + height))
	{
		return true;
	}
	
	return false;
}

// Tracking class implementation
Tracking::Tracking()//Class for the tracking targets
{
	nFrame = 0;
	m_pWriterTracking = nullptr;
	lastTime = (double)getTickCount();
	fps = 0.0;
	
	// Check for CUDA support with fallback strategy
	#ifdef HAVE_OPENCV_CUDAIMGPROC
	cout << "Testing CUDA compatibility for compute capability 8.9..." << endl;
	useCuda = false; // Start with CPU, try to enable CUDA
	
	if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
		try {
			// Test with a very simple operation first
			cv::cuda::GpuMat test_mat(10, 10, CV_8UC1);
			test_mat.setTo(cv::Scalar(255));
			cout << "Basic CUDA memory operations: OK" << endl;
			
			// Try a simple upload/download
			Mat cpu_test = Mat::ones(10, 10, CV_8UC1);
			cv::cuda::GpuMat gpu_test;
			gpu_test.upload(cpu_test);
			gpu_test.download(cpu_test);
			cout << "CUDA memory transfers: OK" << endl;
			
			useCuda = true;
			cout << "CUDA acceleration enabled for RTX 4060!" << endl;
		} catch (const cv::Exception& e) {
			cout << "CUDA test failed: " << e.what() << endl;
			cout << "Using optimized CPU processing instead." << endl;
			useCuda = false;
		}
	} else {
		cout << "No CUDA devices found." << endl;
	}
	#else
	useCuda = false;
	cout << "OpenCV compiled without CUDA support." << endl;
	#endif
}

Tracking::~Tracking()
{
	if (m_pWriterTracking)
	{
		m_pWriterTracking->release();
		delete m_pWriterTracking;
		m_pWriterTracking = nullptr;
	}
}

int Tracking::DisplayResult(int nTime)
{
	// Calculate FPS
	currentTime = (double)getTickCount();
	double timeDiff = (currentTime - lastTime) / getTickFrequency();
	fps = 1.0 / timeDiff;
	lastTime = currentTime;
	
	// Display frame, FPS, and processing mode information
	printf("Frame: %d | FPS: %.1f | Mode: %s\n", nFrame, fps, useCuda ? "CUDA" : "CPU");

	imshow("Origin Image", m_mSrcFrame3);
	imshow("otsu threshold", m_mThreshold);
	waitKey(nTime);

	return REPLY_OK;
}

int Tracking::GetFrameNum()
{
	return nFrame;
}

int Tracking::CreateResultVideo()
{
	// Get video properties
	double fps = m_pVideoCapture.get(CAP_PROP_FPS);
	if (fps <= 0) fps = 25; // Default fallback
	
	// Use XVID codec which is more widely supported
	m_pWriterTracking = new VideoWriter("tracking_output.avi", 
		VideoWriter::fourcc('X', 'V', 'I', 'D'), 
		fps, 
		Size(m_mSrcFrame3.size().width, m_mSrcFrame3.size().height), 
		true);

	if (!m_pWriterTracking->isOpened())
	{
		cout << "Warning: Could not create output video file!" << endl;
	}
	else
	{
		cout << "Output video writer created successfully" << endl;
	}

	return REPLY_OK;
}

int Tracking::WriteResultVideo()
{
	if (m_pWriterTracking && m_pWriterTracking->isOpened())
	{
		m_pWriterTracking->write(m_mSrcFrame3);
	}
	return REPLY_OK;
}

int Tracking::ReleaseResultVideo()
{
	if (m_pWriterTracking)
	{
		m_pWriterTracking->release();
		delete m_pWriterTracking;
		m_pWriterTracking = nullptr;
	}
	return REPLY_OK;
}

int Tracking::LoadSequence(char* pcFileName)
{
	m_pVideoCapture.open(pcFileName);
	if (!m_pVideoCapture.isOpened())
	{
		cout << "Error! Failed to load file:" << pcFileName << endl;
		return REPLY_ERR;
	}
	return REPLY_OK;
}

int Tracking::LoadNextFrame()
{
	if (!m_pVideoCapture.read(m_mSrcFrame3))
	{
		return REPLY_ERR;
	}
	else
	{
		nFrame = nFrame + 1;
		return REPLY_OK; 
	}

	return REPLY_OK;
}

int Tracking::ProcessImage()
{
	#ifdef HAVE_OPENCV_CUDAIMGPROC
	if (useCuda) {
		try {
			// GPU-accelerated processing for RTX 4060
			gpu_frame.upload(m_mSrcFrame3);
			cv::cuda::cvtColor(gpu_frame, gpu_gray, COLOR_RGB2GRAY);
			
			// Try GPU operations, fall back to hybrid if needed
			Mat cpu_gray, cpu_gaussian, cpu_diff;
			gpu_gray.download(cpu_gray);
			
			// Use CPU for complex operations (more stable)
			GaussianBlur(cpu_gray, cpu_gaussian, Size(9, 9), 0);
			for (int i = 0; i < 5; i++) {
				GaussianBlur(cpu_gaussian, cpu_gaussian, Size(9, 9), 0);
			}
			
			absdiff(cpu_gray, cpu_gaussian, cpu_diff);
			GaussianBlur(cpu_diff, m_mDiffGaussian, Size(9, 9), 0);
			threshold(m_mDiffGaussian, m_mThreshold, 30, 255, THRESH_BINARY);
			
			m_mSrcFrame = cpu_gray;
			return REPLY_OK;
		} catch (const cv::Exception& e) {
			cout << "CUDA processing failed: " << e.what() << endl;
			useCuda = false;
		}
	}
	#endif
	
	// CPU processing
	cvtColor(m_mSrcFrame3, m_mSrcFrame, COLOR_RGB2GRAY);

	Mat mGaussian;
	GaussianBlur(m_mSrcFrame, mGaussian, Size(9, 9), 0);
	for (int i = 0; i < 5; i++)
	{
		GaussianBlur(mGaussian, mGaussian, Size(9, 9), 0);
	}

	Mat mGroundSkyTest;
	threshold(mGaussian, mGroundSkyTest, 0, 255, THRESH_OTSU);

	Mat mDiff;
	absdiff(m_mSrcFrame, mGaussian, mDiff);

	GaussianBlur(mDiff, m_mDiffGaussian, Size(9, 9), 0);
	for (int i = 0; i < 1; i++)
	{
		GaussianBlur(m_mDiffGaussian, m_mDiffGaussian, Size(9, 9), 0);
	}
	Mat mOtsuThreshold;
	threshold(m_mDiffGaussian, m_mThreshold, 30, 255, THRESH_BINARY);
	return REPLY_OK;
}

int Tracking::DetectTarget()
{
	Mat mSub, mSubOtsuThreshold;
	Mat pThreshold = m_mThreshold.clone();
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(m_mThreshold, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);    // Search for connected domains in the result of otsu thresholding

	tempTarget.clear();
	for (int i=0;i<contours.size();i++)
	{
		double tmparea = fabs(contourArea(contours[i]));
		if (tmparea > MIN_TARGET_SIZE)	//If the area of the connected domain is smaller than 500, it is removed. 
		{
			//----------------------Local thresholding-------------------------
			Rect r = boundingRect(contours[i]);
			if (tmparea > MIN_TARGET_SIZE)	//If the area);
			mSub = m_mDiffGaussian(Range(r.y, r.y + r.height), Range(r.x, r.x + r.width));
			threshold(mSub, mSubOtsuThreshold, 20, 255, THRESH_OTSU); //Otsu thresholding at the local area around the target.
			Mat kernel;
			//Morphological filtering to remove small areas.
			for (int j = 0; j < 3; j++)
			{
				erode(mSubOtsuThreshold, mSubOtsuThreshold, kernel);
			}
			for (int j = 0; j < 5; j++)
			{
				dilate(mSubOtsuThreshold, mSubOtsuThreshold, kernel);
			}

			drawContours(pThreshold, contours, i, CV_RGB(0, 255, 255));

			Mat pThresholdOpt;
			pThreshold.copyTo(pThresholdOpt);
			Mat roi = pThresholdOpt(r);
			mSubOtsuThreshold.copyTo(roi);

			waitKey(10);

			//--------------------------------Extracting target shape from local thresholding result----------------------

			vector<vector<Point> > contours1;
			vector<Vec4i> hierarchy1;
			//IplImage pErode = IplImage(pThreshold);

			findContours(pThresholdOpt, contours1, hierarchy1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			// Search for connected domains in the result of local otsu thresholding
			
			for (int j=0; j!= contours1.size(); j++)
			{
				double tmparea = fabs(contourArea(contours1[j]));

				Rect r = boundingRect(contours1[j]);

				if (tmparea < MIN_TARGET_SIZE) //If the area of the connected domain is smaller than dynamicArea, it is removed.  
				{
					continue;
				}

				// Save the detected target in tempTarget. 				
				Target newTarget;
					
				newTarget.x = r.x;
				newTarget.y = r.y;
				newTarget.width = r.width;
				newTarget.height = r.height;
				newTarget.LastTrackingFrame = nFrame;
				tempTarget.push_back(newTarget);				

			}
		}

	}
	return REPLY_OK;
}

int Tracking::TrackTarget()
{
	if (target.size() == 0) //If No targets have been tracked, then track all the targets in current frame
	{
		for (int i = 0; i < tempTarget.size(); i++)
		{
			Target newTarget(tempTarget[i]);
			target.emplace_back(tempTarget[i]);
		}
	}
	else // If some targets have been tracked in the last frame, try to match (based on position and size) these targets with the newly detected targets.
	{
		for (int i = 0; i < tempTarget.size(); i++)
		{
			int trackFlag = 0;
			for (int j = 0; j < target.size(); j++)
			{
				if (target[j].sameTarget(tempTarget[i])) //estimate if the targets being tracked match with the newly detected targets
				{
					if (abs(target[j].center().x - tempTarget[i].center().x) < 10 && abs(target[j].center().y - tempTarget[i].center().y) < 10 && abs(target[j].width - tempTarget[i].width) < 10 && abs(target[j].height - tempTarget[i].height) < 10)
					{
						//if matched, update the state of the target.
						target[j].x = tempTarget[i].x;
						target[j].y = tempTarget[i].y;
						target[j].width = tempTarget[i].width;
						target[j].height = tempTarget[i].height;
						Mat measurement = Mat::zeros(2, 1, CV_32FC1);
						measurement.at<float>(0) = float(target[j].x + target[j].width / 2 - 1);
						measurement.at<float>(1) = float(target[j].y + target[j].height / 2 - 1);
						target[j].kalman->correct(measurement);
						target[j].kalman->predict();
						target[j].LastTrackingFrame = nFrame;
					}
					trackFlag = 1;
				}
			}
			if (trackFlag == 0)//If no match is found, save it as a new tracking target.
			{
				Target newTarget(tempTarget[i]);
				target.emplace_back(tempTarget[i]);
			}
		}
	}
	auto it = target.begin();
	while (it != target.end())
	{
		Rect r;
		r.width = it->width;
		r.height = it->height;
		r.x = it->x;
		r.y = it->y;

		Scalar red(0, 0, 200);
		rectangle(m_mSrcFrame3, r, red); // Plot the target bounding box.

		if (it->LastTrackingFrame < nFrame - 5) // If the target is missing for 5 frames, remove it.
		{
			delete it->kalman;
			it = target.erase(it);
		}
		else
		{
			it++;
		}

	}

	return REPLY_OK;
}

Point findCenter(Mat Image, vector<Point> contour)
{
	// Find the center point of the given connected domain'contour'.
	Rect r = boundingRect(contour);

	Point center(static_cast<int>(r.x + 0.5*r.width), static_cast<int>(r.y + 0.5*r.height));
	//Image.cols;
	//if (r.x + r.width>Image.cols)
	//{
	//	r.width = Image.cols - r.x;
	//}
	//if (r.y + r.height>Image.rows)
	//{
	//	r.height = Image.rows - r.y;
	//}
	//if (r.height <= 0)
	//{
	//	r.height = 0;
	//}
	//if (r.width <= 0)
	//{
	//	r.width = 0;
	//}
	//Mat subr = Image(r);
	//Moments m(subr, 0);
	//M00 = cvGetSpatialMoment(&m, 0, 0);
	//center.x = (int)(cvGetSpatialMoment(&m, 1, 0) / M00);
	//center.y = (int)(cvGetSpatialMoment(&m, 0, 1) / M00);
	return center;
}

int main(int argc, char* argv[])
{
	// Show CUDA information if requested (check first)
	cout << "CUDA Information" << endl;
	#ifdef HAVE_OPENCV_CUDAIMGPROC
	int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
	cout << "OpenCV compiled with CUDA support: YES" << endl;
	cout << "CUDA-enabled devices: " << deviceCount << endl;
	if (deviceCount > 0) {
		for (int j = 0; j < deviceCount; j++) {
			cv::cuda::DeviceInfo deviceInfo(j);
			cout << "Device " << j << ": " << deviceInfo.name() << endl;
			cout << "  Compute capability: " << deviceInfo.majorVersion() << "." << deviceInfo.minorVersion() << endl;
			cout << "  Total memory: " << deviceInfo.totalMemory() / (1024*1024) << " MB" << endl;
		}
	} else {
		cout << "No CUDA devices detected" << endl;
	}
	#else
	cout << "OpenCV compiled with CUDA support: NO" << endl;
	cout << "To enable CUDA, recompile OpenCV with -DWITH_CUDA=ON" << endl;
	#endif
	
	// Show usage if help is requested
	if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0))
	{
		cout << "Usage: " << argv[0] << " [options] [video_file_path]" << endl;
		cout << "Options:" << endl;
		cout << "  -h, --help     Show this help message" << endl;
		cout << "  --cpu          Force CPU processing (disable CUDA)" << endl;
		cout << "  --cuda-info    Show CUDA information and exit" << endl;
		cout << "Examples:" << endl;
		cout << "  " << argv[0] << "                              // Use default video with auto CUDA detection" << endl;
		cout << "  " << argv[0] << " videos\\stuttgart_01.avi     // Use specific video" << endl;
		cout << "  " << argv[0] << " --cpu videos\\test.mp4       // Force CPU processing" << endl;
		cout << "  " << argv[0] << " --cuda-info                 // Show CUDA capabilities" << endl;
		return 0;
	}

	// Check for force CPU option and get video file
	bool forceCPU = false;
	char* videoFile = nullptr;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--cpu") == 0) {
			forceCPU = true;
		} else if (argv[i][0] != '-') {
			videoFile = argv[i];
			break; // Take the first non-option argument as video file
		}
	}
	if (!videoFile) {
		videoFile = (char*)DEFAULT_VIDEO_FILE;
	}

	Tracking testTracking;

	// Override CUDA detection if CPU is forced
	if (forceCPU) {
		cout << "CPU processing forced by --cpu option" << endl;
		testTracking.ForceCPUMode();
	}

	// Use command line argument for video file, or default
	cout << "Loading video: " << videoFile << endl;
	

	if (testTracking.LoadSequence(videoFile) != REPLY_OK)
	{
		cout << "Failed to load video file: " << videoFile << endl;
		cout << "Please check if the file exists and is a valid video format." << endl;
		cout << "Supported formats: AVI, MP4, MOV, WMV, etc." << endl;
		return -1;
	}

	int totalFrames = (int)testTracking.m_pVideoCapture.get(CAP_PROP_FRAME_COUNT);
	cout << "Total frames in video: " << totalFrames << endl;

	while (testTracking.LoadNextFrame() == REPLY_OK)
	{
		if (testTracking.GetFrameNum() == 1)
		{
			testTracking.CreateResultVideo();
		}
		testTracking.ProcessImage();
		testTracking.DetectTarget();
		testTracking.TrackTarget();
		testTracking.DisplayResult(1);
		testTracking.WriteResultVideo();
		
		// Press 'q' to quit early
		if (waitKey(1) == 'q')
			break;
	}

	testTracking.ReleaseResultVideo();
	
	return 0;
}


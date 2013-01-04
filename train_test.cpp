#include<iostream>
#include<vector>
#include<string>
#include<sstream>
#include"opencv/cv.h"
//#include"opencv2/core/core.hpp"
//#include"opencv/cxcore.h"
#include"opencv2/opencv.hpp"
//#include"opencv2/highgui/highgui.hpp"
//#include"opencv2/core/core.hpp"
using namespace std;

string filename( "D:\\港口\\test\\test.tif" );
char* traindata_file = "D:\\港口\\args\\traindata.xml";
char* testdata_file = "D:\\港口\\args\\testdata.xml";
char* model_file = "D:\\港口\\args\\model.xml";
char* result_file = "D:\\港口\\result\\test_result.xml";

string target_file( "D:\\港口\\result\\test_result.tif" );


int output_target( cv::Mat _image, std::vector<int> count_points, int rn, int cn, int blk_len, int _n_blks, cv::Mat _results);

int main()
{
	/** read train data and do the training**/
	cv::Mat _response, _feature_train, _feature_test;
	cv::FileStorage fs;	
	fs.open( traindata_file, cv::FileStorage::READ );
	fs["response"] >> _response;
	fs["feature"] >> _feature_train;
	fs.release();

	/** read test data and do the predicting **/
	cv::FileStorage fs1;
	int _n_blks, rn, cn, blk_len;
	cv::Mat _count_points;
	fs1.open( testdata_file, cv::FileStorage::READ );
	fs1["n_blks"] >> _n_blks;
	fs1["feature"] >> _feature_test;
	fs1["count_point"] >> _count_points;
	fs1["rn"] >> rn;
	fs1["cn"] >> cn;
	fs1["blk_len"] >> blk_len;
	std::vector<int> count_points;
	for( int i = 0; i < _count_points.rows; i++ )
		count_points.push_back( _count_points.at<int>( i, 0 ) );
	cv::Mat _results( _n_blks, 1, CV_32FC1 );
	fs1.release();

	/*svm-train*/
	CvSVM svm;
	CvSVMParams _params;
	_params.svm_type = CvSVM::C_SVC;
	_params.kernel_type = CvSVM::RBF; // which is better?
	_params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER, 100, 1e-6 );  
	svm.train_auto( _feature_train, _response, cv::Mat() , cv::Mat(), _params );
	
	svm.predict( _feature_test, _results );

	/** output the target **/
	cv::Mat __image = cv::imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
	output_target( __image, count_points, rn, cn, blk_len, _n_blks, _results );
	svm.save( model_file, "my_svm");
	cv::FileStorage fs2( result_file, cv::FileStorage::WRITE );
	fs2 << "result" <<  _results;
	fs2.release();
	return 0;
}



int output_target( cv::Mat _image, std::vector<int> count_points, int rn, int cn, int blk_len, int _n_blks, cv::Mat _results)
{
	int i, j;
	std::vector<int>::const_iterator iter = count_points.begin();
	_results = _results.reshape( 1, _n_blks );
	cv::Mat image( _image.rows, _image.cols, _image.type() );
	image = 0;
	for( i = 0; i < rn; i++ )
		for(j = 0; j < cn; j++ )
		{
			if( *iter != 0 && _results.at<float>( i, j ) == 1 )
			{
				image.rowRange( blk_len * i, blk_len * ( i + 1 ) ).colRange( blk_len * j, blk_len * ( j + 1 ) ) = _image.rowRange( blk_len * i, blk_len * ( i + 1 ) ).colRange( blk_len * j, blk_len * ( j + 1 ) ); 
			}
			iter++;
		}
	cv::imwrite( target_file, image );
	return 0;
}
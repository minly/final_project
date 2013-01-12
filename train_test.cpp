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

string filename( "D:\\港口\\test\\test3_Sandiego18.tif" );
char* traindata_file = "D:\\港口\\args\\traindata.xml";
char* testdata_file = "D:\\港口\\args\\testdata_San18.xml";
char* model_file = "D:\\港口\\args\\model.xml";
string result_file = "D:\\港口\\result\\test3_Sandiego18_result.xml";

string target_file( "D:\\港口\\result\\test3_Sandiego18" );


int output_target( cv::Mat _image, std::vector<int> count_points, int rn, int cn, int blk_w, int blk_h, int _n_blks, cv::Mat _results, string _target_file);
cv::Mat nomalize( cv::Mat& mat );

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
	int _n_blks, rn, cn, blk_h, blk_w;
	cv::Mat _count_points;
	fs1.open( testdata_file, cv::FileStorage::READ );
	fs1["n_blks"] >> _n_blks;
	fs1["feature"] >> _feature_test;
	fs1["count_point"] >> _count_points;
	fs1["rn"] >> rn;
	fs1["cn"] >> cn;
	fs1["blk_h"] >> blk_h;
	fs1["blk_w"] >> blk_w;
	fs1.release();
	std::vector<int> count_points;
	for( int i = 0; i < _count_points.rows; i++ )
		count_points.push_back( _count_points.at<int>( i, 0 ) );	
	cv::Mat _results( _n_blks, 1, CV_32FC1 );
	_results = cv::Scalar( 0 );
	_feature_train = nomalize( _feature_train );
	_feature_test = nomalize( _feature_test );


	string _target_file;
	stringstream sstr;	
	cv::FileStorage fs2( result_file, cv::FileStorage::WRITE );
	/*svm-train*/
	CvSVM svm;
	CvSVMParams _params;
	_params.svm_type = CvSVM::C_SVC;
	_params.kernel_type = CvSVM::RBF; // which is better?	
	_params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER, 100, 1e-6 );
	cv::Mat __image = cv::imread( filename, 1);
	for( int i = 1; i < 10; i++ )
	{
		_params.gamma = 1 / (float)( i * i * 2);
		svm.train( _feature_train, _response, cv::Mat() , cv::Mat(), _params );
		svm.predict( _feature_test, _results );	
		svm.save( model_file, "my_svm" ) ;
		sstr << i;
		_target_file =  target_file + "_" + sstr.str() + ".tif";
		output_target( __image, count_points, rn, cn, blk_w, blk_h, _n_blks, _results, _target_file );		
		sstr.str("");
		_target_file = "";
		fs2 << "result" <<  _results;
	} 
	
	fs2.release();
	return 0;
}



int output_target( cv::Mat _image, std::vector<int> count_points, int rn, int cn, int blk_w, int blk_h, int _n_blks, cv::Mat _results, string _target_file )
{
	int i, j;
	std::vector<int>::const_iterator iter = count_points.begin();
	_results = _results.reshape( 1, rn );
	cv::Mat image( _image.rows, _image.cols, _image.type() );
	image = 0;	
	for( i = 0; i < rn; i++ )
		for(j = 0; j < cn; j++ )
		{
			if( *iter != 0 && _results.at<float>( i, j ) == 1.0 )
			{
				_image.rowRange( blk_h * i, blk_h * ( i + 1 ) ).colRange( blk_w * j, blk_w * ( j + 1 ) ).copyTo( image.rowRange( blk_h * i, blk_h * ( i + 1 ) ).colRange( blk_w * j, blk_w * ( j + 1 ) ) ); 
			}
			iter++;
		}
	cv::imwrite( _target_file, image );
	return 0;
}


cv::Mat nomalize( cv::Mat& mat )
{
	int i, j;
	int rows = mat.rows;
	int cols = mat.cols;
	cv::Scalar s = cv::sum( mat );
	float sum = s[0];
	for( i = 0; i < rows; i++ )
		for( j = 0; j < cols; j++ )
			mat.at<float> ( i, j ) /= sum;
	return mat;
}
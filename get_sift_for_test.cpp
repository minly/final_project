/*
extract feature discriptors of the test image given and then combine them
@author Minly
@version 
*/

/***********************************Macro********************************************/
//#define LINUX
/************************************************************************************/
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<sstream>
#include"opencv/cv.h"
#include"opencv/cxcore.h"
#include"opencv2/opencv.hpp"
#include"Image.h"
#include"ImageFeature.h"
using namespace std;


#define DES_DIMENSION 128
#define N_CENTERS 500
/****************************** Globals *********************************************/
#ifdef LINUX
string filename( "../test.tif" );
char* centers_file = "./args/center.xml";
char* model_file = "./args/model.xml";
string target_file( "./result/test_result.tif" );
#else
string filename( "D:\\港口\\test\\test.tif" );
char* centers_file = "D:\\港口\\args\\center.xml";
char* model_file = "D:\\港口\\args\\model.xml";
string target_file( "D:\\港口\\result\\test_result.tif" );
#endif

int total_points = 0;
int blk_len = 400;


/************************************************************************************/
#ifndef LINUX
void printMat(const cv::Mat mat, char* filename, char* way );
#endif
void bof( cv::Mat _descriptors, cv::Mat& feature, std::vector<int> count_points, int _n_blks, cv::Mat center );
int output_target( cv::Mat _image, std::vector<int> count_points, int rn, int cn, int blk_len, int _n_blks, cv::Mat _results);


int main()
{
	
	int i, j;		
	vector<string>::const_iterator iterator;
	Image<uchar> image;
	cv::Mat mask_img;	
	Image<float> imsift;
	int width, height, n;
	int c = 0, r = 0, num = 0, _n_blks = 0, offset = 0; 
	std::vector<int> count_points;
	cv::Mat _descriptor;
	std::vector<float> descriptors;
	image.imread( filename.c_str() );//read image and convert to grayscale
	//image = cv::imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
	
	int h = image.height();
	int w = image.width();
	
	int rn = h / blk_len;
	int cn = w / blk_len;
	

	int chn = image.nchannels();
	Image<uchar> _image( blk_len, blk_len, chn );
	
	int npixels = blk_len * blk_len * chn;
	int offset_r = 0, offset_c = 0;
	n = 0;
	int n_a_line = blk_len * chn;
	/*extract surf for each block*/
	cout << "extracting feature..." <<endl;
	for( r = 0; r < rn; r++)
	{			
		for( c = 0; c < cn; c++)
		{	
			offset_r = ( blk_len * r ) * w * chn;
			offset_c = ( blk_len * c ) * chn ;
			//get the block image
			for( j = 0; j < npixels; j++ )
			{					
				_image.pData[ j ] = image.pData[ offset_r + offset_c + ( k++ ) ];
				_image.pData[ ++j ] = image.pData[ offset_r + offset_c + ( k++ ) ];
				_image.pData[ ++j ] = image.pData[ offset_r + offset_c + ( k++ ) ];
				n += 3 ;
				if( n == n_a_line )
				{
					offset_r += w * chn;
					n = 0;
					k = 0;
				}
			}
			//_image.imwrite("D://_image.tif" );
			
				
			/*set the mask*/
			//mask_img = image.rowRange( blk_len * r, blk_len * ( r + 1 ) ).colRange( blk_len * c, blk_len * ( c + 1 ) );
			//_image.
			//extract sift descriptor
			//sift( _image, *mask, keypoints, descriptor, false );//gray_image convert to 8-bit?
			//surf( _image, cv::noArray(), keypoints, _descriptor, true);//surf automatically merge the keypoints, if the vector is not reallocated
			ImageFeature::imSIFT<uchar, float>( _image, imsift, 3, 1, true );
			//printMat(_descriptor, "D:/_descriptor", "w" );
			width = imsift.width();
			height = imsift.height();	   				
			num = height * width;

			count_points.push_back( num );//store num of points for each block
			if( num != 0)
			{
				for( i = 0; i < height; i++)
					for( j = 0; j < width; j++ )
					{
						offset = (i*width+j)*DES_DIMENSION;
						for( k = 0; k < DES_DIMENSION; k++ )
							descriptors.push_back( imsift.pData[offset+k]);
					}
				total_points += num;
				_n_blks++;
			}
			
		}
	}
	_image.~Image();
	std::cout << rn*cn << endl;
	
	cv::Mat _descriptors( descriptors, false );
	_descriptors = _descriptors.reshape( 1, total_points );

	descriptors.clear();
	
	/*kmeans*/
	cout << "kmeans..." <<endl;
	cv::Mat centers, label;
	cv::Mat _feature;
	cv::FileStorage fs;
	fs.open( centers_file, cv::FileStorage::READ );
	fs["centers"] >> centers;
	bof( _descriptors, _feature, count_points, _n_blks, centers );//final feature
	fs.release();

	_descriptors.~Mat();

//////////////////////////////remove on linux///////////////////////////////////////////////////////////////////	
	cv::FileStorage fs2(  "D:\\港口\\args\\test_descriptors.xml", cv::FileStorage::WRITE );
	fs2 << "test_descriptors" << _descriptors;
	fs2.release();
	/** save test data which will be used in the train_test**/
	char* testdata_file = "D:\\港口\\args\\testdata.xml";
	cv::Mat _count_points( count_points, false );
	cv::FileStorage fs1( testdata_file, cv::FileStorage::WRITE );
	fs1 << "feature" << _feature;
	fs1 << "count_point" << _count_points;//on the premise of densesift	
	fs1 << "n_blks" << _n_blks;
	fs1 << "rn" << rn;
	fs1 << "cn" << cn;
	fs1 <<"blk_len" << blk_len;

	fs1.release();

/////////////////////////////remove on linux//////////////////////////////////////////////////////////////////////

	/*svm-test*/
	cout << "predicting..." <<endl;
	CvSVM svm = CvSVM();   
	cv::Mat __image;
	cv::Mat _results( _n_blks, 1, CV_32FC1 );
	svm.load( model_file, "my_svm" );
	svm.predict( _feature, _results );
	__image = cv::imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
	output_target( __image, count_points, rn, cn, blk_len, _n_blks, _results );
	cout << "Done!" <<endl;
	return 0;
}



int output_target( cv::Mat _image, std::vector<int> count_points, int rn, int cn, int blk_len, int _n_blks, cv::Mat _results)
{
	int i, j;
	std::vector<int>::const_iterator iter = count_points.begin();
	_results = _results.reshape( 1, _n_blks );
	cv::Mat image( _image.rows, _image.cols, _image.type() );
	image = cv::Scalar( 0 );
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

void bof( cv::Mat _descriptors, cv::Mat& feature, std::vector<int> count_points, int _n_blks, cv::Mat centers )//feature: output
{
	int n_blks = count_points.size();
	int n_centers = centers.rows;
	int count = 0, i, j, k, t = 0, num, center_idx;
	std::vector<int>::const_iterator iter = count_points.begin();
	cv::Mat desc, dist, sorted_idx;	
	feature = cv::Mat::zeros(_n_blks, n_centers, CV_32FC1 );//initialize
	for( i = 0; i < n_blks; i++ )
	{
		num = *iter;
		if( num != 0) 
		{
			//desc.create( num, DES_DIMENSION, _descriptors.type() );
			dist.create( num, n_centers, CV_32FC1 );
			desc = _descriptors.rowRange( count, count + num );//descriptors for the block
			for( j = 0; j < num; j++ )//distance for each block to each center
			{
				for( k = 0; k < n_centers; k++ )
				{
					dist.at<float>(j, k) = cv::norm( desc.row( j ) - centers.row( k ), cv::NORM_L2 );
				}
			}
			cv::exp( (-0.001) * dist, dist );//from now on, mat dist is hold the similarity not the distance

			cv::sortIdx( dist, sorted_idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING );//sort each row descendently(indices are sorted actually)
			for( j =0; j < num; j++ )
			{
				for( k = 0; k < 4; k++ )
				{
					center_idx = sorted_idx.at<int>( j, k );
					feature.at<float>( t, center_idx ) +=  dist.at<float>( j, center_idx ) / std::pow( 2.0, k );
				}
			}
			t++;
			count += num;
		}
		iter++;

	}
}


#ifndef LINUX
void printMat(const cv::Mat mat, char* filename, char* way)
{
	int rows = mat.rows;
	int cols = mat.cols;
	int i, j;
	FILE * fp;
	fp = fopen(filename, way);
	if(!fp) exit(0);
	//std::ofstream output( filename );
	for( i =0; i< rows; i++ )
	{
		for(j =0; j<cols; j++)
			fprintf( fp,"%f ",*(mat.data + mat.step[0]*i + mat.step[1]*j));
			//output << *(mat.data + mat.step[0]*i + mat.step[1]*j);
		fprintf(fp, "\n" );
		//output << endl;
	}
		
	fclose(fp);
	//output.close();
}
#endif



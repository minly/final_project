/*
extract feature discriptors of all the images under the dir and then combine them
@author Minly
@version 
*/

/************************************************************************************/
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<sstream>
#include"opencv/cv.h"
#include"opencv/cxcore.h"
#include"opencv2/opencv.hpp"
//#include"opencv2/highgui/highgui.hpp"
//#include"opencv2/core/core.hpp"
#include"opencv2/features2d/features2d.hpp"
#include"opencv2/nonfree/features2d.hpp"
#include"opencv2/nonfree/nonfree.hpp"
using namespace std;


#define DES_DIMENSION 128
#define N_CENTERS 500
/****************************** Globals *********************************************/
string filename( "D:\\港口\\test\\test1.tif" );
int total_points = 0;
int blk_len = 30;

char* centers_file = "D:\\airport training data\\train\\center.xml";
char* model_file = "D:\\airport training data\\train\\model.xml";
string target_file( "D:\\港口\\test\\test1_result.tif" );
/************************************************************************************/

void printMat(const cv::Mat mat, char* filename, char* way );
void bof( cv::Mat _descriptors, cv::Mat& feature, std::vector<int> count_points, int _n_blks, cv::Mat center );
int output_target( cv::Mat _image, std::vector<int> count_points, int rn, int cn, int blk_len, int _n_blks, cv::Mat _results);


int main()
{
	
	int i, j;		
	vector<string>::const_iterator iterator;
	cv::Mat _image, mask;
	//cv::SIFT sift(10, 5 ); 
	cv::SURF surf;
	int width, height, cn, rn, k, m, n;
	int c = 0, r = 0, num = 0, count = 0, _n_blks = 0; 
	std::vector<cv::KeyPoint> keypoints;
	std::vector<int> count_points;
	cv::Mat _descriptor;
	std::vector<float> descriptors;
	_image = cv::imread( filename, CV_LOAD_IMAGE_GRAYSCALE );//read image and convert to grayscale
	//_image = cv::imread( "D:\\airport training data\\negative\\negative8.tif", CV_LOAD_IMAGE_GRAYSCALE );
	//cv::cvtColor( _image, gray_image, CV_BGR2GRAY );//BGR? or RGB?
	width = _image.cols;
	height = _image.rows;
	mask.create( height, width, CV_8UC1 );
	cn = width / blk_len;
	rn = height / blk_len;
	
	/*extract surf for each block*/
	for( r = 0; r < rn; r++)
	{			
		for( c = 0; c < cn; c++)
		{			
			/*set the mask*/
			mask = cv::Scalar_< int>( 0 );
			mask.rowRange( blk_len * r, blk_len * ( r + 1 ) ).colRange( blk_len * c, blk_len * ( c + 1 ) ) = 1;
			//printMat( *mask, "D:/mask", "w" );
			/* set the keypoints : for dense sift the mask will not be used, so here to set keypoints for each block every loop*/
			for( i = 0; i < height; i++ )
				for( j = 0; j < width; j++ )
				{
					if( mask.at< uchar >( i, j ) ==  1 )
						keypoints.push_back( cv::KeyPoint( i, j, 10 ) );
				}

			//extract sift descriptor
			//sift( _image, *mask, keypoints, descriptor, false );//gray_image convert to 8-bit?
			surf( _image, cv::noArray(), keypoints, _descriptor, true);//surf automatically merge the keypoints, if the vector is not reallocated
			//printMat(_descriptor, "D:/_descriptor", "w" );
			num = _descriptor.rows;

			count_points.push_back( num );//store num of points for each block
			if( num != 0)
			{
				for( m = 0; m < num; m++)
						for( n = 0; n < DES_DIMENSION; n++ )
							descriptors.push_back( _descriptor.at<float>( m, n ) );
				total_points += num;
				_n_blks++;
			}
			keypoints.clear();//surf would not deallocate the keypoits vector
			
		}
	}
	
	std::cout << rn*cn << endl;
	
	cv::Mat _descriptors( descriptors, false );
	_descriptors = _descriptors.reshape( 1, total_points );

	


	/*kmeans*/
	cv::Mat centers, label;
	cv::Mat _feature;
	cv::FileStorage fs;
	fs.open( centers_file, cv::FileStorage::READ );
	fs["centers"] >> centers;
	bof( _descriptors, _feature, count_points, _n_blks, centers );//final feature	
	printMat( _feature, "D:/feature", "w");


	/*svm-train*/
	/*CvSVM svm;	
	cv::Mat _results( _n_blks, 1, CV_32FC1 );
	svm.load( model_file );
	svm.predict( _feature, _results );
	output_target( _image, count_points, rn, cn, blk_len, _n_blks, _results );*/
	
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
					feature.at<float>( t, center_idx ) +=  dist.at<float>( j, center_idx ) / float(2^( k -1 ));
				}
			}
			t++;
			count += num;
		}
		iter++;

	}
}

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




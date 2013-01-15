/*
extract feature discriptors of all the images under the dir and then combine them
@author Minly
@version 
*/


/************************************************************************************/

#include<iostream>
#include<vector>
#include<string>
#include<sstream>
#include"opencv/cv.h"
#include"opencv2/opencv.hpp"
#include"opencv2/nonfree/features2d.hpp"
//#include "Image.h"
//#include"ImageFeature.h"
using namespace std;


#define DES_DIMENSION 128
#define N_CENTERS 500
/****************************** Globals *********************************************/
string dir( "D:\\港口\\train" );
int np_files = 31;//positive samples
int nn_files = 31;//negative samples

int total_points = 0;

char* traindata_file = "D:\\港口\\args_nondense\\traindata.xml";
char* centers_file = "D:\\港口\\args_nondense\\center.xml";

/**************************Declaration***************************************************/

void printMat(const cv::Mat mat, char* filename, char* way );
void bof( cv::Mat _descriptors, cv::Mat& feature, std::vector<int> count_points, int nz_imgs, cv::Mat center );


int main()
{
	vector<string> flist;
	string filename;
	stringstream  sstr;
	int i, j;
	/*get all the positive sample filenames in the folder*/
	for( i = 1; i <= np_files; i++ )
	{
		sstr << i;
		filename = dir + "\\target" + sstr.str() + ".tif";
		sstr.str("");
		flist.push_back( filename );
	}
	
	/*get all the negative sample filenames in the folder*/
	for( i = 1; i <= nn_files; i++ )
	{
		sstr << i;
		filename = dir + "\\negative" + sstr.str() + ".tif";
		sstr.str("");
		flist.push_back( filename );
	}
		
	vector<string>::const_iterator iterator;
	cv::Mat _image;
	//Image<uchar> _image;
	//Image<float> imsift;
	cv::SIFT sift( 0, 4, 0.04, 10.0, 0.5 ); 
	//cv::SURF surf;
	std::vector<cv::KeyPoint> keypoints;
	int width, height, k, offset = 0;
	int num = 0, count = 0;
	int n_imgs = 0;//total num of images
	int nz_imgs = 0; //num of images have keypoints
	int np_imgs = 0;//num of positive images that have keypoints
	std::vector<int> count_points;
	cv::Mat _descriptor;
	std::vector<float> descriptors;
	float* tmp;
	for( iterator = flist.begin(); iterator != flist.end(); iterator++ )
	{
		//_image.imread( (*iterator).c_str() );//read image and convert to grayscale
		_image = cv::imread( (*iterator).c_str(), CV_LOAD_IMAGE_GRAYSCALE );
		//cv::cvtColor( _image, gray_image, CV_BGR2GRAY );//BGR? or RGB?
		
		/*** extract dense sift image ***/
		//ImageFeature::imSIFT<uchar, float>( _image, imsift, 3, 1, true ); 
		sift( _image, cv::noArray(), keypoints, _descriptor, false );//gray_image convert to 8-bit?
		//surf( _image, cv::noArray(), keypoints, _descriptor, true);//surf automatically merge the keypoints, if the vector is not reallocated
		/*width = imsift.width();
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
			nz_imgs++;
		}*/
		//int chn = _descriptor.channels();
		num =  _descriptor.rows;
		count_points.push_back( num );
		/*put the descriptors data into vector*/
		if( num != 0 )
		{
			for( i = 0; i < num; i++ )
				for( j = 0; j < DES_DIMENSION; j++ )
				{
					 tmp = (float*)(_descriptor.data + _descriptor.step[0] * i + _descriptor.step[1] * j);
					 descriptors.push_back( *tmp );
				}
			
			total_points += num;
			nz_imgs++;
		}
		keypoints.clear();
		n_imgs++;
		
		if( n_imgs == np_files ) np_imgs = nz_imgs;//record the num of the positive blocks
	}
	cv::Mat _descriptors( descriptors, false );
	_descriptors = _descriptors.reshape( 1, total_points );
	std::cout << "total points: " << total_points << endl;
	
	/*kmeans*/
	cv::Mat centers, label;
	cv::Mat _feature;
	cv::FileStorage fs( centers_file, cv::FileStorage::WRITE );
	cv::kmeans( _descriptors, N_CENTERS, label, cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0 ), 1, cv::KMEANS_PP_CENTERS, centers );
	fs << "centers" << centers;
	fs.release();

	bof( _descriptors, _feature, count_points, nz_imgs, centers );//bagged feature	

	/** save train data for the procceding train_test**/
	cv::FileStorage fs1( traindata_file, cv::FileStorage::WRITE );
	fs1 << "train_descriptors" << _descriptors;
	_descriptors.~Mat();
	descriptors.clear();	
	cv::Mat _response( nz_imgs, 1, CV_32FC1 );
	_response.rowRange( 0, np_imgs ) = 1;
	_response.rowRange( np_imgs, nz_imgs ) = -1;	
	fs1 << "response" << _response;
	fs1 << "feature" << _feature;
	fs1.release();
	
	return 0;
}

void bof( cv::Mat _descriptors, cv::Mat& feature, std::vector<int> count_points, int nz_imgs, cv::Mat centers )//feature: output
{
	int n_imgs = count_points.size();//num of images
	int n_centers = centers.rows;
	int count = 0, i, j, k, t = 0, num, center_idx;
	std::vector<int>::const_iterator iter = count_points.begin();
	cv::Mat desc, dist, sorted_idx;	
	feature = cv::Mat::zeros( nz_imgs, n_centers, CV_32FC1 );//initialize
	for( i = 0; i < n_imgs; i++ )
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
					dist.at<float>(j, k) = (float)cv::norm( desc.row( j ) - centers.row( k ), cv::NORM_L2 );
				}
			}

			
			cv::exp( (-0.001) * dist, dist );//from now on, mat dist is hold the similarity not the distance
			cv::sortIdx( dist, sorted_idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING );//sort each row descendently(indices are sorted actually)
			float f = 0;
			for( j = 0; j < num; j++ )
			{
				for( k = 0; k < 4; k++ )
				{
					center_idx = sorted_idx.at<int>( j, k );
					feature.at<float>( t, center_idx ) +=  dist.at<float>( j, center_idx ) / std::pow( 2.0, k );
					f = feature.at<float>( t, 4 );
					
				}
			}
			t++;
			count += num;
		}
		iter++;

	}
}

/*
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
*/



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
string dir( "D:\\¸Û¿Ú\\train" );
int np_files = 31;//positive samples
int nn_files = 31;//negative samples

int total_points = 0;

char* centers_file = "D:\\airport training data\\train\\center.xml";
char* model_file = "D:\\airport training data\\train\\model.xml";
/************************************************************************************/

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
	cv::SIFT sift; 
	//cv::SURF surf;
	int width, height, k, m, n;
	int num = 0, count = 0;
	int n_imgs = 0;//total num of images
	int nz_imgs = 0; //num of images have descriptors
	int np_imgs = 0;//num of positive images that have descriptors
	std::vector<cv::KeyPoint> keypoints;
	std::vector<int> count_points;
	cv::Mat _descriptor;
	std::vector<float> descriptors;
	for( iterator = flist.begin(); iterator != flist.end(); iterator++ )
	{
		_image = cv::imread( *iterator, CV_LOAD_IMAGE_GRAYSCALE );//read image and convert to grayscale
		//_image = cv::imread( "D:\\airport training data\\negative\\negative8.tif", CV_LOAD_IMAGE_GRAYSCALE );
		//cv::cvtColor( _image, gray_image, CV_BGR2GRAY );//BGR? or RGB?
		width = _image.cols;
		height = _image.rows;
	   				
		
		/*** set the keypoints : for dense sift the mask will not be used, so here to set keypoints for each block every loop ***/
		for( i = 0; i < height; i++ )
			for( j = 0; j < width; j++ )
			{
				keypoints.push_back( cv::KeyPoint( i, j, 0 ) );
			}

		/*** extract sift descriptor ***/
		sift( _image, cv::noArray(), keypoints, _descriptor, true );//gray_image convert to 8-bit?
		//surf( _image, cv::noArray(), keypoints, _descriptor, true);//surf automatically merge the keypoints, if the vector is not reallocated
		//printMat(_descriptor, "D:/_descriptor", "w" );
		num = _descriptor.rows;
		count_points.push_back( num );//store num of points for each block
		if( num != 0)
		{
			for( m = 0; m < num; m++)
				for( n = 0; n < DES_DIMENSION; n++ )
					descriptors.push_back( _descriptor.at<float>( m, n ) );
			total_points += num;
			nz_imgs++;
		}
		keypoints.clear();//surf would not deallocate the keypoits vector		
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
	printMat( _descriptors, "D://_descriptors", "w");
	fs << "centers" << centers;
	fs.release();

	//printMat( centers, "D:/centers", "w");
	bof( _descriptors, _feature, count_points, nz_imgs, centers );//final feature	
	//printMat( _feature, "D:/feature", "w");


	/*svm-train*/
	CvSVM svm;
	CvSVMParams _params;
	_params.svm_type = CvSVM::C_SVC;
	_params.kernel_type = CvSVM::LINEAR; // which is better?
	_params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER, 100, 1e-6 );  
	//params.degree
	//params.gamma
	//params.coef0
	cv::Mat _response( nz_imgs, 1, CV_32FC1 );
	_response.rowRange( 0, np_imgs ) = 1;
	_response.rowRange( np_imgs, nz_imgs ) = -1;
	svm.train( _feature, _response, cv::Mat() , cv::Mat(), _params );
	svm.save( model_file );
	return 0;
}

void bof( cv::Mat _descriptors, cv::Mat& feature, std::vector<int> count_points, int nz_imgs, cv::Mat centers )//feature: output
{
	int n_imgs = count_points.size();
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
					dist.at<float>(j, k) = cv::norm( desc.row( j ) - centers.row( k ), cv::NORM_L2 );
				}
			}
			//printMat( dist, "D:/dist_former", "w");
			//cv::Mat multi_dist = (-0.001) * dist;
			//printMat( multi_dist, "D:/dist0.001", "w");
			
			cv::exp( (-0.001) * dist, dist );//from now on, mat dist is hold the similarity not the distance
			//printMat( dist, "D:/dist_after", "w");
			cv::sortIdx( dist, sorted_idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING );//sort each row descendently(indices are sorted actually)
			for( j = 0; j < num; j++ )
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




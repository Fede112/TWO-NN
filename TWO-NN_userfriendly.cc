// Usage: ./TWO-NN.x -input <filename> {-coord|-dist} [-discard <fraction>]


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex>
#include <functional>
#include <algorithm> 
#include <numeric>
#include <assert.h>
#include <limits>

using namespace std;




double DistValue(int i, int j, int N, const vector<double>& Dist, bool TriangSup){
	int ll, mm, kk;
	ll=max(i,j);
	mm=min(i, j);
	if(TriangSup) 
		kk=mm*(2*N-mm-3)/2.+ll;
	else
		kk=(ll*ll-ll)/2.+mm+1;
	return Dist[kk-1];
}

//Find the two nearest neighbors in the case of a file of coordinates

void find_nearest_coo(vector<double>& D1,vector<double>& D2, vector<double>& X, int N, int ncoords, bool periodicB){
	// vector<double>& D1: vector of distance to 1 neighbour for each particle.
	// vector<double>& D2: vector of distance to 2 neighbour for each particle.
	// vector<double> X: 1d vector of size N*ncoords. It contains the coordinates of all N particles.
	// N: number of particles
	// ncoords: dimension of the embedding space
	double maxdist=numeric_limits<double>::max();
	double dist, d1, d2;
	// L is the hypercube size for periodic boundary conditions	
	double L[ncoords]; 
	for(int cc=0; cc<ncoords; cc++){
	    	L[cc]=1.;  
	}
	double Xtemp[ncoords];

	// avoid double loop (can be improved)
	for(int i=0; i<N; i++){
		d1=maxdist;
		d2=maxdist;
		for(int j=0; j<N; j++){
			if(j!=i){
				dist=0.;
				for(int cc=0; cc<ncoords; cc++){
					// .at checks for out of bounds while [] doesn't
					Xtemp[cc]=X.at(i*ncoords+cc)-X.at(j*ncoords+cc);
					// check for boundary conditions (can be improved)
					if(periodicB){
						if(abs(Xtemp[cc])>L[cc]*0.5) 
							if(X.at(i*ncoords+cc)>X.at(j*ncoords+cc)) Xtemp[cc]=L[cc]-Xtemp[cc];
							else Xtemp[cc]=L[cc]+Xtemp[cc];
					}
					dist+=Xtemp[cc]*Xtemp[cc];
				}
				// can be avoided and work with dist^2 (can be improved)
				dist=sqrt(dist); 

				// if dist<d1 I think is automatic that dist<d2 (can be improved)
				if( dist<d1 && dist<d2 )
				{
					d2=d1;
					d1=dist;
				}
				else if( dist>=d1 && dist<d2 )
				{
				d2=dist;
				}                   
			}
		}
		D1.push_back(d1);
		D2.push_back(d2);
	}

	return;
}

//Find the two nearest neighbors in the case of a file of distances

void find_nearest_dist(vector<double>& D1, vector<double>& D2, const vector<double>& Dist, const int N, const bool TriangSup)
{
	for(int i=0; i<N; i++){
		double d1 = numeric_limits<double>::max();
		double d2 = numeric_limits<double>::max();
		for(int j=0; j<N; j++){
			if(j!=i){
				double dist = DistValue(i,j,N,Dist,TriangSup);
				if(dist<d1&&dist<d2){//controllare
					d2=d1;
					d1=dist;
				}
				else if(dist>=d1&&dist<d2){
					d2=dist;
				}
			}
		}
		D1.push_back(d1);
		D2.push_back(d2);
	}

}

void compute_d(vector<double>& D1, vector<double>& D2, double& dim, int N, double frac, int nbox, int& N_retained){

	// COMPUTE VECTOR NU
	

	vector<double> NU;

	double num, den;
	double nu;

	for(int i=0; i<N; i++)
	{
		num=D2[i];
		den=D1[i];

		nu=num/den;
			
		NU.push_back(nu);
	
		if(nu==1.) cout<<"Point "<<i<<" has the first two neighbors at the same distance!"<<endl;
	}

	// order linearithmic ( N.log_2(N) )
	sort(NU.begin(), NU.end());


	// FIT
 
	double XX[N], YY[N]; //XX=log(nu), YY=-log(1-F(nu))
  
	if (nbox==1) // only for the full dataset write the relevant files
	{
		ofstream file_rlist("r_list.dat"); // file containing the list of distances between the point and 											its first and second neighbor
		ofstream file_nulist("nu_list.dat"); // file containing the list of nu values   

		ofstream file_fun("fun.dat"); // file containing the coordinates to plot the S-set

		for(int i=0; i<N; i++)
		{
	 		XX[i] = log(NU.at(i));
			YY[i] = - log(1.-double(i)/double(N));

			file_fun << XX[i] << ' ' << YY[i] << endl;

			num = D2[i];
			den = D1[i];

			nu = num/den;
			
			file_rlist << den << ' ' << num << endl;
			file_nulist << nu << endl;
		}

		file_fun.close();
		file_nulist.close();
		file_rlist.close();

	}
  
	for(int i=0; i<N; i++)
	{
	 	XX[i] = log(NU.at(i));
		YY[i] = log(1.-double(i)/double(N));
	}

	double sumX, sumY, sumXY, sumX2, sumY2, sumErr;

	sumX=0.;
	sumY=0.;
	sumXY=0.;
	sumX2=0.;
	sumY2=0.;
	sumErr=0.;
 

	int Ncut=int(double(N)*frac);

	if (nbox==1)
	{
		N_retained=Ncut;
	}

	for(int i=0; i<Ncut; i++)
	{
		sumX += XX[i];
		sumY += YY[i];
		sumXY += XX[i] * YY[i];
		sumX2 += XX[i] * XX[i]; 
		sumY2 += YY[i] * YY[i];     
	}
  
	dim = -sumXY / sumX2;  // fit formula with the straight line through the origin a*x

	double minval = sqrt( sumY2 - sumXY*sumXY / sumX2 ) / double( Ncut );

	if (nbox==1) cout<< "estimated dimension= "<<dim<<' '<<endl;



}



bool parse_command_line(string& filename, bool& coordinates, double& frac_retained,
						int argc, char* argv[])
{
	static string usage_string =
		"\n"		
		"Usage: " + string(argv[0]) + " -input <filename> {-coord|-dist} [-discard <fraction>]\n"
		"\n"	
		"       -input:   specify the file from which to read the input data;\n"
		"       -coord:   interpret the input data as a table of coordinates, where\n"
		"                 each row is an n-dimensional list of numbers;\n"
		"       -dist:    interpret the input data as a list of distances, where each\n"
		"                 line is a triplet (i,j,dist) and dist is the distance between\n"
		"                 points i and j;\n"
		"       -discard: (optional) specify the fraction of points to discard in [0,1) ( default: 0.1)\n"
		"\n";	

	if (argc<=1) {
		cerr << usage_string << flush;
		return false;
	}

	bool found_input = false;
	bool found_format = false;
	for (int i=1; i<argc; i++) {
		if (!strcmp(argv[i], "-input")) {
			if(i+1 >= argc) {
				cerr << "\nERROR: Missing argument for option -input\n";
				return false;
			}
			found_input = true;
			filename = string(argv[i+1]);
			i++;
		} else if (!strcmp(argv[i], "-dist")) {
			found_format = true;
			coordinates = false;
		} else if (!strcmp(argv[i], "-coord")) {
			found_format = true;
			coordinates = true;
		} else if (!strcmp(argv[i], "-discard")) {
			if(i+1 >= argc) {
				cerr << "\nERROR: Missing argument for option -discard\n";
				return false;
			}
			double frac_discard = std::atof(argv[i+1]);
			if (frac_discard < 0. || frac_discard >= 1.) {
				cerr << "\nERROR: the fraction of discarded points must be between 0 and 1 (not included)" << endl;
				return false;
			}
			frac_retained = 1. - frac_discard;
			i++;
		} else {
			cerr << "\nERROR: Unrecognized option " << argv[i] << "\n";
			cerr << usage_string << flush;
			return false;
		}
	}
	
	if (!found_input) {
		cerr << "\nERROR: missing mandatory option -input\n" ;
	}
	if (!found_format) {
		cerr << "\nERROR: missing mandatory option -coord or -dist\n" ;
	}
	if ((!found_input) || (!found_format)) {
		cerr << usage_string << flush;
		return false;
	}
	
	return true;
}

//##########################################


int main(int argc, char* argv[])
{

	string file_in;
	bool coordinates=0;
	int N=0;
	double xx, yy;
	double dd;
	double rd;
	string line;
	int ncoords=0;
	vector<double> X, Y;
	vector<double> D1, D2;
	bool TriangSup=1;
	bool periodicB=0;
	double frac=0.9;
	double rdval=0.000005; //order of the random correction in case of null distances;
						   //(null distances should be avoided!)

	if (!parse_command_line(file_in, coordinates, frac, argc, argv)) 
	{
		exit(EXIT_FAILURE);
	};

	ifstream file_in_1(file_in.c_str());
	double MAXdist;

	long int Ntemp=0;
	vector<double> Dist;

	// store the input file in a vector: X for coordinates, Dist for distances 
	// and generate vectors D1 D2 containing the first and second neighbour for each point

	if(coordinates)
	{
		getline(file_in_1, line);
		stringstream sline(line);
		while (sline >> xx)
		{
                  	ncoords++;
			X.push_back(xx);
		}
		N++;
		while(getline(file_in_1, line))
		{
			stringstream sline(line);
			while (sline >> xx)
			{
				X.push_back(xx);
			}
			N++;
		}
		if (file_in_1.fail() && !(file_in_1.eof())) 
		{
			cout << "Errore di lettura!" << endl;
		}  
		//cout<<" * DO you want periodic boundary conditions? (YES=1, NO=0)  PER ORA METTI SEMPRE NO  ";
		//cin>>periodicB;  
		//cout<<endl;

         }
	else{    
		while(true) {//dddd
			file_in_1 >> xx >> yy >> dd;
			//file_in_1 >> dd;
			if (file_in_1.good()) 
			{
				Ntemp++;
				if(dd==0) 
				{
					cout<<"A null distance has been provided:it is corrected by a random number of the order "<<rdval<<endl;
					rd=(double)rand()/RAND_MAX;        
					rd=rd*rdval; 
					Dist.push_back(dd+rd);
				} else 
				{
					Dist.push_back(dd);
				}
			} else if (file_in_1.eof()) 
			{
				break;
			} else 
			{
				cout << "Errore di lettura!" << endl;
				exit(EXIT_FAILURE);
			}
		}
		if(xx>yy) TriangSup=0;
		cout << "xx=" << xx << ", yy=" << yy << ", TriangSup=" << TriangSup << endl;
		cout << "Ntemp=" << Ntemp << endl;  

		MAXdist=*max_element(Dist.begin(),Dist.end());
		N=(int)((1.+sqrt(1.+8.*Ntemp))*0.5); //Number of points in the dataset
	}

	// RESHUFFLE THE DATASET

	vector<int> IND;
	IND.reserve(N);

	for(int i=0; i<N; i++)
	{
		IND.push_back(i);
	}
	std::random_shuffle(IND.begin(), IND.end());

	// BLOCK ANALYSIS

	int NBOX[]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20};
	//int NBOX[]={1};

	ofstream file_ba("block_analysis.dat");

	int Ndat=sizeof(NBOX) / sizeof(NBOX[0]);
    int N_retained;
         
	double AVE_D[Ndat];
	double VAR_D[Ndat];

	double dim;
	double dim_tot;

	for(int i=0; i<Ndat; i++)
	{
		AVE_D[i]=0;
		VAR_D[i]=0;	
	}

	cout<<endl;

	for(int i=0; i<Ndat; i++)
	{

		int nbox=NBOX[i];
		int Npoints=N/nbox;
		double ave_temp=0;
		double var_temp=0;
		cout<<"nbox"<<' '<<nbox<<endl;
		cout<<endl;

		for(int box=0; box<nbox; box++)
		{
				
			if(coordinates)
			{

				vector<int> IND_loc(IND.begin()+(Npoints*box),IND.begin()+(Npoints*(box+1)));
				vector<double> D1_loc, D2_loc;		    
				vector<double> X_loc;
				X_loc.reserve(Npoints*ncoords);
				
				for (int j=0; j<IND_loc.size(); j++)
				{
					int ind=IND_loc[j];

					for (int k=0; k<ncoords; k++) 
					{
						X_loc.push_back(X.at(ind*ncoords+k));
					}
				}

				assert(X_loc.size()==Npoints*ncoords);
				
				find_nearest_coo( D1_loc, D2_loc, X_loc, Npoints, ncoords, periodicB); 

				compute_d( D1_loc, D2_loc, dim, Npoints, frac, nbox, N_retained);
				AVE_D[i]+=dim;
				VAR_D[i]+=dim*dim;
			}else
			{
				vector<int> IND_loc(IND.begin()+(Npoints*box),IND.begin()+(Npoints*(box+1)));
				vector<double> D1_loc, D2_loc;		    
				vector<double> Dist_loc;
				Dist_loc.reserve(Npoints*(Npoints-1)/2);

				for (int j=0; j<IND_loc.size(); j++)
				{
					for (int k=j+1; k<IND_loc.size(); k++)
					{
						int ind1=IND_loc[j];
						int ind2=IND_loc[k];
						double dist=DistValue(ind1,ind2,N,Dist,TriangSup);
						Dist_loc.push_back(dist);
					}
				}
				assert(Dist_loc.size()==Npoints*(Npoints-1)/2);
				
				find_nearest_dist(D1_loc, D2_loc, Dist_loc, Npoints, TriangSup);

				compute_d( D1_loc, D2_loc, dim, Npoints, frac, nbox, N_retained);
				AVE_D[i]+=dim;
				VAR_D[i]+=dim*dim;
			}
			
		}

		double a=AVE_D[i]/NBOX[i];
		double b=VAR_D[i]/NBOX[i];

		file_ba<<Npoints<<' '<<a<<' '<<sqrt(b-a*a)<<endl;

		if(i==0) dim_tot=dim;
	}

	int upper_bound=int(dim_tot+10);
	int lower_bound=max(int(dim_tot-10),0);

	file_ba.close();



	// block analysis plot


	ofstream file_gpl_ba("block_analysis.gpl");

	file_gpl_ba<< "set key top right box lw 1  lc rgb \"#7F7F7F\" font \",10\" spacing 10" <<endl;
	file_gpl_ba<< "set border 4095 lw 1 lc rgb \"#7F7F7F\" " <<endl;
	file_gpl_ba<< "set grid lw 1" <<endl;
	file_gpl_ba<< "set yrange["<<lower_bound<<":"<<upper_bound<<"]"<<endl; 
	file_gpl_ba<< "set title \"block analysis "<<file_in<<"\" font \",20\"" <<endl;
	file_gpl_ba<< "set xlabel \"N\" font \",15\"" <<endl;
	file_gpl_ba<< "set ylabel \"d\" font \",15\" rotate by 0" <<endl;
	file_gpl_ba<< "p \'block_analysis.dat\' u 1:2 w lp lw 2 lc rgb \"#DC143C\" not, '' u 1:2:3 w e lc rgb \"#DC143C\" not, "<< dim_tot<<" lw 2 lc rgb \"#00BFFF\" t \"  full dataset dimension\"" <<endl;
	system("gnuplot -persist block_analysis.gpl");

	file_gpl_ba.close();
	
	// S-set plot
	
	ofstream file_gpl_fun("S_set.gpl");
    	file_gpl_fun<<fixed;
	file_gpl_fun.precision(2);
	file_gpl_fun<< "set key top left box lw 1  lc rgb \"#7F7F7F\" font \",10\" spacing 5" <<endl;
	file_gpl_fun<< "set border 4095 lw 1 lc rgb \"#7F7F7F\" " <<endl; 
	file_gpl_fun<< "set title \"S set "<<file_in<<"\" font \",20\"" <<endl;
	file_gpl_fun<< "p \'fun.dat\' lc rgb \"#7F7F7F\" pt 7 not, "<<"\"<head -"<<N_retained<<" 'fun.dat'\" lc rgb \"#DC143C\" pt 7 t \"S set\", "<<dim_tot<<"*x"<<" lw 2 lc rgb \"#00BFFF\" t \" "<<dim_tot<<"*x\"" <<endl;
	system("gnuplot -persist S_set.gpl");

	file_gpl_fun.close();
		
	return 0;


	
}






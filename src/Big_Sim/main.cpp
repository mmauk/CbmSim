

#include <time.h>
#include <iostream>
#include <fstream>

#include <stdDefinitions/pstdint.h>

#include <interfaces/cbmstate.h>
#include <interfaces/iconnectivityparams.h>
#include <interfaces/iactivityparams.h>

#include <interface/cbmsimcore.h>

#include <ecmfpopulation.h>
#include <poissonregencells.h>

#include "control.h"


int **paramArrayPre;
int **paramArray;

using namespace std;

int main() {

	int numTrainingTrials = 50, homeoTuningTrials = 0, granuleActivityDetectionTrials = 0;

	int totalNumTrials = homeoTuningTrials+granuleActivityDetectionTrials+numTrainingTrials;  
	float csTonicF = 0.05;
	float CSlength[3] = {2000, 1000, 1500};//{250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750};
	float goMin = 0.26;

	float GRGO, MFGO, GOGR;
	
	float spillFrac = 0.15;

	float csRateMin = 100.0;
	float csRateMax = 110.0;

	//float grW = 0.0007;
	float mfW = 0.0035;
	//float ws[5] = {0.9125, 0.975, 1.0375, 1.1, 1.1625};//0.9125;
	//float gogoW[5] = {0.0125, 0.107, 0.0094, 0.0083, 0.0075};
	//float ws[25] = {0.75, 0.8, 0.85, 0.9, 0.975, 1.05, 1.0875, 1.125, 1.1625, 1.2, 1.25, 1.3, 1.375, 1.45, 1.525, 1.6, 1.625, 1.65, 1.675, 1.7, 1.7125, 1.725, 1.7375, 1.75, 1.7625}; //0.2
	//float ws = 1.02;
	//float ws[25] = {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.025, 1.05, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1};//0.1
	float numcon[21] = {1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40}; 

	//0.0125 gauss span 10
	//float ws[21] = {0.37, 0.45, 0.54, 0.66, 0.8, 0.9, 1.0, 1.07, 1.17, 1.22, 1.28, 1.34, 1.39, 1.48, 1.55, 1.62, 1.745, 1.84, 1.94, 2.02, 2.01};
	//float gogr[21] = {0.0125, 0.018, 0.01875, 0.018, 0.017, 0.017, 0.015, 0.015, 0.013, 0.012, 0.01175, 0.011, 0.0109, 0.0103, 0.01, 0.0095, 0.00935, 0.00915, 0.00902, 0.0089, 0.0088};


	//0.0125 span 2
	//float ws[5] = {0.37, 0.41, 0.525, 0.635, 0.715};
	//float gogr[5] = {0.0124, 0.016, 0.018, 0.016, 0.0138};

	//0.0125 flat span 4
	//float ws[13] = {0.37, 0.42, 0.528, 0.658, 0.77, 0.88, 0.961, 1.04, 1.114, 1.17, 1.22, 1.3, 1.4};
	//float gogr[13] = {0.0124, 0.016, 0.01875, 0.0174, 0.01675, 0.0156, 0.015, 0.01421, 0.0134, 0.0125, 0.012, 0.011, 0.01};

	//0.0125 flat span 6
	//float ws[21] = {0.37, 0.45, 0.54, 0.66, 0.8, 0.91, 1.015, 1.08, 1.161, 1.23, 1.29, 1.349, 1.415, 1.49, 1.575, 1.65, 1.72, 1.8, 1.94, 2.04, 2.09};
	//float gogr[21] = {0.0125, 0.018, 0.01875, 0.0185, 0.01825, 0.018, 0.0165, 0.01575, 0.0148, 0.014, 0.0129, 0.012, 0.0115, 0.011, 0.0105, 0.0105, 0.0102, 0.01, 0.0095, 0.0089, 0.0089};

	//0.0125 flat span 8
	//float ws[21] = {0.37, 0.45, 0.54, 0.665, 0.8, 0.92, 1.018, 1.12, 1.2, 1.3, 1.365, 1.4, 1.49, 1.57, 1.62, 1.695, 1.775, 1.84, 1.92, 2.01, 2.09};
	//float gogr[21] = {0.0125, 0.018, 0.01875, 0.0186, 0.0184, 0.018, 0.0174, 0.0168, 0.0156, 0.014, 0.0132, 0.0125, 0.0123, 0.01125, 0.011, 0.011, 0.0105, 0.0102, 0.01, 0.0099, 0.00975};

	//0.0125 flat span 10
	//float ws[21] = {0.37, 0.45, 0.5375, 0.665, 0.797, 0.935, 1.05, 1.15, 1.228, 1.31, 1.38, 1.455, 1.515, 1.585, 1.66, 1.74, 1.8, 1.88, 1.94, 2.02, 2.075};
	//float gogr[21] = {0.0125, 0.018, 0.01935, 0.0187, 0.0182, 0.01815, 0.017, 0.0165, 0.0152, 0.0145, 0.0131, 0.01225, 0.012, 0.0116, 0.011, 0.0105, 0.0105, 0.0103, 0.0104, 0.0103, 0.01};

	//0.0125 span all
	//float ws[21] = {0.37, 0.45, 0.5375, 0.665, 0.797, 0.94, 1.05, 1.15, 1.23, 1.31, 1.39, 1.46, 1.52, 1.59, 1.67, 1.74, 1.8, 1.88, 1.94, 2.02, 2.075};
	//float gogr[21] = {0.0125, 0.018, 0.0189, 0.0185, 0.0182, 0.01815, 0.017, 0.0165, 0.0155, 0.0145, 0.0135, 0.0125, 0.012, 0.01175, 0.0115, 0.011, 0.011, 0.0107, 0.0104, 0.0104, 0.01};

	float ws = 0.3275;//0.3275;//0.961;
	float gogr = 0.0105;//0.0105;//0.015;

	float grW[8] = {0.00056, 0.0007, 0.000933, 0.0014, 0.0028, 0.0056, 0.0112, 0.0224};
    int grWLength = sizeof(grW) / sizeof(grW[0]);

	float gogoW;
	clock_t time;
	time = clock();
    	
	for(int i=0; i < 1; i++){	
		cout << "ParamNum:  " << i << endl;
		
		for(int j=0; j < 1; j++){
			
			for(int k=0; k < grWLength; k++){
					
				GRGO = grW[k] * ws;
				MFGO = mfW * ws;
				GOGR = gogr;
				gogoW = 0.0125;			

				for(int num=0; num<10; num++){
					
					Control joeSimulation;
					//joeSimulation.runTuningSimulation(homeoTuningTrials, totalNumTrials, num, CSlength, csTonicF, goMin);
					joeSimulation.runSimulationWithGRdata(i, k, homeoTuningTrials, granuleActivityDetectionTrials, totalNumTrials, num, CSlength[i], csTonicF, goMin, GOGR, GRGO, MFGO, csRateMin, csRateMax, gogoW, j, i, spillFrac);
				}
			}
		}
	}
	time = clock() - time;
	cout << "Simulation time seconds: " << (float) time / CLOCKS_PER_SEC << endl;
}


#include <string>
#include <time.h>
#include "control.h"

using namespace std;


Control::Control(){};

Control::~Control(){};

void Control::runSimulationWithGRdata(int fileNum, int goRecipParam, int tuningTrials, int grDetectionTrials, int numTrials, int simNum, int csSize, float csFracMFs, float goMin, float GOGR, float GRGO, float MFGO, float csMinRate, float csMaxRate, float gogoW, int inputStrength, int inputWeight_two, float spillFrac){
	

	cout << "fileNum:  " << fileNum << endl;
	SetSim simulation(fileNum, goRecipParam, simNum);
	joestate = simulation.getstate();
	joesim = simulation.getsim();
	joeMFFreq = simulation.getMFFreq(csMinRate, csMaxRate);
	joeMFs = simulation.getMFs();	

	int preTrialNumber = tuningTrials+grDetectionTrials;
	int collectionTrials = numTrials;//numTrials - preTrialNumber;
	
	
	
//	mfNoiseRates = allocate2DArray<float>(5000*100, numGO);	
//	arrayInitialize<float>(mfNoiseRates[0], 0, numGO*5000*100);

//	mfR = new float[numGO];
	
	
	cout << "Done filling MF arrays" << endl;	

	int recipName[25] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50};
	//int numCon[25] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50};
	//int numCon[25] = {2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50};
	int conv[8] = {5000, 4000, 3000, 2000, 1000, 500, 250, 125};
//Set up granule cell data collection
	//Find granule cells with CS MF input	
/*	int* granuleIndWithInput =  getGRIndicies(csFracMFs); 
	int numpActiveGranule = getNumGRIndicies(csFracMFs);	
	cout << "Number of Granule cells w/ inputs	" << numpActiveGranule << endl;	
	//Allocate and Initialize 
	int activeCounter=0;
	int *activeGranuleIndex;
	activeGranuleIndex = new int[numpActiveGranule];
	for(int i=0; i<numpActiveGranule; i++){activeGranuleIndex[i] = false;}
	grPSTHCS = new ct_uint8_t[numpActiveGranule];
	for(int i=0; i<numpActiveGranule; i++){grPSTHCS[i] = 0;}
	grPSTHPreCS = new ct_uint8_t[numpActiveGranule];
	for(int i=0; i<numpActiveGranule; i++){grPSTHPreCS[i] = 0;}
	preGRPSTHCS=allocate2DArray<ct_uint8_t>(numpActiveGranule, csSize);
	arrayInitialize<ct_uint8_t>(preGRPSTHCS[0], 0, numpActiveGranule*csSize);
	preGRPSTHPreCS=allocate2DArray<ct_uint8_t>(numpActiveGranule, csSize);
	arrayInitialize<ct_uint8_t>(preGRPSTHPreCS[0], 0, numpActiveGranule*csSize);

	int grSaveNumber = 30000;
*/
	//grpcWeights = allocate2DArray<float>(numPC, 32768);
	//arrayInitialize<float>(grpcWeights[0], 0, numPC*32768);
// Allocate and Initialize PSTH and Raster arrays
	allPCRaster = allocate2DArray<ct_uint8_t>(numPC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allPCRaster[0], 0, numPC*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	
	allNCRaster = allocate2DArray<ct_uint8_t>(numNC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allNCRaster[0], 0, numNC*(csSize+msPreCS+msPostCS)*(collectionTrials) );

	allSCRaster = allocate2DArray<ct_uint8_t>(numSC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allSCRaster[0], 0, numSC*(csSize+msPreCS+msPostCS)*(collectionTrials) );

	allBCRaster = allocate2DArray<ct_uint8_t>(numBC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allBCRaster[0], 0, numBC*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	
//	allIORaster = allocate2DArray<ct_uint8_t>(numIO, (5000)*(collectionTrials));	
//	arrayInitialize<ct_uint8_t>(allIORaster[0], 0, numIO*(5000)*(collectionTrials) );
	


	//allGORaster = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	//arrayInitialize<ct_uint8_t>(allGORaster[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
cout << "PC arrays" << endl;	
/*	allGORaster = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allGORaster[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	allGORaster1 = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allGORaster1[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	allGORaster2 = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allGORaster2[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	allGORaster3 = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allGORaster3[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	allGORaster4 = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allGORaster4[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	allGORaster5 = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allGORaster5[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	allGORaster6 = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allGORaster6[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	allGORaster7 = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allGORaster7[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	allGORaster8 = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allGORaster8[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
*/	

	allGOPSTH = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS));	
	arrayInitialize<ct_uint8_t>(allGOPSTH[0], 0, numGO*(csSize+msPreCS+msPostCS) );

//	allGORaster = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
//	arrayInitialize<ct_uint8_t>(allGORaster[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );

//	allGOGORaster = allocate2DArray<float>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
//	arrayInitialize<float>(allGOGORaster[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	
//	allGRGORaster = allocate2DArray<float>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
//	arrayInitialize<float>(allGRGORaster[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	
	float medTrials;
        float *mTall = new float[numTrials];
        float *grgoGall = new float[numTrials];
        float *mfgoGall = new float[numTrials];
	//allGORaster_gogoG = allocate2DArray<float>(numGO, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	//arrayInitialize<float>(allGORaster_gogoG[0], 0, numGO*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	

	

//	allPCRaster = allocate2DArray<ct_uint8_t>(numPC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
//	arrayInitialize<ct_uint8_t>(allPCRaster[0], 0, numPC*(csSize+msPreCS+msPostCS)*(collectionTrials) );
	
	
	
	



	//int trialsPerMFRate = 10;
	


	clock_t T;
	int rasterCounter = 0;
	int rasterCounterIO = 0;
	//clock_t tt;
	int grSpkCounter = 0;
	int grSpkCounterPre = 0;
	
	
	vector<int> goSpkCounter;
	goSpkCounter.assign(numGO, 0);
	//int *goSpkCounterPre;
	//goSpkCounterPre = new int[numGO];
	
	
	float r;
	int tsCSCounter = 0;
	//mfBackRate = new float[4096]; 
	for (trial = 0; trial < numTrials; trial++){
		T = clock();


		for(int i=0; i<numGO; i++){
			goSpkCounter[i] = 0;
			//goSpkCounterPre[i] = 0;
		}
		float gGRGO_sum = 0;
		float gMFGO_sum = 0;


		if(trial<=tuningTrials){
			cout << "TrialT" << trial << endl;
			trialTime = 5000;	
		}
		if(trial>tuningTrials){
			cout << "TrialR" << trial << endl;
			trialTime = 5000;	
		}
/*		if(trial == 0){
			tsCSCounter = 0;
			rasterCounter = 0;	
			ifstream myReadFile;
			myReadFile.open("SimulationInput/rateCS_ISI"+to_string(csSize)+"_randomWalk_csNoise_withColl_noiselvl_0_0.bin", ios::in | ios::binary);
			;for(int i=0; i<100*5000; i++){
				for(int j=0; j<numGO; j++){
					myReadFile.read((char*) &mfNoiseRates[i][j], sizeof(float));
				}	
			}
			myReadFile.close();	
		
			for(int i=0; i<5000; i++){
				cout << mfNoiseRates[i][0] << " ";
			}
		
		}
*/
		int PSTHCounter = 0;	
		int grPSTHCounter = 0;	
		int preCounterGRCS = 0;
		int preCounterGRPre = 0;
		//int rasterCounter = 0;
		

		// Homeostatic plasticity trials

		if(trial >= tuningTrials){
			// Run active granule cell detection 	
			if(trial == preTrialNumber){			
				int spk;
				int binCountPre;
				int binCountCS;
			/*	for(int grInd=0; grInd<numpActiveGranule; grInd++)
				{
					for(int timeStep=0; timeStep<csSize; timeStep++)
					{
						binCountCS = preGRPSTHCS[grInd][timeStep];
						grPSTHCS[grInd] += binCountCS;
					}
					
					for(int timeStep=0; timeStep<csSize; timeStep++)
					{
						binCountPre = preGRPSTHPreCS[grInd][timeStep];
						grPSTHPreCS[grInd] += binCountPre;
					}
		
					
					if(grPSTHCS[grInd] >= (grPSTHPreCS[grInd]+20)){
						activeGranuleIndex[activeCounter] = granuleIndWithInput[grInd];	
						activeCounter++;
					}
				
				
				}
			*/	
			//	cout << "		*****Golgi median C0 Rate:	" << medTrials / 20.0 << " ****** "<<endl; 
				//ofstream fileActGRs;
				//fileActGRs.open("activeGRInds.txt");
				//for(int i=0; i<activeCounter; i++){
				//	fileActGRs << activeGranuleIndex[i] << endl;
				//}
				//cout << "Active Granule Number:		" << activeCounter << endl;
				//cout << "GO CS rate:	" << (float)goSpkCounter / (2.0*10.0*(float)numGO) << endl;;
				//cout << "GO Base rate:	" << (float)goSpkCounterPre / (2.0*10.0*(float)numGO) << endl;;
			//	cout << "GR CS rate:	" << (float)grSpkCounter / (10.0*(float)numGR) << endl;;
			//	cout << "GR Base rate:	" << (float)grSpkCounterPre / (10.0*(float)numGR) << endl;;
					
			//	activeGRPSTH=allocate2DArray<ct_uint8_t>(activeCounter, (csSize+msPreCS+msPostCS));
			//	arrayInitialize<ct_uint8_t>(activeGRPSTH[0], 0, activeCounter*(csSize+msPreCS+msPostCS) );		
				
			//	activeGRRaster = allocate2DArray<ct_uint8_t>(grSaveNumber, (csSize+msPreCS+msPostCS)*(collectionTrials));	
			//	arrayInitialize<ct_uint8_t>(activeGRRaster[0], 0, grSaveNumber*(csSize+msPreCS+msPostCS)*(collectionTrials) );
			}
			
			
			
			
			for (tts = 0 ; tts <trialTime; tts++){
			//	for(int i=0; i<numGO; i++){
			//		mfR[i] =  mfNoiseRates[tts][i];
			//	}
				//tsCSCounter++;			
				//mfAP = joeMFs->calcPoissActivity(mfR, joesim->getMZoneList());
				// Deliver US 
				if (tts == csSize + csStart  ){joesim->updateErrDrive(0,0.0);}
				// Background MF activity in the Pre and Post CS period
				if(tts < csStart || tts >= csStart+csSize){ mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFBG(), joesim->getMZoneList() ); }	
				// Phasic MF activity during the CS for a duration set in control.h 
				else if(tts >= csStart && tts < csStart + csPhasicSize){mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFFreqInCSPhasic(), joesim->getMZoneList());}
				// Tonic MF activity during the CS period
				else{mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFInCSTonicA(), joesim->getMZoneList());}

				bool *isTrueMF = joeMFs->calcTrueMFs(joeMFFreq->getMFBG());
				joesim->updateTrueMFs(isTrueMF);
				joesim->updateMFInput(mfAP);
				joesim->calcActivity(goMin, simNum, GOGR, GRGO, MFGO, gogoW, spillFrac);	
				
				//if(trial == numTrials-1 && tts == 0){
				//	grpcWeights = joesim->getMZoneList()[0]->exportPFPCWeights();  
				
				//}


                                if(tts>=csStart && tts<csStart+csSize){

                                        //grSpks = joesim->getInputNet()->exportAPGR();
                                        //for(int i=0; i<numpActiveGranule; i++){
                                //              preGRPSTHCS[i][preCounterGRCS] = preGRPSTHCS[i][preCounterGRCS] + grSpks[ granuleIndWithInput[i] ];
                                //      }
                                //      preCounterGRCS++;

                                        mfgoG = joesim->getInputNet()->exportgSum_MFGO();
                                        grgoG = joesim->getInputNet()->exportgSum_GRGO();
                                        goSpks = joesim->getInputNet()->exportAPGO();
                                        for(int i=0; i<numGO; i++){
                                                goSpkCounter[i] += goSpks[i];
                                                gGRGO_sum += grgoG[i];
                                                gMFGO_sum += mfgoG[i];
                                        }

                                //      for(int i=0; i<numGR; i++){
                                //              grSpkCounter += grSpks[i];
                                //      }
                                }
                                if(tts == csStart+csSize){
                                        int n = sizeof(goSpkCounter) / sizeof(goSpkCounter[0]);
                                        sort(goSpkCounter.begin(), goSpkCounter.begin()+4096);
                                        int m = (goSpkCounter[2047] + goSpkCounter[2048])/2.0;

                                        float goSpkSum = 0;
                                        for(int i=0; i<numGO; i++){
                                                goSpkSum+=goSpkCounter[i];
                                        }
                                        cout << "Mean GO Rate:  " << goSpkSum/(float)numGO << endl;

                                        medTrials += m / 2.0;
                                        cout << "Median GO Rate:  " << m / 2.0 << endl;

                                        mTall[trial] = m / 2.0;
                                        cout << "mean gGRGO = " << gGRGO_sum / (numGO*csSize) << endl;
                                        cout << "mean gMFGO = " << gMFGO_sum / (numGO*csSize) << endl;
                                        cout << "       GR:MF ratio  =  " << gGRGO_sum / gMFGO_sum << endl;

                                        grgoGall[trial] = gGRGO_sum / (numGO*csSize);
                                        mfgoGall[trial] = gMFGO_sum / (numGO*csSize);

                                }



				// CS GR activity
			/*	if(trial<preTrialNumber && trial>=tuningTrials && tts>=csStart && tts<csStart+csSize){
					
					//grSpks = joesim->getInputNet()->exportAPGR();			
					//for(int i=0; i<numpActiveGranule; i++){
				//		preGRPSTHCS[i][preCounterGRCS] = preGRPSTHCS[i][preCounterGRCS] + grSpks[ granuleIndWithInput[i] ];
				//	}
				//	preCounterGRCS++;
				
					goSpks = joesim->getInputNet()->exportAPGO();
					for(int i=0; i<numGO; i++){
						goSpkCounter[i] += goSpks[i];	
					}
				
				//	for(int i=0; i<numGR; i++){
				//		grSpkCounter += grSpks[i];	
				//	}
				}
				if(tts == csStart+csSize){
					int n = sizeof(goSpkCounter) / sizeof(goSpkCounter[0]);
					sort(goSpkCounter.begin(), goSpkCounter.begin()+4096);
					int m = (goSpkCounter[2047] + goSpkCounter[2048])/2.0;	
					
					medTrials += m / 2.0; 
					cout << m / 2.0 << endl;
				}	
				// preCS GR activity 
				if(trial<preTrialNumber && trial>=tuningTrials && tts>=csStart-csSize && tts<csStart){
					
				//	grSpks = joesim->getInputNet()->exportAPGR();			
				//	for(int i=0; i<numpActiveGranule; i++){
				//		preGRPSTHPreCS[i][preCounterGRPre] = preGRPSTHPreCS[i][preCounterGRPre] + grSpks[ granuleIndWithInput[i] ];
				//	}
				//	preCounterGRPre++;
					goSpks = joesim->getInputNet()->exportAPGO();
					for(int i=0; i<numGO; i++){
						//goSpkCounterPre[i] += goSpks[i];	
					}
					//for(int i=0; i<numGR; i++){
					//	grSpkCounterPre += grSpks[i];	
					//}
				}*/
				
				if(trial>=preTrialNumber && tts>=csStart-msPreCS && tts<csStart+csSize+msPostCS){
					
					//Granule collection
			//		grSpks = joesim->getInputNet()->exportAPGR();			
			//		for(int i=0; i<grSaveNumber; i++){	
			//			activeGRPSTH[i][grPSTHCounter] = activeGRPSTH[i][grPSTHCounter] + grSpks[ activeGranuleIndex[i]  ]; 
				//		activeGRRaster[i][rasterCounter] = grSpks[i];
			//		}
			//		grPSTHCounter++;
				
					//Golgi
				//	for(int i=0; i<numGO; i++){
				//		allGORaster[i][rasterCounter] = mfAP[i];
				//	}

					//gogoG = joesim->getInputNet()->exportgSum_GOGO();
					//grgoG = joesim->getInputNet()->exportgSum_GRGO(); 
				//	goSpks = joesim->getInputNet()->exportAPGO();
				//		for(int i=0; i<numGO; i++){	
					//		allGOGORaster[i][rasterCounter] = gogoG[i];
					//		allGRGORaster[i][rasterCounter] = grgoG[i];
					//		allGORaster[i][rasterCounter] = goSpks[i];
				//			allGOPSTH[i][PSTHCounter] = allGOPSTH[i][PSTHCounter] + goSpks[i];
				//		}

					//PKJ
					const ct_uint8_t* pcSpks=joesim->getMZoneList()[0]->exportAPPC();
					for(int i=0; i<numPC; i++){
						allPCRaster[i][rasterCounter] = pcSpks[i];
					}
					
					//NC
					const ct_uint8_t* ncSpks=joesim->getMZoneList()[0]->exportAPNC();
					for(int i=0; i<numNC; i++){
						allNCRaster[i][rasterCounter] = ncSpks[i];
					}
					
					
					//BC
	  				const ct_uint8_t* bcSpks=joesim->getMZoneList()[0]->exportAPBC();
					for(int i=0; i<numBC; i++){
						allBCRaster[i][rasterCounter] = bcSpks[i];
					}
					//SC
	  				const ct_uint8_t* scSpks=joesim->getInputNet()->exportAPSC();
					for(int i=0; i<numSC; i++){
						allSCRaster[i][rasterCounter] = scSpks[i];
					}	
					PSTHCounter++;
					rasterCounter++;
				}

			}
		}
		
	T = clock() - T;
	cout << "Trial time seconds:	" << ((float)T)/CLOCKS_PER_SEC << endl;

	}




	
 /*     float medRate_sum = 0;
        for(int i=0; i<numTrials; i++){
                medRate_sum += mTall[i];
        }
        cout << "median GO Rate:   "<< medRate_sum / ((float)numTrials) << endl;

        float grgosum = 0;
        for(int i=0; i<numTrials; i++){
                grgosum += grgoGall[i];
        }
        cout << "avg GRGO conductance:   "<< grgosum / ((float)numTrials) << endl;
        float mfgosum = 0;
        for(int i=0; i<numTrials; i++){
                mfgosum += mfgoGall[i];
        }
        cout << "avg MFGO conductance:   "<< mfgosum / ((float)numTrials) << endl;


        cout << "               GR:GO Ratio:   " << (grgosum / ((float)numTrials)) / (mfgosum / ((float)numTrials)) << endl;

*/
	delete joestate;
	delete joesim;
	delete joeMFFreq;
	delete joeMFs;
	//delete2DArray<float>(mfNoiseRates);
	//`delete[] mfR;	 



// Save Data 
	ofstream myfilegogoGbin("allGOPSTH_noGOGO_grgoConv"+to_string(conv[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS); j++){
			myfilegogoGbin.write((char*) &allGOPSTH[i][j], sizeof(ct_uint8_t));
		}
	}
	myfilegogoGbin.close();
	delete2DArray<ct_uint8_t>(allGOPSTH);
	
	
/*	ofstream myfileGOrasterbin("allGORaster_wGOGO_grgoConv"+to_string(conv[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS)*(collectionTrials); j++){
			myfileGOrasterbin.write((char*) &allGORaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGOrasterbin.close();
	delete2DArray<ct_uint8_t>(allGORaster);
*/


/*	ofstream myfileGOGOrasterbin("allGOGORaster_grgoConv"+to_string(recipName[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS)*(collectionTrials); j++){
			myfileGOGOrasterbin.write((char*) &allGOGORaster[i][j], sizeof(float));
		}
	}
	myfileGOGOrasterbin.close();
	delete2DArray<float>(allGOGORaster);
	
	ofstream myfileGRGOrasterbin("allGRGORaster_grgoConv"+to_string(recipName[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS)*(collectionTrials); j++){
			myfileGRGOrasterbin.write((char*) &allGRGORaster[i][j], sizeof(float));
		}
	}
	myfileGRGOrasterbin.close();
	delete2DArray<float>(allGRGORaster);
*/
/*	cout << "Filling MF file" << endl;
	ofstream myfileMFbin("mfRasterChunk.bin", ios::out | ios::binary);	
	for(int i=0; i<4096; i++){
		for(int j=0; j<10000*100; j++){
			myfileMFbin.write((char*) &mfRaster_chunk[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileMFbin.close();
	delete2DArray<ct_uint8_t>(mfRaster_chunk);
*/	
		/*cout << "Filling GO Baseline file" << endl;
	ofstream myfileGOBasebin("allGORaster_coup0.0__min"+to_string(goMin)+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<10000; j++){
			myfileGOBasebin.write((char*) &allGORaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGOBasebin.close();
	delete2DArray<ct_uint8_t>(allGORaster);*/
/*	
	cout << "Filling GR file" << endl;
	ofstream myfileGRbin("allGRRaster_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<grSaveNumber; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGRbin.write((char*) &activeGRRaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGRbin.close();
	delete2DArray<ct_uint8_t>(activeGRRaster);
	
	ofstream fileGRNumber;
	fileGRNumber.open("activeGRNumber_"+to_string(simNum)+".txt");	
	fileGRNumber << activeCounter << endl;
	fileGRNumber.close();
*/	
	
	

	
	
/*	ofstream myfileGObin("allPCRaster.bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<1000*(csSize+msPreCS+msPostCS); j++){
			myfileGObin.write((char*) &allPCRaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin.close();
	delete2DArray<ct_uint8_t>(allPCRaster);
*/	
	
	
	/*
	
	
	
	cout << "Filling GO file 0" << endl;
	ofstream myfileGObin("allPCRaster_0_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<1000*(csSize+msPreCS+msPostCS); j++){
			myfileGObin.write((char*) &allPCRaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin.close();
	delete2DArray<ct_uint8_t>(allPCRaster);
	
	cout << "Filling GO file 1" << endl;
	ofstream myfileGObin1("allPCRaster_1_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGObin1.write((char*) &allPCRaster1[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin1.close();
	delete2DArray<ct_uint8_t>(allPCRaster1);
	
	cout << "Filling GO file 2" << endl;
	ofstream myfileGObin2("allPCRaster_2_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGObin2.write((char*) &allPCRaster2[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin2.close();
	delete2DArray<ct_uint8_t>(allPCRaster2);
	
	cout << "Filling GO file 3" << endl;
	ofstream myfileGObin3("allPCRaster_3_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGObin3.write((char*) &allPCRaster3[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin3.close();
	delete2DArray<ct_uint8_t>(allPCRaster3);
	
	cout << "Filling GO file 4" << endl;
	ofstream myfileGObin4("allPCRaster_4_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGObin4.write((char*) &allPCRaster4[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin4.close();
	delete2DArray<ct_uint8_t>(allPCRaster4);
	
	cout << "Filling GO file 5" << endl;
	ofstream myfileGObin5("allPCRaster_5_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGObin5.write((char*) &allPCRaster5[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin5.close();
	delete2DArray<ct_uint8_t>(allPCRaster5);

	cout << "Filling GO file 6" << endl;
	ofstream myfileGObin6("allPCRaster_6_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGObin6.write((char*) &allPCRaster6[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin6.close();
	delete2DArray<ct_uint8_t>(allPCRaster6);

	cout << "Filling GO file 7" << endl;
	ofstream myfileGObin7("allPCRaster_7_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGObin7.write((char*) &allPCRaster7[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin7.close();
	delete2DArray<ct_uint8_t>(allPCRaster7);
	
	cout << "Filling GO file 8" << endl;
	ofstream myfileGObin8("allPCRaster_8_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGObin8.write((char*) &allPCRaster8[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin8.close();
	delete2DArray<ct_uint8_t>(allPCRaster8);

	cout << "Filling GO file 9" << endl;
	ofstream myfileGObin9("allPCRaster_9_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGObin9.write((char*) &allPCRaster9[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin9.close();
	delete2DArray<ct_uint8_t>(allPCRaster9);

	cout << "Filling GO file 10" << endl;
	ofstream myfileGObin10("allPCRaster_10_withColl_conv4000_ISI"+to_string(csSize)+"_coup0.15_gogo"+to_string(numCon[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<collectionTrials*(csSize+msPreCS+msPostCS); j++){
			myfileGObin10.write((char*) &allPCRaster10[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGObin10.close();
	delete2DArray<ct_uint8_t>(allPCRaster10);

*/




 /*       cout << "Filling PC files" << endl;
	ofstream myfilePCbin("allPCRaster_recip100_nocoll_ISI"+to_string(csSize)+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numPC; i++){
		for(int j=0; j<(collectionTrials)*(csSize+msPreCS+msPostCS); j++){
			myfilePCbin.write((char*) &allPCRaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfilePCbin.close();
	delete2DArray<ct_uint8_t>(allPCRaster);
*/	
	/*
	cout << "Filling NC files" << endl;
	ofstream myfileNCbin("allNCRaster_noGOGO_coll0.02_ISI"+to_string(csSize)+"_Param"+to_string(inputWeight_two)+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numNC; i++){
		for(int j=0; j<(numTrials-preTrialNumber)*(csSize+msPreCS+msPostCS); j++){
			myfileNCbin.write((char*) &allNCRaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileNCbin.close();
	delete2DArray<ct_uint8_t>(allNCRaster);
*/
	/*
	cout << "Filling IO files" << endl;
	ofstream myfileIObin("allIORaster_IOEQ1.2_IOThresh59_"+to_string(csSize)+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numIO; i++){
		for(int j=0; j<(numTrials-preTrialNumber)*(5000); j++){
			myfileIObin.write((char*) &allIORaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileIObin.close();
	delete2DArray<ct_uint8_t>(allIORaster);
*/
	cout << "Filling BC files" << endl;
	ofstream myfileBCbin("allBCRaster_paramSet"+to_string(inputStrength)+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numBC; i++){
		for(int j=0; j<(numTrials-preTrialNumber)*(csSize+msPreCS+msPostCS); j++){
			myfileBCbin.write((char*) &allBCRaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileBCbin.close();
	delete2DArray<ct_uint8_t>(allBCRaster);

	cout << "Filling SC files" << endl;
	ofstream myfileSCbin("allSCRaster_paramSet"+to_string(inputStrength)+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numSC; i++){
		for(int j=0; j<(numTrials-preTrialNumber)*(csSize+msPreCS+msPostCS); j++){
			myfileSCbin.write((char*) &allSCRaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileSCbin.close();
	delete2DArray<ct_uint8_t>(allSCRaster);
	
/*	cout << "Filling GO Raster" << endl;
	ofstream myfileSCbin("allGORaster_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<100*(csSize+msPreCS+msPostCS); j++){
			myfileSCbin.write((char*) &allGORaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileSCbin.close();
	cout << "GO Raster Filled" << endl;
	delete2DArray<ct_uint8_t>(allGORaster);
	cout << "GO Raster Deleted" << endl;
*/
	
	
	
//	cout << "Filling GO PSTH" << endl;
	//ofstream myfileGOpsthbin("allGOPSTH_recip100_conv"+to_string(conv[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
/*	ofstream myfileGOpsthbin("allGOPSTH_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS); j++){
			myfileGOpsthbin.write((char*) &allGOPSTH[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGOpsthbin.close();
	delete2DArray<ct_uint8_t>(allGOPSTH);
	
	ofstream myfileGRGOpsthbin("allGRGOPSTH_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS); j++){
			myfileGRGOpsthbin.write((char*) &grGOgPSTH[i][j], sizeof(float));
		}
	}
	myfileGRGOpsthbin.close();
	delete2DArray<float>(grGOgPSTH);*/

/*
	ofstream myfileGOrasterbin("allGORaster_recip100_conv"+to_string(conv[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS)*(collectionTrials); j++){
			myfileGOrasterbin.write((char*) &allGORaster[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGOrasterbin.close();
	delete2DArray<ct_uint8_t>(allGORaster);


	
 	
	
	
	ofstream myfileMFGORasterbin("allMFGORaster__recip100_conv"+to_string(conv[goRecipParam])+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS)*collectionTrials; j++){
			myfileMFGORasterbin.write((char*) &mfGOgRaster[i][j], sizeof(float));
		}
	}
	myfileMFGORasterbin.close();
	delete2DArray<float>(mfGOgRaster);
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	ofstream fileGRNumber;
	fileGRNumber.open("activeGRNumber.txt");	
	fileGRNumber << activeCounter << endl;
	fileGRNumber.close();
*/
/*	cout << "Filling GR PSTH" << endl;
	ofstream myfileGRpsthbin("allGRPSTH_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<grSaveNumber; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS); j++){
			myfileGRpsthbin.write((char*) &activeGRPSTH[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGRpsthbin.close();
	delete2DArray<ct_uint8_t>(activeGRPSTH);
*/
 /* 	
	ofstream myfileMFpsthbin("allMFPSTH.bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS); j++){
			myfileMFpsthbin.write((char*) &allMFPSTH[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileMFpsthbin.close();
	delete2DArray<ct_uint8_t>(allMFPSTH);
*/


/*
	ofstream myfileGRGOpsthbin("allGRGOPSTH_bg1030_GOGR0.025_recip100_10Con_gogoW"+to_string(gogoW)+"_Wscale"+to_string(goRecipParam)+"_GJ"+to_string(gjName[fileNum])+"_conv4000_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	//ofstream myfileGRGOpsthbin("allGRGOPSTH_recip100_10Con_gogoW"+to_string(gogoW)+"_Wscale"+to_string(goRecipParam)+"_GJ"+to_string(gjName[fileNum])+"_conv5000_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS); j++){
			myfileGRGOpsthbin.write((char*) &grGOconductancePSTH[i][j], sizeof(float));
		}
	}
	myfileGRGOpsthbin.close();
	delete2DArray<float>(grGOconductancePSTH);
*/
	/*
	ofstream myfileMFGOpsthbin("allMFGOPSTH_conv5000_coup075_min"+to_string(goMin)+"_"+to_string(simNum)+".bin", ios::out | ios::binary);	
	for(int i=0; i<numGO; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS); j++){
			myfileMFGOpsthbin.write((char*) &mfGOconductancePSTH[i][j], sizeof(float));
		}
	}
	myfileMFGOpsthbin.close();
	delete2DArray<float>(mfGOconductancePSTH);
*/




/*
	cout << "Filling GR files" << endl;
	ofstream myfileGRpsthbin("activeGRPSTH.bin", ios::out | ios::binary);	
	for(int i=0; i<activeCounter; i++){
		for(int j=0; j<(csSize+msPreCS+msPostCS); j++){
			myfileGRpsthbin.write((char*) &activeGRPSTH[i][j], sizeof(ct_uint8_t));
		}
	}
	myfileGRpsthbin.close();
	delete2DArray<ct_uint8_t>(activeGRPSTH);
	

	ofstream fileGRNumber;
	fileGRNumber.open("activeGRNumber.txt");	
	fileGRNumber << activeCounter << endl;
	fileGRNumber.close();

	delete [] granuleIndWithInput;  
	delete [] activeGranuleIndex;
	delete [] grPSTHCS;		
	delete [] grPSTHPreCS;
	delete2DArray<ct_uint8_t>(preGRPSTHCS);
	delete2DArray<ct_uint8_t>(preGRPSTHPreCS);
	delete grSpks;*/
}



int* Control::getGRIndicies(float CStonicMFfrac) 
{

	int numMF = 4096;
	int numGR = 1048576;


	//float CStonicMFfrac = 0.05;
	float CSphasicMFfrac = 0.0;
	float contextMFfrac = 0.00;
	
	bool* contextMFs = joeMFFreq->getContextMFInd();
	bool* phasicMFs = joeMFFreq->getPhasicMFInd();
	bool* tonicMFsA = joeMFFreq->getTonicMFInd();
	bool* tonicMFsB = joeMFFreq->getTonicMFIndOverlap();
	
	
	int numContext = numMF*contextMFfrac; 
	int numPhasic = numMF*CSphasicMFfrac; 
	int numTonic = numMF*CStonicMFfrac; 
	int numActiveMFs = numTonic;

	cout << "Number of CS MossyFibers:	" << numActiveMFs << endl;
	
	int *activeMFIndA;
	activeMFIndA = new int[numActiveMFs];
	int *activeMFIndB;
	activeMFIndB = new int[numActiveMFs];
	
	int counterMFA=0;
	int counterMFB=0;
	for(int i=0; i<numMF; i++)
	{	
		if(tonicMFsA[i])
		{
			activeMFIndA[ counterMFA ] = i;
			counterMFA++;
		}
		if(tonicMFsB[i])
		{
			activeMFIndB[ counterMFB ] = i;
			counterMFB++;
		}
	}
	cout << "NumMFs in A:	" << counterMFA << endl;
	cout << "NumMFs in B:	" << counterMFB << endl;
	
/*	ofstream fileActMFA;
	fileActMFA.open("mfTonicCSA.txt");
	for(int i=0; i<counterMFA; i++){
		fileActMFA << activeMFIndA[i] << endl;
	}
	ofstream fileActMFB;
	fileActMFB.open("mfTonicCSB.txt");
	for(int i=0; i<counterMFB; i++){
		fileActMFB << activeMFIndB[i] << endl;
	}
*/
	vector<int> MFtoGRs;	
	int numPostSynGRs;
	int *pActiveGRsBool;
	pActiveGRsBool = new int[numGR];
	for(int i=0; i<numGR; i++){ pActiveGRsBool[i]=0;}

	for(int i=0; i<numActiveMFs; i++)
	{
		MFtoGRs = joestate->getInnetConStateInternal()->getpMFfromMFtoGRCon( activeMFIndA[i] );
		numPostSynGRs = MFtoGRs.size();
		
		for(int j=0; j<numPostSynGRs; j++)
		{
			pActiveGRsBool[ MFtoGRs[j] ] = pActiveGRsBool[ MFtoGRs[j] ] + 1;
		}
	
	}
	
	int counterGR = 0;
	for(int i=0; i<numGR; i++)
	{
		if(pActiveGRsBool[i] >= 1)
		{		
			counterGR++;
		}
	}

	int *pActiveGRs;
	pActiveGRs = new int[counterGR];
	for(int i=0; i<counterGR; i++){ pActiveGRs[i]=0;}

	int counterAGR=0;
	for(int i=0; i<numGR; i++)
	{
		if(pActiveGRsBool[i] >= 1)
		{		
			pActiveGRs[counterAGR] = i;
			counterAGR++;
		}
		
	}	

	return pActiveGRs;

}


int Control::getNumGRIndicies(float CStonicMFfrac) 
{
	int numMF = 4096;
	int numGR = 1048576;


	//float CStonicMFfrac = 0.05;
	float CSphasicMFfrac = 0.0;
	float contextMFfrac = 0.0;
	
	bool* contextMFs = joeMFFreq->getContextMFInd();
	bool* phasicMFs = joeMFFreq->getPhasicMFInd();
	bool* tonicMFs = joeMFFreq->getTonicMFInd();
	
	int numContext = numMF*contextMFfrac; 
	int numPhasic = numMF*CSphasicMFfrac; 
	int numTonic = numMF*CStonicMFfrac; 
	int numActiveMFs = numContext+numPhasic+numTonic;
	
	int *activeMFInd;
	activeMFInd = new int[numActiveMFs];
	
	int counterMF=0;
	for(int i=0; i<numMF; i++)
	{	
		if(contextMFs[i] || tonicMFs[i] || phasicMFs[i])
		{
			activeMFInd[ counterMF ] = i;
			counterMF++;
		}
	}

	vector<int> MFtoGRs;	
	int numPostSynGRs;
	int *pActiveGRsBool;
	pActiveGRsBool = new int[numGR];
	for(int i=0; i<numGR; i++){ pActiveGRsBool[i]=0;}


	for(int i=0; i<numActiveMFs; i++)
	{
		MFtoGRs = joestate->getInnetConStateInternal()->getpMFfromMFtoGRCon( activeMFInd[i] );
		numPostSynGRs = MFtoGRs.size();
		
		for(int j=0; j<numPostSynGRs; j++)
		{
			pActiveGRsBool[ MFtoGRs[j] ] = pActiveGRsBool[ MFtoGRs[j] ] + 1;
		}
	}
	
	int counterGR = 0;
	for(int i=0; i<numGR; i++)
	{
		if(pActiveGRsBool[i] >= 1)
		{		
			counterGR++;
		}
	}

	return counterGR;

}




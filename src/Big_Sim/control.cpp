#include <string>
#include <time.h>
#include "control.h"

//using namespace std;

//Control::Control(){};

//Control::~Control(){};

void Control::runSimulationWithGRdata(int fileNum, int goRecipParam, int numTuningTrials,
		int numGrDetectionTrials, int numTrainingTrials, int simNum, int csSize, float csFracMFs,
		float goMin, float GOGR, float GRGO, float MFGO, float csMinRate, float csMaxRate,
		float gogoW, int inputStrength, int inputWeight_two, float spillFrac)
{

	//TODO: Need to create separate logger
	std::cout << "fileNum: " << fileNum << std::endl;
	SetSim simulation(fileNum, goRecipParam, simNum);
	joestate = simulation.getstate();
	joesim = simulation.getsim();
	joeMFFreq = simulation.getMFFreq(csMinRate, csMaxRate);
	joeMFs = simulation.getMFs();	

	int numTotalTrials = numTuningTrials + numGrDetectionTrials + numTrainingTrials;  
	int preTrialNumber = numTuningTrials + numGrDetectionTrials;
	int collectionTrials = numTotalTrials;//numTrials - preTrialNumber;
	
	std::cout << "Done filling MF arrays" << std::endl;	

	int recipName[25] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50};
	int conv[8] = {5000, 4000, 3000, 2000, 1000, 500, 250, 125};

	// Allocate and Initialize PSTH and Raster arrays
	allPCRaster = allocate2DArray<ct_uint8_t>(numPC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allPCRaster[0], 0, numPC*(csSize+msPreCS+msPostCS)*(collectionTrials));
	
	allNCRaster = allocate2DArray<ct_uint8_t>(numNC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allNCRaster[0], 0, numNC*(csSize+msPreCS+msPostCS)*(collectionTrials));

	allSCRaster = allocate2DArray<ct_uint8_t>(numSC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allSCRaster[0], 0, numSC*(csSize+msPreCS+msPostCS)*(collectionTrials));

	allBCRaster = allocate2DArray<ct_uint8_t>(numBC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allBCRaster[0], 0, numBC*(csSize+msPreCS+msPostCS)*(collectionTrials));
	
	std::cout << "PC arrays" << std::endl;	

	allGOPSTH = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS));	
	arrayInitialize<ct_uint8_t>(allGOPSTH[0], 0, numGO*(csSize+msPreCS+msPostCS));

	float medTrials;
	float *mTall    = new float[numTotalTrials];
	float *grgoGall = new float[numTotalTrials];
	float *mfgoGall = new float[numTotalTrials];
	
	clock_t timer;
	int rasterCounter = 0;
	int rasterCounterIO = 0;
	int grSpkCounter = 0;
	int grSpkCounterPre = 0;
	
	
	std::vector<int> goSpkCounter;
	goSpkCounter.assign(numGO, 0);
	
	float r;
	int tsCSCounter = 0;
	for (int trial = 0; trial < numTotalTrials; trial++)
	{
		timer = clock();

		for(int i=0; i < numGO; i++)
		{	
			goSpkCounter[i] = 0;
		}

		float gGRGO_sum = 0;
		float gMFGO_sum = 0;

		if (trial <= numTuningTrials)
	   	{
			std::cout << "Pre-tuning trial number: " << trial << std::endl;
			trialTime = 5000;	
		}

		if (trial > numTuningTrials)
		{
			std::cout << "Post-tuning trial number: " << trial << std::endl;
			trialTime = 5000;	
		}

		int PSTHCounter = 0;	
		int grPSTHCounter = 0;	
		int preCounterGRCS = 0;
		int preCounterGRPre = 0;

		// Homeostatic plasticity trials

		if (trial >= numTuningTrials)
		{
			// Run active granule cell detection 	
			if (trial == preTrialNumber)
			{			
				int spk;
				int binCountPre;
				int binCountCS;

				for (tts = 0 ; tts < trialTime; tts++)
					
					if (tts == csSize + csStart)
					{
						// Deliver US 
						joesim->updateErrDrive(0,0.0);
					}
					
					if (tts < csStart || tts >= csStart + csSize)
					{
						// Background MF activity in the Pre and Post CS period
						mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFBG(), joesim->getMZoneList());	
					}
					else if (tts >= csStart && tts < csStart + csPhasicSize) 
					{
						// Phasic MF activity during the CS for a duration set in control.h 
						mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFFreqInCSPhasic(), joesim->getMZoneList());
					}
					else
					{
						// Tonic MF activity during the CS period
						mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFInCSTonicA(), joesim->getMZoneList());
					}

					bool *isTrueMF = joeMFs->calcTrueMFs(joeMFFreq->getMFBG());
					joesim->updateTrueMFs(isTrueMF);
					joesim->updateMFInput(mfAP);
					joesim->calcActivity(goMin, simNum, GOGR, GRGO, MFGO, gogoW, spillFrac);	
					
					if(tts >= csStart && tts < csStart + csSize)
					{

						mfgoG = joesim->getInputNet()->exportgSum_MFGO();
						grgoG = joesim->getInputNet()->exportgSum_GRGO();
						goSpks = joesim->getInputNet()->exportAPGO();
						for (int i = 0; i < numGO; i++)
						{
								goSpkCounter[i] += goSpks[i];
								gGRGO_sum += grgoG[i];
								gMFGO_sum += mfgoG[i];
						}

					}
					
					if(tts == csStart + csSize)
					{
						int n = sizeof(goSpkCounter) / sizeof(goSpkCounter[0]);
						
						std::sort(goSpkCounter.begin(), goSpkCounter.begin()+4096);
						
						int m = (goSpkCounter[2047] + goSpkCounter[2048])/2.0;
						float goSpkSum = 0;
						
						for(int i = 0; i < numGO; i++)
						{
								goSpkSum += goSpkCounter[i];
						}
						
						std::cout << "Mean GO Rate: " << goSpkSum / (float)numGO << std::endl;

						medTrials += m / 2.0;
						std::cout << "Median GO Rate: " << m / 2.0 << std::endl;

						mTall[trial] = m / 2.0;
						std::cout << "mean gGRGO   = " << gGRGO_sum / (numGO * csSize) << std::endl;
						std::cout << "mean gMFGO   = " << gMFGO_sum / (numGO * csSize) << std::endl;
						std::cout << "GR:MF ratio  = " << gGRGO_sum / gMFGO_sum << std::endl;

						grgoGall[trial] = gGRGO_sum / (numGO * csSize);
						mfgoGall[trial] = gMFGO_sum / (numGO * csSize);

					}
					
					if (trial >= preTrialNumber && tts >= csStart-msPreCS && tts< csStart + csSize + msPostCS)
					{
						
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
	
	for(int i=0; i<numGR; i++)
	{ 
		pActiveGRsBool[i]=0;
	}


	for(int i=0; i<numActiveMFs; i++)
	{
		MFtoGRs = joestate->getInnetConStateInternal()->getpMFfromMFtoGRCon(activeMFInd[i]);
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




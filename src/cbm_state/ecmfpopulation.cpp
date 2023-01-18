/*
 * ecmfpopulation.cpp
 *
 *  Created on: Jul 13, 2014
 *      Author: consciousness
 */

#include <random>

#include "logger.h"
#include "ecmfpopulation.h"

ECMFPopulation::ECMFPopulation(int numMF, int randSeed, float fracCSTMF, float fracCSPMF,
	float fracCtxtMF, float fracCollNC, float bgFreqMin, float csBGFreqMin, float ctxtFreqMin,
	float csTFreqMin, float csPFreqMin, float bgFreqMax, float csBGFreqMax, float ctxtFreqMax,
	float csTFreqMax, float csPFreqMax, bool turnOffColls,  float fracImportMF, bool secondCS,
	float fracOverlap)
{
	CRandomSFMT0 randGen(randSeed);

    int numCSTMF;
    int numCSTMFA;
    int numCSTMFB;
    int numCSPMF;
    int numCtxtMF;
	int numCollNC;
	int numImportMF;

	this->turnOffColls = turnOffColls;
	this->numMF = numMF;

	mfFreqBG         = new float[numMF];
	mfFreqInCSPhasic = new float[numMF];
	mfFreqInCSTonicA = new float[numMF];
	mfFreqInCSTonicB = new float[numMF];

	isCSTonicA 	 = new bool[numMF];
	isCSTonicB 	 = new bool[numMF];
	isCSPhasic 	 = new bool[numMF];
	isContext 	 = new bool[numMF];
	isCollateral = new bool[numMF];
	isImport 	 = new bool[numMF];
	isAny 		 = new bool[numMF];

	for (int i = 0; i < numMF; i++)
	{
		mfFreqBG[i] = randGen.Random() * (bgFreqMax - bgFreqMin) + bgFreqMin;
		
		mfFreqInCSTonicA[i] = mfFreqBG[i];
		mfFreqInCSTonicB[i] = mfFreqBG[i];
		mfFreqInCSPhasic[i] = mfFreqBG[i];

        isCSTonicA[i]   = false;
        isCSTonicB[i]   = false;
        isCSPhasic[i]   = false;
        isContext[i] 	= false;
		isCollateral[i] = false;
		isImport[i] 	= false;
		isAny[i] 		= false;
	}

	numCSTMF = fracCSTMF * numMF;
	numCSTMFA = numCSTMF;

	if (secondCS) numCSTMFB = numCSTMF; 
	else numCSTMFB = 0;
    	
	numCSPMF    = fracCSPMF * numMF;
	numCtxtMF   = fracCtxtMF * numMF;
	numCollNC   = fracCollNC * numMF;
	numImportMF = fracImportMF * numMF;

	//Set up Mossy Fibers. Order is important for Rand num generation.
	setMFs(numCollNC, numMF, randGen, isAny, isCollateral);
	setMFs(numImportMF, numMF, randGen, isAny, isImport);
	
	//Order below is important for competing stimulus experiments.
	//Pick MFs for Tonic A
	setMFs(numCSTMFA, numMF, randGen, isAny, isCSTonicA);
	
	//Pick remaining MFs that would have been in full Tonic B so that 
	//random seed leaves off at same place
	setMFs(numCSTMFA, numMF, randGen, isAny, isAny);

	//Pick Tonic B MFs
	// Ensure remaining MFs that would have been in full Tonic A are not picked
	setMFs(numCSTMFB, numMF, randGen, isAny, isAny);

	int phasicSum  	   =  0;
	int contextSum     =  0;
	int collateralSum  =  0;
	int tonicASum  	   =  0;
	int tonicBSum  	   =  0;
	int importSum  	   =  0;
	int anySum  	   =  0;

	for ( int i  =  0; i < numMF; i++)
	{
		phasicSum 	  += isCSPhasic[i];
		contextSum 	  += isContext[i];
		collateralSum += isCollateral[i];
		tonicASum 	  += isCSTonicA[i];
		tonicBSum 	  += isCSTonicB[i];
		importSum 	  += isImport[i];
		anySum 		  += isAny[i]; /* wouldn't this be the whole population? */
	}

    for (int i = 0; i < numMF; i++)
    {
        if (isContext[i])
        {
        	mfFreqBG[i] = randGen.Random() * (ctxtFreqMax - ctxtFreqMin) + 
				ctxtFreqMin;

			mfFreqInCSTonicA[i] = mfFreqBG[i];
			mfFreqInCSTonicB[i] = mfFreqBG[i];
			mfFreqInCSPhasic[i] = mfFreqBG[i];

			randGen.Random();
        }
		else if (isCSPhasic[i])
        {
			mfFreqBG[i] = randGen.Random() * (csBGFreqMax - csBGFreqMin) + csBGFreqMin;

			mfFreqInCSTonicA[i] = mfFreqBG[i];
			mfFreqInCSTonicB[i] = mfFreqBG[i];

			mfFreqInCSPhasic[i] = randGen.Random()*(csPFreqMax - csPFreqMin) + csPFreqMin;
        }
		else if (isCollateral[i] && !turnOffColls) /* confusing: is an afferent to the dcn. why treat collaterals as recurrent dcn->mf feedback? */
		{
			mfFreqBG[i] 		= -1;
			mfFreqInCSTonicA[i] = -1;
			mfFreqInCSTonicB[i] = -1;
			mfFreqInCSPhasic[i] = -1;
			// NOTE: that we were choosing random twice before.
			// any reason why?
			randGen.Random(); 
        } 
		else if (isImport[i])
		{
			mfFreqBG[i] = randGen.Random() * (csBGFreqMax - csBGFreqMin) + csBGFreqMin;
			mfFreqInCSTonicA[i] = -2;
			mfFreqInCSTonicB[i] = -2;
			mfFreqInCSPhasic[i] = -2;

			randGen.Random();
		} 
		else if (isCSTonicA[i])
		{
            mfFreqBG[i] = randGen.Random() * (csBGFreqMax - csBGFreqMin) + csBGFreqMin;

            mfFreqInCSTonicA[i] = randGen.Random() * (csTFreqMax - csTFreqMin) + csTFreqMin;
            mfFreqInCSPhasic[i] = mfFreqInCSTonicA[i];
        } 
		else if (isCSTonicB[i])
        {
            mfFreqBG[i] = randGen.Random() * (csBGFreqMax - csBGFreqMin) + csBGFreqMin;

            mfFreqInCSTonicB[i] = randGen.Random() * (csTFreqMax - csTFreqMin) + csTFreqMin;
            mfFreqInCSPhasic[i] = mfFreqInCSTonicB[i];
        } 
		else
		{
			//NOTE: again, two random generations. 
			//Note sure why.
			randGen.Random(); 
		}
    }
}

ECMFPopulation::~ECMFPopulation()
{
	delete[] mfFreqBG;
	delete[] mfFreqInCSPhasic;
	delete[] mfFreqInCSTonicA;
	delete[] mfFreqInCSTonicB;

    delete[] isCSTonicA;
    delete[] isCSTonicB;
    delete[] isCSPhasic;
    delete[] isContext;
	delete[] isCollateral;
	delete[] isImport;
	delete[] isAny;
}

void ECMFPopulation::setMFs(int numTypeMF, int numMF, CRandomSFMT0 &randGen, bool *isAny, bool *isType)
{
    for (int i = 0; i < numTypeMF; i++)
    {
        while(true)
        {
            int mfInd = randGen.IRandom(0, numMF-1);

            if (!isAny[mfInd])
            {
				isAny[mfInd]  = true;
            	isType[mfInd] = true;
            	break;
			}
        }
    }
}

void ECMFPopulation::setMFsOverlap(int numTypeMF, int numMF, CRandomSFMT0 &randGen, bool *isAny, bool *isTypeA, bool *isTypeB, float fracOverlap)
{
    
	//Get population sizes
	int numOverlapMF 	 = numTypeMF*fracOverlap;
	int numIndependentMF = numTypeMF - numOverlapMF;
	LOG_DEBUG("NumOverlap: %d", numOverlapMF);

	//Select overlaping population
	int counter = 0;
	
	for(int i = 0; i < numMF; i++)
	{
		if(isTypeA[i] && counter < numOverlapMF)
		{
			isTypeB[i] = true;
			isAny[i]   = true;
			counter++;
		}
	}
	//
	//Select non-overlaping population
	for (int i = 0; i < numIndependentMF; i++)
	{
		while(true)
		{
			int mfInd = randGen.IRandom(0, numMF-1);

			if(isAny[mfInd]) continue;
		
			isAny[mfInd]   = true;
			isTypeB[mfInd] = true;
			break;
		}
   	}
}

void ECMFPopulation::writeMFLabels(std::string labelFileName)
{
	LOG_DEBUG("Writing MF labels...");
	std::fstream mflabels(labelFileName.c_str(), std::fstream::out);

	for (int i = 0; i < numMF; i++)
	{
		if(isContext[i])
	   	{
			mflabels << "con ";
		} 
		else if(isCSPhasic[i])
	   	{
			mflabels << "pha ";
		} 
		else if(isCollateral[i] && !turnOffColls)
	   	{
			mflabels << "col ";
		} 
		else if(isImport[i]) 
		{
			mflabels << "imp ";
		} 
		else if(isCSTonicA[i] || isCSTonicB[i]) 
		{
			mflabels << "ton ";
		} 
		else 
		{
			mflabels << "bac ";
		}
	}
	mflabels.close();
	LOG_DEBUG("MF labels written.");
}

void ECMFPopulation::writeToFile(std::fstream &outfile)
{
	outfile.write((char *)&numMF, sizeof(numMF));
	outfile.write((char *)mfFreqBG, numMF*sizeof(float));
	outfile.write((char *)mfFreqInCSTonicA, numMF*sizeof(float));
	outfile.write((char *)mfFreqInCSTonicB, numMF*sizeof(float));
	outfile.write((char *)mfFreqInCSPhasic, numMF*sizeof(float));
}

bool *ECMFPopulation::getContextMFInd()
{
	return isContext; 
}

bool *ECMFPopulation::getTonicMFInd()
{
	return isCSTonicA;
}

bool *ECMFPopulation::getTonicMFIndOverlap()
{
	return isCSTonicB;
}

bool *ECMFPopulation::getPhasicMFInd()
{
	return isCSPhasic;
}

float *ECMFPopulation::getMFBG()
{
	return mfFreqBG;
}

float *ECMFPopulation::getMFInCSTonicA()
{
	return mfFreqInCSTonicA;
}

float *ECMFPopulation::getMFInCSTonicB()
{
	return mfFreqInCSTonicB;
}

float *ECMFPopulation::getMFFreqInCSPhasic()
{
	return mfFreqInCSPhasic;
}


/*
 * innetconstateggialtcon.cpp
 *
 *  Created on: Nov 26, 2013
 *      Author: consciousness
 */

#include "state/innetconstateggialtcon.h"

using namespace std;

InNetConStateGGIAltCon::InNetConStateGGIAltCon(ConnectivityParams *parameters,
		unsigned int msPerStep, int randSeed)
	:InNetConnectivityState(parameters, msPerStep, randSeed, 1, 1)
{
	CRandomSFMT *randGen;
	randGen=new CRandomSFMT0(randSeed+749710656);
	connectGOGO(randGen);

	delete randGen;
}

void InNetConStateGGIAltCon::connectGOGO(CRandomSFMT *randGen)
{
	float numGABAIn;
	numGABAIn=0;

	for(int j=0; j<cp->maxnumpGOGABAInGOGO; j++)
	{
		numGABAIn+=cp->gogoGABALocalCon[j][2];
	}
	numGABAIn=floor(numGABAIn);

	if(numGABAIn>cp->maxnumpGOGABAInGOGO)
	{
		numGABAIn=cp->maxnumpGOGABAInGOGO;
	}

	cout<<numGABAIn<<endl;

	for(int i=0; i<cp->numGO; i++)
	{
		numpGOGABAOutGOGO[i]=0;
	}

	for(int i=0; i<cp->numGO; i++)
	{
		int currPosX;
		int currPosY;
		vector<int> positions;

		for(int j=0; j<numGABAIn; j++)
		{
			while(true)
			{
				int conPos;
				bool hasCon;
				conPos=randGen->IRandom(0, cp->maxnumpGOGABAInGOGO-1);

				hasCon=false;
				for(int k=0; k<positions.size(); k++)
				{
					if(conPos==positions[k])
					{
						hasCon=true;
					}
				}
				if(hasCon)
				{
					continue;
				}

				positions.push_back(conPos);
				break;
			}
		}

		currPosX=i%cp->goX;
		currPosY=i/cp->goX;
		numpGOGABAInGOGO[i]=positions.size();

		for(int j=0; j<positions.size(); j++)
		{
			int srcPosX;
			int srcPosY;
			int srcInd;
			srcPosX=currPosX+cp->gogoGABALocalCon[positions[j]][0];
			srcPosX=(srcPosX%cp->goX+cp->goX)%cp->goX;

			srcPosY=currPosY+cp->gogoGABALocalCon[positions[j]][1];
			srcPosY=(srcPosY%cp->goY+cp->goY)%cp->goY;

			srcInd=cp->goX*srcPosY+srcPosX;

			pGOGABAInGOGO[i][j]=srcInd;

			pGOGABAOutGOGO[srcInd][numpGOGABAOutGOGO[srcInd]]=i;
			numpGOGABAOutGOGO[srcInd]++;
		}
	}
}

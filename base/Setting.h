#ifndef SETTING_H
#define SETTING_H
#define INT long
#define REAL float
#include <cstring>
#include <cstdio>
#include <string>


std::string inPath = "../data/FB15K/";
std::string outPath = "../data/FB15K/";
INT workThreads = 1;
INT entityTotal = 0;
INT relationTotal = 0;
INT tripleTotal = 0;
INT trainTotal = 0;
INT testTotal = 0;
INT validTotal = 0;


extern "C"
void setInPath(
		char* path)
{
	INT len = strlen(path);
	inPath = "";
	for (INT i = 0; i < len; i++)
		inPath = inPath + path[i];
	printf("Input Files Path : %s\n", inPath.c_str());
}


extern "C"
void setOutPath(
		char* path)
{
	INT len = strlen(path);
	outPath = "";
	for (INT i = 0; i < len; i++)
		outPath = outPath + path[i];
	printf("Output Files Path : %s\n", outPath.c_str());
}


extern "C"
INT getEntityTotal()
{
	return entityTotal;
}


extern "C"
INT getRelationTotal()
{
	return relationTotal;
}


extern "C"
INT getTripleTotal()
{
	return tripleTotal;
}


extern "C"
INT getTrainTotal()
{
	return trainTotal;
}


extern "C"
INT getTestTotal()
{
	return testTotal;
}


extern "C"
INT getValidTotal()
{
	return validTotal;
}
#endif

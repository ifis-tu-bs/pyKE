#include <fstream> // std::ofstream
#include <iostream> // std::cout std::endl
#include <cstdio> // std::remove
#include <dlfcn.h> // dlopen dlsym dlerror
#include <cstdint> // int64_t
#define TEMPNAME "temp.temp.temp"


typedef int (*im_f)(const char*, int64_t, int64_t);
typedef int (*qh_f)(char*, int64_t, int64_t);


int main(int count, char* arguments[])
{

	void* dl = dlopen("../libopenke.so", RTLD_NOW bitor RTLD_LOCAL);
	if (!dl)
	{
		std::cout << dlerror() << std::endl;
		return 1;
	}

	void* sym_import = dlsym(dl, "importTrainFiles");
	if (!sym_import)
	{
		std::cout << dlerror() << std::endl;
		dlclose(dl);
		return 2;
	}
	void* sym_queryhead = dlsym(dl, "query_head");
	if (!sym_queryhead)
	{
		std::cout << dlerror() << std::endl;
		dlclose(dl);
		return 3;
	}

	im_f importTrainFiles = *((im_f*)&sym_import);
	qh_f query_head = *((qh_f*)&sym_queryhead);

	std::ofstream(TEMPNAME) << "2\n0 0 0\n0 1 0" << std::endl;
	if (importTrainFiles(TEMPNAME, 2, 1))
	{
		std::cout << "import of train files failed" << std::endl;
		std::remove(TEMPNAME);
		dlclose(dl);
		return 4;
	}

	char result[2];
	for (int i = 0; i < 2; ++i)
		result[i] = 0;
	query_head(result, 0, 0);
	for (int i = 0; i < 2; ++i)
		std::cout << (result[i] ? "true" : "false") << std::endl;

	std::remove(TEMPNAME);
	dlclose(dl);
	return 0;
}

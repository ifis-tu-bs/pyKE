#ifndef TRIPLE_H
#	define TRIPLE_H
#	include "Setting.h"
#	include <algorithm> // std::min


struct Triple
{


	INT h;
	INT r;
	INT t;

	
	static bool cmp_list(
			const Triple& a,
			const Triple& b)
	{
		return std::min(a.h, a.t) > std::min(b.h, b.t);
	}


	static bool cmp_hrt(
			const Triple& a,
			const Triple& b)
	{
		return (a.h < b.h) or (a.h == b.h and a.r < b.r)
				or (a.h == b.h and a.r == b.r and a.t < b.t);
	}


	static bool cmp_trh(
			const Triple& a,
			const Triple& b)
	{
		return (a.t < b.t) or (a.t == b.t and a.r < b.r)
				or (a.t == b.t and a.r == b.r and a.h < b.h);
	}


	static bool cmp_rht(
			const Triple& a,
			const Triple& b)
	{
		return (a.h < b.h) or (a.h == b.h and a.t < b.t)
				or (a.h == b.h and a.t == b.t and a.r < b.r);
	}


};
#endif // TRIPLE_H

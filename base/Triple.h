#ifndef TRIPLE_H
#	define TRIPLE_H
#	include <algorithm> // std::min
#	include <cstdint> // int64_t


using ent_id = int64_t;
using rel_id = int64_t;


struct Triple
{


	ent_id h;
	ent_id t;
	rel_id r;

	
	bool operator<(
			const Triple& other)
	{
		return std::min(h, t) > std::min(other.h, other.t);
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


	static bool cmp_htr(
			const Triple& a,
			const Triple& b)
	{
		return (a.h < b.h) or (a.h == b.h and a.t < b.t)
				or (a.h == b.h and a.t == b.t and a.r < b.r);
	}


	static bool cmp_hr(
			const Triple& a,
			const Triple& b)
	{
		return (a.h < b.h) or (a.h == b.h and a.r < b.r);
	}


	static bool cmp_tr(
			const Triple& a,
			const Triple& b)
	{
		return (a.t < b.t) or (a.t == b.t and a.r < b.r);
	}


	static bool cmp_h(
			const Triple& a,
			const Triple& b)
	{
		return (a.h < b.h);
	}


	static bool cmp_t(
			const Triple& a,
			const Triple& b)
	{
		return (a.t < b.t);
	}


	static bool cmp_r(
			const Triple& a,
			const Triple& b)
	{
		return (a.r < b.r);
	}


	Triple(
			const ent_id& head,
			const ent_id& tail,
			const rel_id& rel):
		h{head}, t{tail}, r{rel}
	{
	}


	Triple(void):
		Triple(0, 0, 0)
	{
	}


};
#endif // TRIPLE_H

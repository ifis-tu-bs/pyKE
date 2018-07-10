#ifndef TRIPLE_H
#	define TRIPLE_H
#	include <algorithm> // std::min
#	include <cstdint> // int64_t


struct Triple
{
	int64_t h;
	int64_t t;
	int64_t r;

	static bool cmp_hrt(const Triple& a, const Triple& b) {
		return (a.h < b.h) or (a.h == b.h and a.r < b.r) or (a.h == b.h and a.r == b.r and a.t < b.t);
	}

	static bool cmp_trh(const Triple& a, const Triple& b) {
		return (a.t < b.t) or (a.t == b.t and a.r < b.r) or (a.t == b.t and a.r == b.r and a.h < b.h);
	}

	static bool cmp_htr(const Triple& a, const Triple& b) {
		return (a.h < b.h) or (a.h == b.h and a.t < b.t) or (a.h == b.h and a.t == b.t and a.r < b.r);
	}

	static bool cmp_h(const Triple& a, const Triple& b) {
		return (a.h < b.h);
	}

	static bool cmp_t(const Triple& a, const Triple& b) {
		return (a.t < b.t);
	}

	static bool cmp_r(const Triple& a, const Triple& b) {
		return (a.r < b.r);
	}

	Triple(const int64_t& head, const int64_t& tail, const int64_t& rel): h{head}, t{tail}, r{rel} {}

	Triple(void): Triple(0, 0, 0) {}
};
#endif // TRIPLE_H

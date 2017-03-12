#pragma once

// this is taken from http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring

#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>
#include <fstream>
#include <sstream>
#include <stdexcept>

// trim from start
static inline std::string ltrim(std::string s)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
	return std::move(s);
}

// trim from end
static inline std::string rtrim(std::string s)
{
	s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
	return std::move(s);
}

// trim from both ends
static inline std::string trim(std::string s)
{
	return std::move(ltrim(rtrim(s)));
}

static inline std::string read_file(std::string name)
{
	std::ifstream t(name);
	if (t.fail())
	{
		throw std::runtime_error("file does not exist");
	}
	std::stringstream buffer;
	buffer << t.rdbuf();
	return buffer.str();
}

template <typename T>
void fill(T* dest, int w, int h, const T value) {
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const int idx = y * w + x;
			dest[idx] = value;
		}
	}
}
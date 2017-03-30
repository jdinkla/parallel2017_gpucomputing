/*
 * Copyright (c) 2015, 2017 by J�rn Dinkla, www.dinkla.net, All rights reserved.
 *
 * See the LICENSE file in the root directory.
 */

#pragma once

#include "Defs.h"

#include <string>
#ifdef MAC
#define __CL_ENABLE_EXCEPTIONS
#include "cl_1_2.hpp"
#else
#include <cl/cl.hpp>
#endif

void info_platform(cl::Platform& p);

void info_platforms();

void info_device(cl::Device& d);

void info_devices();

void info_devices_detailed();

void info_all();

void info_tree();

cl::Platform get_platform_by_name(std::string name);

cl::Device get_device_by_name(cl::Platform& platform, std::string name);

std::vector<cl::Platform> get_platforms();

std::vector<cl::Device> get_devices(cl::Platform& platform);

cl::Context get_context(cl::Platform& platform, cl::Device& device);

cl::Program get_program(cl::Device& device, cl::Context& context, std::string code);

double duration_in_ms(cl::Event& event);
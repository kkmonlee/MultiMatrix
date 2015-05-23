#include "cudaTimer.h"
#include <cstdio>
#include <cstdlib>

#define TimerCall( err ) __TimerCall( err, __FILE__, __LINE__ )

inline void __TimerCall(cudaError err, const char *file, const int line) {
#pragma warning(push)
#pragma warning(disable:4127)
	do {
		if (cudaSuccess != err) {
			fprintf(stderr, "CUDA timer failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
			exit(-1);
		}
	} while (0);
#pragma warning(pop)
	return;
}

CudaTimer::CudaTimer() {
	TimerCall(cudaEventCreate(&_begEvent));
	TimerCall(cudaEventCreate(&_endEvent));
	return;
}

CudaTimer::~CudaTimer() {
	TimerCall(cudaEventDestroy(_begEvent));
	TimerCall(cudaEventDestroy(_endEvent));
	return;
}

void CudaTimer::start() {
	TimerCall(cudaEventRecord(_begEvent, 0));
	return;
}

void CudaTimer::stop() {
	TimerCall(cudaEventRecord(_endEvent, 0));
	return;
}

float CudaTimer::value() {
	TimerCall(cudaEventSynchronize(_endEvent));
	float time;

	TimerCall(cudaEventElapsedTime(&time, _begEvent, _endEvent));
	return time;
}
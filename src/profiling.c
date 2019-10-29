// Mini Profiling Interface for use
#ifdef __cplusplus
extern "C"{
#endif //__cplusplus
#include "../include/profiling.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdarg.h>
#include <unistd.h>


static timefunc wtime;
static double timer_user = 0, timer_backend = 0;
static double time_user =0, time_backend = 0;
static char hostname[32];
static int pid;
static int iteration;
static FILE* fp;
static char fn[256];
extern char* __progname;
static unsigned char isInited = 0;

void profile_init(
    timefunc_t tf
){
    gethostname(hostname, 32);
    pid = getpid();
    snprintf(fn, 256, "%s.%s.%d.csv", __progname, hostname, pid);
#ifdef MEASUREMENT
    fp = fopen(fn, "a+");
    if(fp == NULL){
        exit(199);
    }
#else
    fp = stdout;
#endif

    switch(tf){
        case REALTIME:
            wtime = realtime;
            break;
        case REALTIME_SPARSE:
            wtime = fast_realtime;
            break;
        case CPU_TIME:
            wtime = cputime;
            break;
    }

    isInited = 1;
}

void profile_close(
    void
){
#ifdef MEASUREMENT
    if(fp){
        fclose(fp);
    }
#endif

    isInited = 0;
}


void profile_set_iteration(
    int iter
){
    if(!isInited) return; 
    iteration = iter;
}

//Time Measurement Funcitonality
double realtime(){
    struct timeval tv;
    gettimeofday(&tv, 0);

    return tv.tv_sec+1e-6*tv.tv_usec;
}

// NOTE: See Discussion
// https://stackoverflow.com/questions/6498972/faster-equivalent-of-gettimeofday
// The CLOCK_MONOTONIC_COARSE is way faster but delivers a significant lower precision 
double fast_realtime(){
    #ifdef __APPLE__
    return realtime();
    #else
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &tv);
    return (double)tv.tv_sec+(double)1e-9*tv.tv_nsec;
    #endif
}

double cputime(){
    clock_t clk = clock();
    return (double)clk/CLOCKS_PER_SEC;
}

 void
profile_reset_user(
    void
){
    if(!isInited) return; 
    time_user = timer_user = 0;
}

 void
profile_reset_backend(
    void
){
    if(!isInited) return; 
    time_backend = timer_backend = 0;
}



void profile_reset_all_timer(
    void
){
    if(!isInited) return; 
    profile_reset_user();
    profile_reset_backend();
}

void profile_start_user(
    void
){
    if(!isInited) return; 
    timer_user = wtime();
}

 
void profile_start_backend(
    void
){
    if(!isInited) return; 
    timer_backend = wtime();
}

 void profile_end_user(
    void
){
    if(!isInited) return; 
    time_user += wtime() - timer_user;
}



 void profile_end_backend(
    void
){
    if(!isInited) return; 
    time_backend += wtime() - timer_backend;
}


void profile_writeout()
{
    if(!isInited) return; 

#ifdef MEASUREMENT
    fprintf(fp,
             "%s, %d, %d, %f, %f, %f\n",
             hostname, 
             pid, 
             iteration,
             time_user, 
             time_backend,
             time_user + time_backend
            );
#else
    fprintf(fp, "Hostname: %s, PID: %d, Iteration: %d, User Time: %f, Backend Time: %f, Total Time: %f\n",
             hostname, 
             pid, 
             iteration,
             time_user, 
             time_backend,
             time_user + time_backend);
#endif
    profile_reset_all_timer();
}

// print arbitrary text to file in output-to-file mode
void profile_printf(const char* msg, ...)
{
    if(!isInited) return; 
    va_list args;
    va_start(args, msg);
    vfprintf(fp, msg, args);
}

#ifdef __cplusplus
}
#endif //_cplusplus

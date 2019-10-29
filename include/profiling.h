#ifndef _PROFILING_H_
#define _PROFILING_H_

#ifdef __cplusplus
extern "C"
{
#endif //__cplusplus

#ifndef CLOCK_MONOTONIC_COARSE
#define CLOCK_MONOTONIC_COARSE 6
#endif

typedef double (*timefunc) (void);
typedef enum tag_timefunc_type{
    REALTIME = 1,
    REALTIME_SPARSE = 2,
    CPU_TIME = 3
}timefunc_t;

double realtime();
double fast_realtime();
double cputime();

void profile_init(timefunc_t tf);
void profile_close(void);
void profile_set_iteration(int iter);
void profile_reset_user(void);
void profile_reset_backend(void);
void profile_reset_all_timer(void);
void profile_start_user(void);
void profile_start_backend(void);
void profile_end_user(void);
void profile_end_backend(void);
void profile_writeout();
void profile_printf(const char* msg, ...);



#ifdef __cplusplus
}
#endif //__cplusplus

#endif //_PROFILING_H

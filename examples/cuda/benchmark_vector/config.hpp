#define SINGLE
#define EPS 10e-6

//###########################################################################
//Switching between single precision and double precision
//###########################################################################

#ifdef SINGLE
#define TYPE float
#define LOG logf
#define EXP expf
#else
#define TYPE double
#define LOG log
#define EXP exp
#endif

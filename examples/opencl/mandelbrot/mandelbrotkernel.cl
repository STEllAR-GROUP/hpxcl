// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

                                                                        
#pragma OPENCL EXTENSION cl_khr_fp64 : enable                           
                                                                        
float get_red_from_table(size_t id)                                     
{                                                                       
                                                                        
    switch(id)                                                          
    {                                                                   
        case    0: return  10.50000f;                                   
        case    1: return   3.15846f;                                   
        case    2: return   0.39901f;                                   
        case    3: return   0.50955f;                                   
        case    4: return   2.98021f;                                   
        case    5: return   7.85507f;                                   
        case    6: return  15.74049f;                                   
        case    7: return  27.02524f;                                   
        case    8: return  43.28563f;                                   
        case    9: return  66.63533f;                                   
        case   10: return  95.10477f;                                   
        case   11: return 126.26711f;                                   
        case   12: return 157.66678f;                                   
        case   13: return 186.96906f;                                   
        case   14: return 211.81561f;                                   
        case   15: return 229.84409f;                                   
        case   16: return 239.00920f;                                   
        case   17: return 244.51880f;                                   
        case   18: return 248.55106f;                                   
        case   19: return 251.45792f;                                   
        case   20: return 253.30617f;                                   
        case   21: return 254.38364f;                                   
        case   22: return 254.87805f;                                   
        case   23: return 254.50528f;                                   
        case   24: return 242.28555f;                                   
        case   25: return 216.25360f;                                   
        case   26: return 181.10410f;                                   
        case   27: return 141.51198f;                                   
        case   28: return 102.12467f;                                   
        case   29: return  67.43021f;                                   
        case   30: return  41.82657f;                                   
        case   31: return  23.33998f;                                   
        default: return -1.0f;                                          
    }                                                                   
                                                                        
}                                                                       
                                                                        
float get_green_from_table(size_t id)                                   
{                                                                       
                                                                        
    switch(id)                                                          
    {                                                                   
        case    0: return   2.24772f;                                   
        case    1: return   4.38378f;                                   
        case    2: return   6.41868f;                                   
        case    3: return  11.69186f;                                   
        case    4: return  25.28062f;                                   
        case    5: return  45.46635f;                                   
        case    6: return  70.00731f;                                   
        case    7: return  97.07170f;                                   
        case    8: return 124.60256f;                                   
        case    9: return 151.26897f;                                   
        case   10: return 176.37556f;                                   
        case   11: return 199.12393f;                                   
        case   12: return 218.84903f;                                   
        case   13: return 234.93768f;                                   
        case   14: return 246.77850f;                                   
        case   15: return 253.58151f;                                   
        case   16: return 254.53967f;                                   
        case   17: return 251.94934f;                                   
        case   18: return 245.78825f;                                   
        case   19: return 236.19637f;                                   
        case   20: return 223.36536f;                                   
        case   21: return 207.15530f;                                   
        case   22: return 187.81662f;                                   
        case   23: return 165.01135f;                                   
        case   24: return 136.81346f;                                   
        case   25: return 105.00961f;                                   
        case   26: return  72.90131f;                                   
        case   27: return  43.46163f;                                   
        case   28: return  19.82122f;                                   
        case   29: return   5.26334f;                                   
        case   30: return   2.20703f;                                   
        case   31: return   2.29760f;                                   
        default: return -1.0f;                                          
    }                                                                   
                                                                        
}                                                                       
                                                                        
float get_blue_from_table(size_t id)                                    
{                                                                       
                                                                        
    switch(id)                                                          
    {                                                                   
        case    0: return  68.50000f;                                   
        case    1: return  79.36879f;                                   
        case    2: return  95.08916f;                                   
        case    3: return 116.73592f;                                   
        case    4: return 138.96797f;                                   
        case    5: return 160.55014f;                                   
        case    6: return 180.41800f;                                   
        case    7: return 197.60361f;                                   
        case    8: return 211.45720f;                                   
        case    9: return 223.10614f;                                   
        case   10: return 232.60995f;                                   
        case   11: return 240.26654f;                                   
        case   12: return 246.22346f;                                   
        case   13: return 250.50687f;                                   
        case   14: return 253.26976f;                                   
        case   15: return 254.57617f;                                   
        case   16: return 253.42593f;                                   
        case   17: return 231.85834f;                                   
        case   18: return 191.35536f;                                   
        case   19: return 140.38916f;                                   
        case   20: return  87.35393f;                                   
        case   21: return  40.69637f;                                   
        case   22: return   9.10403f;                                   
        case   23: return   0.47247f;                                   
        case   24: return   1.70649f;                                   
        case   25: return   5.54085f;                                   
        case   26: return  11.43449f;                                   
        case   27: return  19.26110f;                                   
        case   28: return  28.73091f;                                   
        case   29: return  39.77259f;                                   
        case   30: return  51.72056f;                                   
        case   31: return  60.93956f;                                   
        default: return -1.0f;                                          
    }                                                                   
                                                                        
}                                                                       
                                                                        
float get_red(double input)                                             
{                                                                       
    float colid = (float)(input * 32.0f);                               
                                                                        
    size_t colid1 = (size_t)colid;                                      
    size_t colid2 = (colid + 1);                                        
                                                                        
    if(colid2 == 32) colid2 = 0;                                        
                                                                        
    float col2pct = colid - colid1;                                     
    float col1pct = 1.0f - col2pct;                                     
                                                                        
    float col1 = get_red_from_table(colid1);                            
    float col2 = get_red_from_table(colid2);                            
                                                                        
    float col = col1 * col1pct + col2 * col2pct;                        
                                                                        
    return col;                                                         
}                                                                       
                                                                        
float get_green(double input)                                           
{                                                                       
    float colid = (float)(input * 32.0f);                               
                                                                        
    size_t colid1 = (size_t)colid;                                      
    size_t colid2 = (colid + 1);                                        
                                                                        
    if(colid2 == 32) colid2 = 0;                                        
                                                                        
    float col2pct = colid - colid1;                                     
    float col1pct = 1.0f - col2pct;                                     
                                                                        
    float col1 = get_green_from_table(colid1);                          
    float col2 = get_green_from_table(colid2);                          
                                                                        
    float col = col1 * col1pct + col2 * col2pct;                        
                                                                        
    return col;                                                         
}                                                                       
                                                                        
float get_blue(double input)                                            
{                                                                       
    float colid = (float)(input * 32.0f);                               
                                                                        
    size_t colid1 = (size_t)colid;                                      
    size_t colid2 = (colid + 1);                                        
                                                                        
    if(colid2 == 32) colid2 = 0;                                        
                                                                        
    float col2pct = colid - colid1;                                     
    float col1pct = 1.0f - col2pct;                                     
                                                                        
    float col1 = get_blue_from_table(colid1);                           
    float col2 = get_blue_from_table(colid2);                           
                                                                        
    float col = col1 * col1pct + col2 * col2pct;                        
                                                                        
    return col;                                                         
}                                                                       
                                                                        
__kernel void precompute_mandelbrot(global unsigned char* out,          
                                    global double* position)            
{                                                                       
                                                                        
    /* calculating index and position */                                
    // get pixel id                                                     
    size_t tid_x = get_global_id(0);                                    
    size_t tid_y = get_global_id(1);                                    
    size_t size_x = get_global_size(0);                                 
    size_t size_y = get_global_size(1);                                 
                                                                        
    // read input values                                                
    double startx = position[0];                                        
    double starty = position[1];                                        
    double hori_pixdist_x = position[2];                                
    double hori_pixdist_y = position[3];                                
    double vert_pixdist_x = position[4];                                
    double vert_pixdist_y = position[5];                                
                                                                        
    // calculate center position                                        
    double posx = startx                                                
                  + (hori_pixdist_x * (((double)tid_x) - 1))            
                  + (vert_pixdist_x * (((double)tid_y) - 1));           
    double posy = starty                                                
                  + (hori_pixdist_y * (((double)tid_x) - 1))            
                  + (vert_pixdist_y * (((double)tid_y) - 1));           
                                                                        
    /* const, need to be more useful */                                 
    unsigned long maxiter = 50000;                                      
    double bailout = 10000.0;                                           
                                                                        
    // initialize vars                                                  
    double mag_square = 0.0;                                            
    unsigned long iter = 0;                                             
//      double start_x = posx;                                              
//      double start_y = posy;                                              
//      double iter_x = -0.726895347709114071439;                           
//      double iter_y = 0.188887129043845954792;                            
    double start_x = 0.0;                                               
    double start_y = 0.0;                                               
    double iter_x = posx;                                               
    double iter_y = posy;                                               
                                                                        
                                                                        
    double x = start_x;                                                 
    double y = start_y;                                                 
                                                                        
    // main calculation                                                 
    while(mag_square <= bailout && iter < maxiter)                      
    {                                                                   
        double xt = x * x - y * y + iter_x;                             
        double yt = 2 * x * y + iter_y;                                 
        x = xt;                                                         
        y = yt;                                                         
        iter = iter + 1;                                                
        mag_square = x * x + y * y;                                     
    }                                                                   
                                                                        
    // calculate colors                                                 
    if(iter == maxiter)                                                 
    {                                                                   
        out[size_x*tid_y + tid_x] = (unsigned char)0;                   
    }                                                                   
    else                                                                
    {                                                                   
        out[size_x*tid_y + tid_x] = (unsigned char)1;                   
    }                                                                   
                                                                        
}                                                                       
                                                                        
__kernel void mandelbrot_alias_8x8(global unsigned char* precalc,       
                                   global unsigned char* out,           
                                   global double* position)             
{                                                                       
                                                                        
    // the local synchronization array                                  
    __local float local_r[64];                                          
    __local float local_g[64];                                          
    __local float local_b[64];                                          
                                                                        
    /* calculating index and position */                                
    // get center pixel id                                              
    size_t tid_x = get_group_id(0);                                     
    size_t tid_y = get_group_id(1);                                     
    size_t size_x = get_num_groups(0);                                  
    size_t size_y = get_num_groups(1);                                  
                                                                        
    // get local worker id                                              
    size_t worker_id_x = get_local_id(0);                               
    size_t worker_id_y = get_local_id(1);                               
                                                                        
    // don't calculate if precalc gives hint that this is maxiter       
    char pre_00 = precalc[(tid_y + 0) * (size_x + 2) + (tid_x + 0)];    
    char pre_01 = precalc[(tid_y + 0) * (size_x + 2) + (tid_x + 1)];    
    char pre_02 = precalc[(tid_y + 0) * (size_x + 2) + (tid_x + 2)];    
    char pre_10 = precalc[(tid_y + 1) * (size_x + 2) + (tid_x + 0)];    
    char pre_11 = precalc[(tid_y + 1) * (size_x + 2) + (tid_x + 1)];    
    char pre_12 = precalc[(tid_y + 1) * (size_x + 2) + (tid_x + 2)];    
    char pre_20 = precalc[(tid_y + 2) * (size_x + 2) + (tid_x + 0)];    
    char pre_21 = precalc[(tid_y + 2) * (size_x + 2) + (tid_x + 1)];    
    char pre_22 = precalc[(tid_y + 2) * (size_x + 2) + (tid_x + 2)];    
                                                                        
    if(   pre_00 == 0                                                   
       && pre_01 == 0                                                   
       && pre_02 == 0                                                   
       && pre_10 == 0                                                   
       && pre_11 == 0                                                   
       && pre_12 == 0                                                   
       && pre_20 == 0                                                   
       && pre_21 == 0                                                   
       && pre_22 == 0)                                                  
    {                                                                   
        if(worker_id_x == 0 && worker_id_y == 0)                        
        {                                                               
            out[3*(size_x*tid_y + tid_x) + 0] = (unsigned char)0;       
            out[3*(size_x*tid_y + tid_x) + 1] = (unsigned char)0;       
            out[3*(size_x*tid_y + tid_x) + 2] = (unsigned char)0;       
        }                                                               
                                                                        
        return;                                                         
    }                                                                   
                                                                        
    // read input values                                                
    double startx = position[0];                                        
    double starty = position[1];                                        
    double hori_pixdist_x = position[2];                                
    double hori_pixdist_y = position[3];                                
    double vert_pixdist_x = position[4];                                
    double vert_pixdist_y = position[5];                                
                                                                        
    // calculate center position                                        
    double poscenterx = startx                                          
                        + (hori_pixdist_x * tid_x)                      
                        + (vert_pixdist_x * tid_y);                     
    double poscentery = starty                                          
                        + (hori_pixdist_y * tid_x)                      
                        + (vert_pixdist_y * tid_y);                     
                                                                        
    // calculate local position                                         
    double local_pos_x = (worker_id_x - 3.5) * 0.125; // divided by 8   
    double local_pos_y = (worker_id_y - 3.5) * 0.125; // divided by 8   
                                                                        
    // calculate local coords                                           
    double posx = poscenterx + local_pos_x * hori_pixdist_x             
                             + local_pos_y * vert_pixdist_x;            
    double posy = poscentery + local_pos_x * hori_pixdist_y             
                             + local_pos_y * vert_pixdist_y;            
                                                                        
    /* const, need to be more useful */                                 
    unsigned long maxiter = 50000;                                      
    double bailout = 10000.0;                                           
                                                                        
    // initialize vars                                                  
    double mag_square = 0.0;                                            
    unsigned long iter = 0;                                             
                                                                        
//      double start_x = posx;                                              
//      double start_y = posy;                                              
//      double iter_x = -0.726895347709114071439;                           
//      double iter_y = 0.188887129043845954792;                            
    double start_x = 0.0;                                               
    double start_y = 0.0;                                               
    double iter_x = posx;                                               
    double iter_y = posy;                                               
                                                                        
                                                                        
    double x = start_x;                                                 
    double y = start_y;                                                 
                                                                        
    // main calculation                                                 
    while(mag_square <= bailout && iter < maxiter)                      
    {                                                                   
        double xt = x * x - y * y + iter_x;                             
        double yt = 2 * x * y + iter_y;                                 
        x = xt;                                                         
        y = yt;                                                         
        iter = iter + 1;                                                
        mag_square = x * x + y * y;                                     
    }                                                                   
                                                                        
    // calculate abs of final number                                    
    double cabs = sqrt(mag_square);                                     
                                                                        
    // precalculate often needed numbers at compile time                
    double il = 1.0/log(2.0);                                           
    double lp = log(log(128.0));                                        
                                                                        
    // calculate color index                                            
    double index1 = 0.42 * 0.05 * (iter + il*lp - il * log(log(cabs))); 
    double index = fmod(log(index1+1.0), 1.0);                          
                                                                        
    // calculate colors                                                 
    if(iter == maxiter)                                                 
    {                                                                   
        local_r[8*worker_id_y + worker_id_x] = 0.0f;                    
        local_g[8*worker_id_y + worker_id_x] = 0.0f;                    
        local_b[8*worker_id_y + worker_id_x] = 0.0f;                    
    }                                                                   
    else                                                                
    {                                                                   
        local_r[8*worker_id_y + worker_id_x] = get_red(index);          
        local_g[8*worker_id_y + worker_id_x] = get_green(index);        
        local_b[8*worker_id_y + worker_id_x] = get_blue(index);         
    }                                                                   
                                                                        
    // Wait for other threads to finish                                 
    barrier(CLK_LOCAL_MEM_FENCE);                                       
                                                                        
    // combine all color results                                        
    if(worker_id_x == 0 && worker_id_y == 0)                            
    {                                                                   
                                                                        
        float sum_r = 0.0f;                                             
        float sum_g = 0.0f;                                             
        float sum_b = 0.0f;                                             
                                                                        
        for(size_t i = 0; i < 64; i++)                                  
        {                                                               
            sum_r += local_r[i];                                        
            sum_g += local_g[i];                                        
            sum_b += local_b[i];                                        
        }                                                               
                                                                        
        out[3*(size_x*tid_y + tid_x) + 0] = (unsigned char)(sum_r/64.0);
        out[3*(size_x*tid_y + tid_x) + 1] = (unsigned char)(sum_g/64.0);
        out[3*(size_x*tid_y + tid_x) + 2] = (unsigned char)(sum_b/64.0);
                                                                        
    }                                                                   
                                                                        
}                                                                       
                                                                        


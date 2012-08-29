#include <iostream>

#include "demo.h"

using namespace std;

int demo::test(float* arr, int size)
{
    int i;
    for(i=0; i < size; ++i)
        cout << i << ":" << ((float*)arr)[i] << endl;
    return i;
}

#pragma once

#include <cstdint>

inline bool I64ToI32(int64_t input, int& output){
    if(input > INT32_MAX){
        return false;
    }
    output = static_cast<int>(input);
    return true;
}
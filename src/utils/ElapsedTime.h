#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

class ElapsedTime
{
public:
    ElapsedTime() {
        // start timer
        this->begin = std::chrono::steady_clock::now();
    }

    void tic() {
        // start timer
        this->begin = std::chrono::steady_clock::now();
    }


    int toc_milli() {
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        return (int) std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }

    int toc_micro() {
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        return (int) std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    }

    int toc( ) {
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        return (int) std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    }



private:
    std::chrono::steady_clock::time_point begin;

};

class DateAndTime
{
public:
    static std::string current_date_and_time()
    {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);

        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X");
        return ss.str();
    }

    // TODO Can make more functions later to get hr, min, date etc. Not priority.

};

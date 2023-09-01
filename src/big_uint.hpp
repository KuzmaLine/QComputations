#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>

namespace {
    constexpr unsigned int MODULE = (unsigned int)1000000000;
    constexpr unsigned int MODULE_LOG = 9;
    constexpr unsigned int NUMTYPE_MAX = std::numeric_limits<unsigned int>::max();
    constexpr unsigned int BITS_COUNT = 32;
}

namespace QComputations {

class BigUInt {
    using NumType = unsigned int;
    public:
        explicit BigUInt() : num_(), base_(NUMTYPE_MAX) {}
        explicit BigUInt(const std::string& num_str);
        explicit BigUInt(NumType num) : base_(NUMTYPE_MAX) { num_ = {num}; }

        BigUInt& operator=(const BigUInt& num);

        BigUInt operator+(const BigUInt& num) const;
        BigUInt operator-(const BigUInt& num) const;
        BigUInt operator*(const BigUInt& num) const;
        BigUInt operator/(const BigUInt& num) const;
        BigUInt operator%(const BigUInt& num) const;

        void operator+=(const BigUInt& num);
        void operator-=(const BigUInt& num);
        void operator*=(const BigUInt& num);
        void operator/=(const BigUInt& num);
        void operator%=(const BigUInt& num);

        BigUInt operator|(const BigUInt& num) const;
        BigUInt operator&(const BigUInt& num) const;

        void operator|=(const BigUInt& num);
        void operator&=(const BigUInt& num);

        bool operator<(const BigUInt& num) const;
        bool operator<=(const BigUInt& num) const;
        bool operator==(const BigUInt& num) const;
        bool operator!=(const BigUInt& num) const;
        bool operator>(const BigUInt& num) const;
        bool operator>=(const BigUInt& num) const;

        BigUInt operator<<(size_t shift) const;
        BigUInt operator>>(size_t shift) const;
        void operator<<=(size_t shift);
        void operator>>=(size_t shift);

        std::string binary_str() const;
        std::string num_str() const;

        friend std::ostream& operator<<(std::ostream& out, const BigUInt& num);
        friend BigUInt concatenate(const BigUInt& a, const BigUInt& b);

        NumType num(size_t index) const { return num_[index]; }

        long long high_order_bit_id() const;
        size_t num_size() const { return num_.size(); }

        NumType get_bits_count() const { return BITS_COUNT; }
        NumType get_num(size_t id = 0) const { return num_[id]; }
        NumType get_bit(size_t id) const;
        NumType to_uint() const { return num_[0]; }
    private:
        std::vector<NumType> num_; 
        NumType base_;
};

} // namespace QComputations
#include "big_uint.hpp"
#include <bitset>
#include <cassert>
#include "test.hpp"

namespace {
    const BigUInt BigZero(0);
    const BigUInt BigOne(1);

    struct Division {
        BigUInt Quotinent;
        BigUInt Residue;
    };


    // https://systo.ru/prog/pract/int_div.html first algorithm
    Division div(const BigUInt& a, const BigUInt& b)  {
        Division res;
        if (a < b) { return Division{BigZero, a}; }


        auto A(a);
        auto B(b);

        long long shift_count = A.high_order_bit_id() - B.high_order_bit_id();
        B <<= shift_count;

        //std::cout << A.binary_str() << std::endl;
        //std::cout << B.binary_str() << std::endl;
        for (size_t i = 0; i != shift_count + 1; i++) {
            //std::cout << shift_count << " " << A.num(0) << " " << B.num(0) << " | PATH - " << (A >= B) << " " << A.num_size() << " " << B.num_size() << std::endl;
            if (A >= B) {
                res.Quotinent = ((res.Quotinent << 1) + BigOne);
                A = A - B;
            } else {
                res.Quotinent <<= 1;
            }

            B >>= 1;
            //std::cout << "RES: " << B.binary_str() << std::endl;
            //std::cout << "RES: " << A.num(0) << " " << B.num(0) << " | " << res.Quotinent.binary_str() << std::endl;
        }

        res.Residue = A;

        return res;
    }
}

/*
BigUInt::BigUInt(const std::string& str) : base_(std::pow(10, 9)) {
    for (long long i = (long long)str.size(); i > 0; i -= NUM_COUNT_IN_VEC_ELEM) {
        if (i < NUM_COUNT_IN_VEC_ELEM) {
            num_.push_back(std::atoi(str.substr(0, i).c_str()));
        } else {
            num_.push_back(std::atoi(str.substr(i - NUM_COUNT_IN_VEC_ELEM, NUM_COUNT_IN_VEC_ELEM).c_str()));
        }
    }
}
*/

std::string BigUInt::num_str() const {
    std::string res;

    auto tmp = *this;
    while(tmp != BigZero) {
        Division dv = div(tmp, BigUInt(MODULE));

        res = std::to_string(dv.Residue.num(0)) + res;
        tmp = dv.Quotinent;
    }

    return res;
}

std::ostream& operator<<(std::ostream& out, const BigUInt& num) {
    out << num.num_str();
    return out;
}


long long BigUInt::high_order_bit_id() const {
    if (num_.empty()) return -1;

    size_t res = num_.size() * BITS_COUNT - 1;

    NumType mask = NumType(0x80000000);
    //std::cout << "START - " << this-> binary_str() << std::endl;
    //printing::binary_print(num_[num_.size() - 1]);
    //printing::binary_print(mask);
    //std::cout << mask << std::endl;
    while ((num_[num_.size() - 1] & mask) == NumType(0) and mask != NumType(0)) {
        //printing::binary_print(num_[num_.size() - 1]);
        //printing::binary_print(mask);
        //std::cout << std::endl;
        res--;
        mask >>= 1;
    }

    //std::cout << "ID RES: " << res << std::endl;

    return res;
}

std::string BigUInt::binary_str() const {
    std::string res;
    for (long long i = num_.size() - 1; i >= 0; i--) {
        std::bitset<BITS_COUNT> bset(num_[i]);
        res += bset.to_string();
    }

    return res;
}

BigUInt& BigUInt::operator=(const BigUInt& num) {
    num_ = num.num_;
    base_ = num.base_;
    return *this;
}

bool BigUInt::operator<(const BigUInt& num) const {
    if (num_.size() < num.num_.size()) {
        return true;
    } else if (num_.size() > num.num_.size()) {
        return false;
    }

    for (long long i = num_.size() - 1; i >= 0; i--) {
        if (num_[i] < num.num_[i]) {
            return true;
        } else if (num_[i] > num.num_[i]) {
            return false;
        }
    }

    return false;
}

bool BigUInt::operator<=(const BigUInt& num) const {
    if (num_.size() < num.num_.size()) {
        return true;
    } else if (num_.size() > num.num_.size()) {
        return false;
    }

    for (long long i = num_.size() - 1; i >= 0; i--) {
        if (num_[i] < num.num_[i]) {
            return true;
        } else if (num_[i] > num.num_[i]) {
            return false;
        }
    }

    return true;
}

bool BigUInt::operator>=(const BigUInt& num) const {
    if (num_.size() < num.num_.size()) {
        return false;
    } else if (num_.size() > num.num_.size()) {
        return true;
    }

    for (long long i = num_.size() - 1; i >= 0; i--) {
        if (num_[i] < num.num_[i]) {
            return false;
        } else if (num_[i] > num.num_[i]) {
            return true;
        }
    }

    return true;
}

bool BigUInt::operator>(const BigUInt& num) const {
    if (num_.size() < num.num_.size()) {
        return false;
    } else if (num_.size() > num.num_.size()) {
        return true;
    }

    for (long long i = num_.size() - 1; i >= 0; i--) {
        if (num_[i] < num.num_[i]) {
            return false;
        } else if (num_[i] > num.num_[i]) {
            return true;
        }
    }

    return false;
}

bool BigUInt::operator==(const BigUInt& num) const {
    if (num_.size() != num.num_.size()) {
        return false;
    }

    for (long long i = num_.size() - 1; i >= 0; i--) {
        if (num_[i] != num.num_[i]) {
            return false;
        }
    }

    return true;
}

bool BigUInt::operator!=(const BigUInt& num) const {
    if (num_.size() != num.num_.size()) {
        return true;
    }

    for (long long i = num_.size() - 1; i >= 0; i--) {
        if (num_[i] != num.num_[i]) {
            return true;
        }
    }

    return false;
}

BigUInt BigUInt::operator>>(size_t shift) const {
    BigUInt res;

    res.num_ = num_;
    if (shift == 0) return res;

    if (BITS_COUNT < shift) {
        size_t start = 0;
        for (start = 0; BITS_COUNT < shift; shift -= BITS_COUNT) { start++; }

        size_t index = 0;
        for (size_t i = start; i < res.num_.size(); i++) {
            res.num_[index++] = res.num_[i];
        }

        //std::cout << ">> : " << start << " " << index << std::endl;
        res.num_.resize(index);
    }

    NumType carry = 0;
    for (long long i = res.num_.size() - 1; i >= 0; i--) {
        NumType tmp = res.num_[i] << (BITS_COUNT - shift);

        res.num_[i] >>= shift;
        res.num_[i] |= carry;

        carry = tmp;
    }

    if (res.num_size() * BITS_COUNT - BITS_COUNT > res.high_order_bit_id()) res.num_.resize(num_.size() - 1);

    return res;
}

void BigUInt::operator>>=(size_t shift) {
    *this = *this >> shift;
}

BigUInt BigUInt::operator<<(size_t shift) const {
    BigUInt res;

    res.num_ = num_;
    if (shift == 0) return res;

    for (size_t i = 0; BITS_COUNT < shift; shift -= BITS_COUNT) {
        res = concatenate(BigZero, res);
    }

    NumType carry = 0;
    //std::cout << res.binary_str() << std::endl;
    //std::cout << "Here - " << BITS_COUNT << " " << res.high_order_bit_id() % BITS_COUNT << " " << BITS_COUNT - res.high_order_bit_id() % BITS_COUNT << std::endl; 
    if (BITS_COUNT - res.high_order_bit_id() % BITS_COUNT <= shift or res.num_size() == 0) {
        res = concatenate(res, BigZero);
    }
    for (size_t i = 0; i < res.num_.size(); i++) {
        NumType tmp = res.num_[i] >> (BITS_COUNT - shift);
        res.num_[i] <<= shift;
        res.num_[i] |= carry;

        carry = tmp;
    }

    if (carry != 0) res = concatenate(res, BigUInt(carry));
 
    return res;
}

void BigUInt::operator<<=(size_t shift) {
    *this = *this << shift;
}

BigUInt BigUInt::operator-(const BigUInt& num) const {
    assert(*this >= num);

    BigUInt res;

    auto res_size = std::max(num_.size(), num.num_.size());
    auto min_size = std::min(num_.size(), num.num_.size());
    std::vector<NumType> carry_bit(res_size + 1, 0);
    for (size_t i = 0; i < res_size; i++) {
        NumType a, b;
        if (i >= min_size) {
            if (num_.size() < num.num_.size()) {
                a = 0;
                b = num.num_[i];
            } else {
                a = num_[i];
                b = 0;
            }
        } else {
            a = num_[i];
            b = num.num_[i];
        }

        carry_bit[i + 1] = (a == 0 and carry_bit[i] == 1) or ((a - carry_bit[i]) < b) ? 1 : 0;
    }

    bool flag_is_start = false;
    for (long long i = res_size - 1; i >= 0; i--) {
        NumType a, b;
        if (i >= min_size) {
            if (num_.size() < num.num_.size()) {
                a = 0;
                b = num.num_[i];
            } else {
                a = num_[i];
                b = 0;
            }
        } else {
            a = num_[i];
            b = num.num_[i];
        }        

        //std::cout << i << " " << a << " " << b << " " << carry_bit[i] << std::endl;
        auto tmp = BigUInt(a - b - carry_bit[i]);
        if (flag_is_start or tmp != BigZero) {
            flag_is_start = true;
            res = concatenate(BigUInt(a - b - carry_bit[i]), res);
        }
    }

    if (!flag_is_start) res = BigZero;

    return res;
}

BigUInt BigUInt::operator/(const BigUInt& num) const {
    return (div(*this, num)).Quotinent;
}

BigUInt BigUInt::operator%(const BigUInt& num) const {
    return (div(*this, num)).Residue;
}

BigUInt BigUInt::operator+(const BigUInt& num) const {
    BigUInt res;

    auto res_size = std::max(num_.size(), num.num_.size());
    auto min_size = std::min(num_.size(), num.num_.size());
    std::vector<NumType> carry_bit(res_size + 1, 0);
    for (size_t i = 0; i < res_size; i++) {
        NumType a, b;
        if (i >= min_size) {
            if (num_.size() < num.num_.size()) {
                a = 0;
                b = num.num_[i];
            } else {
                a = num_[i];
                b = 0;
            }
        } else {
            a = num_[i];
            b = num.num_[i];
        }

        carry_bit[i + 1] = (b + carry_bit[i] > NUMTYPE_MAX - a) or (b == NUMTYPE_MAX and carry_bit[i] == 1) ? 1 : 0;
    }

    for (size_t i = 0; i < res_size; i++) {
        NumType a, b;
        if (i >= min_size) {
            if (num_.size() < num.num_.size()) {
                a = 0;
                b = num.num_[i];
            } else {
                a = num_[i];
                b = 0;
            }
        } else {
            a = num_[i];
            b = num.num_[i];
        }

        res = concatenate(res, BigUInt(a + b + carry_bit[i]));
    }

    if (carry_bit[res_size] == 1) {
        res.num_.emplace_back(1);
    }

    return res;
}

BigUInt concatenate(const BigUInt& a, const BigUInt& b) {
    BigUInt res;

    for (auto num: a.num_) {
        res.num_.push_back(num);
    }

    for (auto num: b.num_) {
        res.num_.push_back(num);
    }

    return res;
}
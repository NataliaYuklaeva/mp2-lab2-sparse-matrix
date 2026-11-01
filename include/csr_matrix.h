#pragma once

#include <cstdlib>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

template <typename T>
struct ValDifferent {
    static bool Different(const T& a, const T& b) {
        return a == b;
    }
};

template <>
struct ValDifferent<float> {
    static bool Different(const float& a, const float& b) {
        float eps = std::numeric_limits<float>::epsilon() *
            std::max({ 1.0f, std::abs(a), std::abs(b) });
        return std::abs(a - b) <= eps;
    }
};

template <>
struct ValDifferent<double> {
    static bool Different(const double& a, const double& b) {
        double eps = std::numeric_limits<double>::epsilon() *
            std::max({ 1.0, std::abs(a), std::abs(b) });
        return std::abs(a - b) <= eps;
    }
};

template <typename ValT>
class CSR_matrix {
public:
    size_t n_rows = 0; // число строк
    size_t n_cols = 0; // число столбцов
    size_t nz = 0; // число ненулевых элементов
    size_t capacity = 0; // объем памяти для хранения значений
    size_t* RowIndex = nullptr;
    size_t* Col = nullptr;
    ValT* Val = nullptr;

    CSR_matrix(size_t _m = 1, size_t _n = 1) : n_rows(_m), n_cols(_n), Col(nullptr), Val(nullptr) {
        RowIndex = new size_t[n_rows + 1]();
    }

    CSR_matrix(size_t _m, size_t _n, size_t _nz) : n_rows(_m), n_cols(_n), nz(_nz) {
        RowIndex = new size_t[n_rows + 1] ();
        if (nz > 0) {
            Col = new size_t[nz];
            Val = new ValT[nz]();
        }
        else {
			Col = nullptr;
			Val = nullptr;
        }
        capacity = nz;
    }

    CSR_matrix(const CSR_matrix& copy) : n_rows(copy.n_rows), n_cols(copy.n_cols), nz(copy.nz), capacity(copy.capacity) {
        RowIndex = new size_t[n_rows + 1];
        std::copy(copy.RowIndex, copy.RowIndex + n_rows + 1, RowIndex);
        if (copy.nz > 0 && copy.Col != nullptr && copy.Val != nullptr) {
            Val = new ValT[nz];
            std::copy(copy.Val, copy.Val + nz, Val);
            Col = new size_t[nz];
            std::copy(copy.Col, copy.Col + nz, Col);
        }
        else {
            Val = nullptr;
			Col = nullptr;
        }
    }

    CSR_matrix(CSR_matrix&& mov) : n_rows(mov.n_rows), n_cols(mov.n_cols), nz(mov.nz), capacity(mov.capacity) {
        Col = mov.Col;
        RowIndex = mov.RowIndex;
        Val = mov.Val;

        mov.Col = nullptr;
        mov.RowIndex = nullptr;
        mov.Val = nullptr;
    }

    ~CSR_matrix() {
        delete[] Col;
        delete[] RowIndex;
        delete[] Val;
    }

    CSR_matrix& operator=(const CSR_matrix& copy) {
        if (this == &copy)
            return *this;

        if (n_rows != copy.n_rows) {
            delete[] RowIndex;
            RowIndex = new size_t[copy.n_rows + 1];
        }
        std::copy(copy.RowIndex, copy.RowIndex + copy.n_rows + 1, RowIndex);
        if (capacity < copy.nz) {
            delete[] Col;
            delete[] Val;

            Col = new size_t[copy.nz];
            Val = new ValT[copy.nz];
            capacity = copy.nz;
        }
        std::copy(copy.Col, copy.Col + copy.nz, Col);
        std::copy(copy.Val, copy.Val + copy.nz, Val);

        n_rows = copy.n_rows;
        n_cols = copy.n_cols;
        nz = copy.nz;

        return *this;
    }

    CSR_matrix& operator=(CSR_matrix&& mov) {
        if (this == &mov)
            return *this;

		std::swap(Col, mov.Col);
		std::swap(RowIndex, mov.RowIndex);
		std::swap(Val, mov.Val);
		std::swap(n_rows, mov.n_rows);
		std::swap(n_cols, mov.n_cols);
		std::swap(nz, mov.nz);
		std::swap(capacity, mov.capacity);

        return *this;
    }
    

    void set_from_vectors(const std::vector<size_t>& cols, const std::vector<size_t>& rows, 
        const std::vector<ValT>& vals) {

        size_t new_nz = cols.size();
        if (nz != new_nz) {
            delete[] Col;
            delete[] Val;
            nz = new_nz;
            Col = new size_t[new_nz];
            Val = new ValT[new_nz];
        }
		if (n_rows != rows.size() - 1) {
			delete[] RowIndex;
			RowIndex = new size_t[n_rows + 1];
            n_rows = rows.size() - 1;
			n_cols = std::max(*std::max_element(cols.begin(), cols.end()) + 1, n_rows);
		}
		std::copy(cols.begin(), cols.end(), Col);
		std::copy(vals.begin(), vals.end(), Val);
        std::copy(rows.begin(), rows.end(), RowIndex);
    }

    void generate_matrix(const size_t avg_degree, ValT minVal, ValT maxVal) {
        size_t new_nz = n_rows * avg_degree;
        if (avg_degree > n_cols) {
            throw std::invalid_argument("avg_degree cannot be greater than the number of columns");
        }

        if (nz != new_nz) {
            delete[] Col;
            delete[] Val;

            nz = new_nz;
            capacity = nz;

            Col = new size_t[nz];
            Val = new ValT[nz];
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        using DistT = std::conditional_t<std::is_floating_point_v<ValT>,
            std::uniform_real_distribution<ValT>,
            std::uniform_int_distribution<ValT>>;
        DistT val_dist(minVal, maxVal);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        double p = static_cast<double>(avg_degree) / n_cols;

        size_t current_elem = 0;
        RowIndex[0] = 0;

        for (size_t i = 0; i < n_rows; ++i) {
            for (size_t j = 0; j < n_cols && current_elem < nz; ++j) {
                if (prob_dist(gen) < p) {
                    Col[current_elem] = j;
                    Val[current_elem] = val_dist(gen);
                    current_elem++;
                }
            }
            RowIndex[i + 1] = current_elem;
        }
        nz = RowIndex[n_rows];
        capacity = nz;
    }


	void print_coo() const {
		std::cout << "row  col val\n";
		for (size_t i = 0; i < n_rows; ++i)
			for (size_t j = RowIndex[i]; j < RowIndex[i + 1]; ++j)
				std::cout << i << " " << Col[j] << " " << Val[j] << '\n';
	}


    void print_dense() const {
		for (size_t i = 0; i < n_rows; ++i) {
			size_t k = 0;
			for (size_t j = RowIndex[i]; j < RowIndex[i + 1]; ++j, ++k) {
				while (k < Col[j]) {
					std::cout << 0 << " ";
					++k;
				}
				std::cout << Val[j] << " ";
			}
			while (k < n_cols) {
				std::cout << 0 << " ";
				++k;
			}
			std::cout << '\n';
		}
		std::cout << '\n';

    }

	// сложение матриц в формате CSR
    CSR_matrix operator+(const CSR_matrix& other) const {
        if (n_rows != other.n_rows || n_cols != other.n_cols)
            throw std::invalid_argument("CSR_matrix::operator+: size mismatch");

        size_t max_nz = nz + other.nz;
        CSR_matrix result(n_rows, n_cols, max_nz);

        size_t k = 0;             
        result.RowIndex[0] = 0;

        for (size_t i = 0; i < n_rows; ++i) {
            size_t a = RowIndex[i], aEnd = RowIndex[i + 1];
            size_t b = other.RowIndex[i], bEnd = other.RowIndex[i + 1];

            while (a < aEnd && b < bEnd) {
                size_t ca = Col[a];
                size_t cb = other.Col[b];

                if (ca == cb) {
                    ValT v = Val[a] + other.Val[b];
                    result.Col[k] = ca;
                    result.Val[k] = v;
                    ++k; ++a; ++b;
                }
                else if (ca < cb) {
                    result.Col[k] = ca;
                    result.Val[k] = Val[a];
                    ++k; ++a;
                }
                else {
                    result.Col[k] = cb;
                    result.Val[k] = other.Val[b];
                    ++k; ++b;
                }
            }
            while (a < aEnd) {
                result.Col[k] = Col[a];
                result.Val[k] = Val[a];
                ++k; ++a;
            }
            while (b < bEnd) {
                result.Col[k] = other.Col[b];
                result.Val[k] = other.Val[b];
                ++k; ++b;
            }
            result.RowIndex[i + 1] = k;
        }

        result.nz = k;
        result.capacity = max_nz;
        return result;
    }
    
    //умножение на вектор
    std::vector<ValT> operator*(const std::vector<ValT>& x) const {
        if (x.size() != n_cols)
            throw std::invalid_argument("CSR_matrix::operator*: vector size mismatch");

        std::vector<ValT> y(n_rows, ValT{});

        for (size_t i = 0; i < n_rows; ++i) {
            ValT sum = ValT{};
            for (size_t k = RowIndex[i]; k < RowIndex[i + 1]; ++k) {
                sum = sum + Val[k] * x[Col[k]];
            }
            y[i] = sum;
        }
        return y;
    }

	// транспонирование матрицы в формате CSR
    CSR_matrix transpose() const {
            CSR_matrix T(n_cols, n_rows, nz);  

            for (size_t i = 0; i <= n_cols; ++i) T.RowIndex[i] = 0;
            for (size_t k = 0; k < nz; ++k)
                ++T.RowIndex[Col[k] + 1];
            for (size_t j = 0; j < n_cols; ++j)
                T.RowIndex[j + 1] += T.RowIndex[j];

            std::vector<size_t> next(n_cols);
            for (size_t j = 0; j < n_cols; ++j)
                next[j] = T.RowIndex[j];

            for (size_t i = 0; i < n_rows; ++i) {
                for (size_t k = RowIndex[i]; k < RowIndex[i + 1]; ++k) {
                    size_t col = Col[k];
                    size_t dst = next[col]++;
                    T.Col[dst] = i;
                    T.Val[dst] = Val[k];
                }
            }

            T.nz = nz;
            T.capacity = nz;
            return T;
    }

	// конвертер из координатного формата в формат CSR
    static CSR_matrix from_coo(size_t n_rows, size_t n_cols,
        const std::vector<size_t>& rows,
        const std::vector<size_t>& cols,
        const std::vector<ValT>& vals)
    {
        if (rows.size() != cols.size() || rows.size() != vals.size())
            throw std::invalid_argument("CSR_matrix::from_coo: size mismatch");

        const size_t nnz = vals.size();
        CSR_matrix M(n_rows, n_cols, nnz);

        for (size_t i = 0; i <= n_rows; ++i) M.RowIndex[i] = 0;
        for (size_t k = 0; k < nnz; ++k) {
            if (rows[k] >= n_rows || cols[k] >= n_cols)
                throw std::out_of_range("CSR_matrix::from_coo: index out of range");
            ++M.RowIndex[rows[k] + 1];
        }

        for (size_t i = 0; i < n_rows; ++i)
            M.RowIndex[i + 1] += M.RowIndex[i];

        std::vector<size_t> next(n_rows);
        for (size_t i = 0; i < n_rows; ++i) next[i] = M.RowIndex[i];

        for (size_t k = 0; k < nnz; ++k) {
            size_t r = rows[k];
            size_t dst = next[r]++;
            M.Col[dst] = cols[k];
            M.Val[dst] = vals[k];
        }
    
        M.nz = nnz;
        M.capacity = nnz;
        return M;
    }

	
    bool operator==(const CSR_matrix& other) const {
        if (n_rows != other.n_rows || n_cols != other.n_cols || nz != other.nz) {
            return false;
        }
        for (size_t i = 0; i <= n_rows; ++i) {
            if (RowIndex[i] != other.RowIndex[i]) {
                return false;
            }
        }
        for (size_t k = 0; k < nz; ++k) {
            if (Col[k] != other.Col[k]) {
                return false;
            }
            if (!ValDifferent<ValT>::Different(Val[k], other.Val[k])) {
                return false;
            }
        }
        return true;
    }
};
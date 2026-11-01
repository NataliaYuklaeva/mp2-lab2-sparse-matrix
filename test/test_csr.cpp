#include <gtest/gtest.h>
#include "csr_matrix.h"
#include <limits>
#include <vector>
#include <sstream>

template <typename T>
static CSR_matrix<T> make_matrix(size_t m, size_t n,
    const std::vector<size_t>& cols,
    const std::vector<size_t>& rows,
    const std::vector<T>& vals) {
    CSR_matrix<T> M(m, n, vals.size());
    M.set_from_vectors(cols, rows, vals);
    return M;
}


TEST(TestMatrix, can_create_matrix)
{
    ASSERT_NO_THROW(CSR_matrix<double> m(10, 10, 20));
}

TEST(TestMatrix, create_matrix_sets_dimensions)
{
    CSR_matrix<double> m(7, 9, 5);
    EXPECT_EQ(m.n_rows, 7u);
    EXPECT_EQ(m.n_cols, 9u);
    EXPECT_EQ(m.nz, 5u);
    EXPECT_EQ(m.capacity, 5u);
    ASSERT_NE(m.RowIndex, nullptr);
    ASSERT_NE(m.Col, nullptr);
    ASSERT_NE(m.Val, nullptr);
    for (size_t i = 0; i <= m.n_rows; ++i) {
        EXPECT_EQ(m.RowIndex[i], 0u);
    }
}

TEST(TestMatrix, create_matrix_zero_nz_has_null_ptrs)
{
    CSR_matrix<double> m(5, 5, 0);
    EXPECT_EQ(m.nz, 0u);
    EXPECT_EQ(m.capacity, 0u);
    EXPECT_EQ(m.Col, nullptr);
    EXPECT_EQ(m.Val, nullptr);
    ASSERT_NE(m.RowIndex, nullptr);
    for (size_t i = 0; i <= m.n_rows; ++i) EXPECT_EQ(m.RowIndex[i], 0u);
}

TEST(TestMatrix, default_constructor)
{
    CSR_matrix<double> m;
    EXPECT_EQ(m.n_rows, 1u);
    EXPECT_EQ(m.n_cols, 1u);
    EXPECT_EQ(m.nz, 0u);
    EXPECT_EQ(m.Col, nullptr);
    EXPECT_EQ(m.Val, nullptr);
    ASSERT_NE(m.RowIndex, nullptr);
    EXPECT_EQ(m.RowIndex[0], 0u);
    EXPECT_EQ(m.RowIndex[1], 0u);
}

TEST(TestMatrix, copy_constructor_deep_copies)
{
    auto m = make_matrix<double>(3, 4, { 0,2,1,3 }, { 0,2,2,4 }, { 1.0,2.0,3.0,4.0 });
    CSR_matrix<double> copy(m);

    EXPECT_EQ(copy.n_rows, m.n_rows);
    EXPECT_EQ(copy.n_cols, m.n_cols);
    EXPECT_EQ(copy.nz, m.nz);
    EXPECT_TRUE(copy == m);

    EXPECT_NE(copy.RowIndex, m.RowIndex);
    EXPECT_NE(copy.Col, m.Col);
    EXPECT_NE(copy.Val, m.Val);

    copy.Val[0] = 999.0;
    EXPECT_NE(copy.Val[0], m.Val[0]);
}

TEST(TestMatrix, copy_constructor_empty)
{
    CSR_matrix<double> m(3, 3, 0);
    CSR_matrix<double> copy(m);
    EXPECT_EQ(copy.nz, 0u);
    EXPECT_EQ(copy.Col, nullptr);
    EXPECT_EQ(copy.Val, nullptr);
    EXPECT_TRUE(copy == m);
}

TEST(TestMatrix, move_constructor_transfers_ownership)
{
    auto m = make_matrix<double>(3, 3,
    { 0, 1, 2 },
    { 0, 1, 2, 3 },
    { 1.0, 2.0, 3.0 });
    auto* rowPtr = m.RowIndex;
    auto* colPtr = m.Col;
    auto* valPtr = m.Val;

    CSR_matrix<double> mv(std::move(m));

    EXPECT_EQ(mv.n_rows, 3u);
    EXPECT_EQ(mv.n_cols, 3u);
    EXPECT_EQ(mv.nz, 3u);
    EXPECT_EQ(mv.RowIndex, rowPtr);
    EXPECT_EQ(mv.Col, colPtr);
    EXPECT_EQ(mv.Val, valPtr);

    EXPECT_EQ(m.RowIndex, nullptr);
    EXPECT_EQ(m.Col, nullptr);
    EXPECT_EQ(m.Val, nullptr);
}

TEST(TestMatrix, copy_assignment_same_sizes)
{
    auto a = make_matrix<double>(3, 3, { 0,2,1 }, { 0,2,3 }, { 1.0,2.0,3.0 });
    auto b = make_matrix<double>(3, 3, { 0,1,2 }, { 0,2,3 }, { 9.0,8.0,7.0 });

    b = a;
    EXPECT_TRUE(b == a);
    EXPECT_NE(b.RowIndex, a.RowIndex);
    EXPECT_NE(b.Val, a.Val);
}

TEST(TestMatrix, copy_assignment_different_sizes)
{
    auto a = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0,2.0 });
    auto b = make_matrix<double>(5, 5, { 0,1,2,3,4 }, { 0,1,2,3,4,5 },
        { 1.0,2.0,3.0,4.0,5.0 });
    b = a;
    EXPECT_EQ(b.n_rows, 2u);
    EXPECT_EQ(b.n_cols, 2u);
    EXPECT_TRUE(b == a);
}

TEST(TestMatrix, copy_assignment_self)
{
    auto a = make_matrix<double>(3, 3, { 0,1,2 }, { 0,1,2,3 }, { 1.0,2.0,3.0 });
    auto& ref = a;
    a = ref;
    EXPECT_EQ(a.nz, 3u);
    EXPECT_DOUBLE_EQ(a.Val[0], 1.0);
    EXPECT_DOUBLE_EQ(a.Val[2], 3.0);
}

TEST(TestMatrix, move_assignment)
{
    auto a = make_matrix<double>(3, 3, { 0,1,2 }, { 0,1,2,3 }, { 1.0,2.0,3.0 });
    auto b = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 7.0,8.0 });

    b = std::move(a);
    EXPECT_EQ(b.n_rows, 3u);
    EXPECT_EQ(b.nz, 3u);
    EXPECT_DOUBLE_EQ(b.Val[2], 3.0);
}

TEST(TestMatrix, can_set_from_vectors)
{
    CSR_matrix<double> m(5, 5, 50);
    std::vector<size_t> cols = { 0, 2, 4, 1, 3 };
    std::vector<size_t> rows = { 0, 2, 2, 4, 5 };
    std::vector<double> vals = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    ASSERT_NO_THROW(m.set_from_vectors(cols, rows, vals));
    EXPECT_EQ(m.nz, 5u);
}

TEST(TestMatrix, set_from_vectors_contents_correct)
{
    auto m = make_matrix<double>(4, 5,
    { 0, 3, 1, 4, 2 },
    { 0, 2, 2, 4, 5 },
    { 1.0, 2.0, 3.0, 4.0, 5.0 });

    EXPECT_EQ(m.RowIndex[0], 0u);
    EXPECT_EQ(m.RowIndex[1], 2u);
    EXPECT_EQ(m.RowIndex[2], 2u);
    EXPECT_EQ(m.RowIndex[3], 4u);
    EXPECT_EQ(m.RowIndex[4], 5u);

    EXPECT_EQ(m.Col[0], 0u);
    EXPECT_EQ(m.Col[1], 3u);
    EXPECT_EQ(m.Col[4], 2u);

    EXPECT_DOUBLE_EQ(m.Val[0], 1.0);
    EXPECT_DOUBLE_EQ(m.Val[4], 5.0);
}

TEST(TestMatrix, set_from_vectors_empty)
{
    CSR_matrix<double> m(3, 3, 5);
    std::vector<size_t> cols = {};
    std::vector<size_t> rows = { 0, 0, 0, 0 };
    std::vector<double> vals = {};
    ASSERT_NO_THROW(m.set_from_vectors(cols, rows, vals));
    EXPECT_EQ(m.nz, 0u);
}

TEST(TestMatrix, can_generate_matrix)
{
    CSR_matrix<double> m(10, 10, 50);
    ASSERT_NO_THROW(m.generate_matrix(3, 0.5, 2.5));
    EXPECT_LE(m.nz, 10u * 10u);
}

TEST(TestMatrix, generate_matrix_values_in_range)
{
    CSR_matrix<double> m(20, 20, 0);
    m.generate_matrix(4, 1.0, 3.0);

    for (size_t k = 0; k < m.nz; ++k) {
        EXPECT_GE(m.Val[k], 1.0);
        EXPECT_LE(m.Val[k], 3.0);
    }
}

TEST(TestMatrix, generate_matrix_columns_sorted_per_row)
{
    CSR_matrix<int> m(15, 15, 0);
    m.generate_matrix(5, 0, 100);

    for (size_t i = 0; i < m.n_rows; ++i) {
        for (size_t k = m.RowIndex[i] + 1; k < m.RowIndex[i + 1]; ++k) {
            EXPECT_LT(m.Col[k - 1], m.Col[k]);
        }
    }
}

TEST(TestMatrix, generate_matrix_row_index_consistent)
{
    CSR_matrix<double> m(12, 12, 0);
    m.generate_matrix(3, 0.0, 1.0);
    EXPECT_EQ(m.RowIndex[0], 0u);
    EXPECT_EQ(m.RowIndex[m.n_rows], m.nz);
    for (size_t i = 0; i < m.n_rows; ++i) {
        EXPECT_LE(m.RowIndex[i], m.RowIndex[i + 1]);
    }
}

TEST(TestMatrix, generate_matrix_throws_if_avg_degree_too_big)
{
    CSR_matrix<double> m(5, 5, 0);
    EXPECT_THROW(m.generate_matrix(6, 0.0, 1.0), std::invalid_argument);
}

TEST(TestMatrix, generate_matrix_int_values_in_range)
{
    CSR_matrix<int> m(10, 10, 0);
    m.generate_matrix(3, 1, 5);
    for (size_t k = 0; k < m.nz; ++k) {
        EXPECT_GE(m.Val[k], 1);
        EXPECT_LE(m.Val[k], 5);
    }
}

TEST(TestMatrix, add_same_structure)
{
    auto a = make_matrix<double>(3, 3, { 0,1,2 }, { 0,1,2,3 }, { 1.0,2.0,3.0 });
    auto b = make_matrix<double>(3, 3, { 0,1,2 }, { 0,1,2,3 }, { 10.0,20.0,30.0 });

    auto c = a + b;
    EXPECT_EQ(c.n_rows, 3u);
    EXPECT_EQ(c.n_cols, 3u);
    EXPECT_EQ(c.nz, 3u);
    EXPECT_DOUBLE_EQ(c.Val[0], 11.0);
    EXPECT_DOUBLE_EQ(c.Val[1], 22.0);
    EXPECT_DOUBLE_EQ(c.Val[2], 33.0);
}

TEST(TestMatrix, add_disjoint_structure)
{
    auto a = make_matrix<double>(2, 3,
    { 0, 2 },
    { 0, 2, 2 },
    { 1.0, 2.0 });
    auto b = CSR_matrix<double>::from_coo(2, 3,
    { 0 },
    { 1 },
    { 5.0 });

    auto c = a + b;
    EXPECT_EQ(c.n_rows, 2u);
    EXPECT_EQ(c.n_cols, 3u);
    EXPECT_EQ(c.nz, 3u);

    EXPECT_EQ(c.RowIndex[0], 0u);
    EXPECT_EQ(c.RowIndex[1], 3u);
    EXPECT_EQ(c.RowIndex[2], 3u);  

    EXPECT_EQ(c.Col[0], 0u);
    EXPECT_EQ(c.Col[1], 1u);
    EXPECT_EQ(c.Col[2], 2u);
    EXPECT_DOUBLE_EQ(c.Val[0], 1.0);
    EXPECT_DOUBLE_EQ(c.Val[1], 5.0);
    EXPECT_DOUBLE_EQ(c.Val[2], 2.0);
}

TEST(TestMatrix, add_partial_overlap)
{
    auto a = make_matrix<double>(1, 4, { 0,2 }, { 0,2 }, { 1.0,2.0 });
    auto b = make_matrix<double>(1, 4, { 2,3 }, { 0,2 }, { 10.0,20.0 });

    auto c = a + b;
    EXPECT_EQ(c.nz, 3u);
    EXPECT_EQ(c.Col[0], 0u);
    EXPECT_EQ(c.Col[1], 2u);
    EXPECT_EQ(c.Col[2], 3u);
    EXPECT_DOUBLE_EQ(c.Val[1], 12.0);
    EXPECT_DOUBLE_EQ(c.Val[2], 20.0);
}

TEST(TestMatrix, add_empty_matrices)
{
    CSR_matrix<double> a(3, 3, 0);
    CSR_matrix<double> b(3, 3, 0);
    auto c = a + b;
    EXPECT_EQ(c.nz, 0u);
    EXPECT_EQ(c.RowIndex[0], 0u);
    EXPECT_EQ(c.RowIndex[3], 0u);
}

TEST(TestMatrix, add_size_mismatch_throws)
{
    auto a = make_matrix<double>(2, 3, { 0 }, { 0,1,1 }, { 1.0 });
    auto b = make_matrix<double>(3, 2, { 0 }, { 0,1,1,1 }, { 1.0 });
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(TestMatrix, add_does_not_modify_operands)
{
    auto a = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0,2.0 });
    auto b = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 3.0,4.0 });
    auto a_copy = a;
    auto b_copy = b;
    auto c = a + b;
    (void)c;
    EXPECT_TRUE(a == a_copy);
    EXPECT_TRUE(b == b_copy);
}

TEST(TestMatrix, multiply_identity_like)
{
    auto m = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0,1.0 });
    std::vector<double> x = { 3.0, 4.0 };
    auto y = m * x;
    ASSERT_EQ(y.size(), 2u);
    EXPECT_DOUBLE_EQ(y[0], 3.0);
    EXPECT_DOUBLE_EQ(y[1], 4.0);
}

TEST(TestMatrix, multiply_general)
{
    auto m = make_matrix<double>(2, 3, { 0,2,1 }, { 0,2,3 }, { 1.0,2.0,3.0 });
    std::vector<double> x = { 1.0, 2.0, 3.0 };
    auto y = m * x;
    ASSERT_EQ(y.size(), 2u);
    EXPECT_DOUBLE_EQ(y[0], 1.0 * 1.0 + 2.0 * 3.0);
    EXPECT_DOUBLE_EQ(y[1], 3.0 * 2.0);
}

TEST(TestMatrix, multiply_empty_rows_give_zero)
{
    auto m = make_matrix<double>(3, 3, { 0 }, { 0,1,1,1 }, { 5.0 });
    std::vector<double> x = { 1.0, 1.0, 1.0 };
    auto y = m * x;
    ASSERT_EQ(y.size(), 3u);
    EXPECT_DOUBLE_EQ(y[0], 5.0);
    EXPECT_DOUBLE_EQ(y[1], 0.0);
    EXPECT_DOUBLE_EQ(y[2], 0.0);
}

TEST(TestMatrix, multiply_size_mismatch_throws)
{
    auto m = make_matrix<double>(2, 3, { 0 }, { 0,1,1 }, { 1.0 });
    std::vector<double> x = { 1.0, 2.0 }; 
    EXPECT_THROW(m * x, std::invalid_argument);
}

TEST(TestMatrix, multiply_int)
{
    auto m = make_matrix<int>(2, 2, { 0,1 }, { 0,1,2 }, { 2,3 });
    std::vector<int> x = { 5, 7 };
    auto y = m * x;
    EXPECT_EQ(y[0], 10);
    EXPECT_EQ(y[1], 21);
}

TEST(TestMatrix, transpose_swaps_dimensions)
{
    auto m = make_matrix<double>(2, 3, { 0,2,1 }, { 0,2,3 }, { 1.0,2.0,3.0 });
    auto t = m.transpose();
    EXPECT_EQ(t.n_rows, 3u);
    EXPECT_EQ(t.n_cols, 2u);
    EXPECT_EQ(t.nz, m.nz);
}

TEST(TestMatrix, transpose_values_correct)
{
    auto m = make_matrix<double>(2, 3, { 0,2,1 }, { 0,2,3 }, { 1.0,2.0,3.0 });
    auto t = m.transpose();

    ASSERT_EQ(t.RowIndex[1] - t.RowIndex[0], 1u);
    EXPECT_EQ(t.Col[t.RowIndex[0]], 0u);
    EXPECT_DOUBLE_EQ(t.Val[t.RowIndex[0]], 1.0);
    ASSERT_EQ(t.RowIndex[2] - t.RowIndex[1], 1u);
    EXPECT_EQ(t.Col[t.RowIndex[1]], 1u);
    EXPECT_DOUBLE_EQ(t.Val[t.RowIndex[1]], 3.0);
    ASSERT_EQ(t.RowIndex[3] - t.RowIndex[2], 1u);
    EXPECT_EQ(t.Col[t.RowIndex[2]], 0u);
    EXPECT_DOUBLE_EQ(t.Val[t.RowIndex[2]], 2.0);
}

TEST(TestMatrix, transpose_twice_returns_original)
{
    auto m = make_matrix<double>(3, 4,
        { 0,3,1,2 }, { 0,2,3,4 }, { 1.0,2.0,3.0,4.0 });
    auto tt = m.transpose().transpose();
    EXPECT_TRUE(tt == m);
}

TEST(TestMatrix, transpose_empty)
{
    CSR_matrix<double> m(3, 5, 0);
    auto t = m.transpose();
    EXPECT_EQ(t.n_rows, 5u);
    EXPECT_EQ(t.n_cols, 3u);
    EXPECT_EQ(t.nz, 0u);
}

TEST(TestMatrix, from_coo_basic)
{
    std::vector<size_t> rows = { 0, 0, 1, 2 };
    std::vector<size_t> cols = { 0, 2, 1, 0 };
    std::vector<double> vals = { 1.0, 2.0, 3.0, 4.0 };

    auto m = CSR_matrix<double>::from_coo(3, 3, rows, cols, vals);
    EXPECT_EQ(m.n_rows, 3u);
    EXPECT_EQ(m.n_cols, 3u);
    EXPECT_EQ(m.nz, 4u);
    EXPECT_EQ(m.RowIndex[0], 0u);
    EXPECT_EQ(m.RowIndex[1], 2u);
    EXPECT_EQ(m.RowIndex[2], 3u);
    EXPECT_EQ(m.RowIndex[3], 4u);
}

TEST(TestMatrix, from_coo_empty)
{
    std::vector<size_t> rows, cols;
    std::vector<double> vals;
    auto m = CSR_matrix<double>::from_coo(3, 3, rows, cols, vals);
    EXPECT_EQ(m.nz, 0u);
    EXPECT_EQ(m.RowIndex[3], 0u);
}

TEST(TestMatrix, from_coo_size_mismatch_throws)
{
    std::vector<size_t> rows = { 0, 1 };
    std::vector<size_t> cols = { 0 };
    std::vector<double> vals = { 1.0, 2.0 };
    EXPECT_THROW(CSR_matrix<double>::from_coo(2, 2, rows, cols, vals),
        std::invalid_argument);
}

TEST(TestMatrix, from_coo_out_of_range_throws)
{
    std::vector<size_t> rows = { 0, 5 };
    std::vector<size_t> cols = { 0, 1 };
    std::vector<double> vals = { 1.0, 2.0 };
    EXPECT_THROW(CSR_matrix<double>::from_coo(3, 3, rows, cols, vals),
        std::out_of_range);

    std::vector<size_t> rows2 = { 0, 1 };
    std::vector<size_t> cols2 = { 0, 5 };
    EXPECT_THROW(CSR_matrix<double>::from_coo(3, 3, rows2, cols2, vals),
        std::out_of_range);
}

TEST(TestMatrix, from_coo_roundtrip_with_multiplication)
{
    auto m = CSR_matrix<double>::from_coo(3, 3,
        { 0,0,1,2,2 }, { 0,2,1,0,2 }, { 1.0,2.0,3.0,4.0,5.0 });
    std::vector<double> x = { 1.0, 1.0, 1.0 };
    auto y = m * x;
    EXPECT_DOUBLE_EQ(y[0], 3.0);
    EXPECT_DOUBLE_EQ(y[1], 3.0);
    EXPECT_DOUBLE_EQ(y[2], 9.0);
}

TEST(TestMatrix, print_coo_no_throw)
{
    auto m = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0,2.0 });
    testing::internal::CaptureStdout();
    ASSERT_NO_THROW(m.print_coo());
    std::string out = testing::internal::GetCapturedStdout();
    EXPECT_NE(out.find("row"), std::string::npos);
    EXPECT_NE(out.find("col"), std::string::npos);
    EXPECT_NE(out.find("val"), std::string::npos);
}

TEST(TestMatrix, print_dense_no_throw)
{
    auto m = make_matrix<double>(2, 3, { 0,2 }, { 0,1,2 }, { 1.0,2.0 });
    testing::internal::CaptureStdout();
    ASSERT_NO_THROW(m.print_dense());
    std::string out = testing::internal::GetCapturedStdout();
    EXPECT_NE(out.find("0"), std::string::npos);
}

TEST(TestMatrix, print_empty_no_throw)
{
    CSR_matrix<double> m(2, 2, 0);
    testing::internal::CaptureStdout();
    ASSERT_NO_THROW(m.print_coo());
    ASSERT_NO_THROW(m.print_dense());
    testing::internal::GetCapturedStdout();
}


TEST(TestMatrix, can1)
{
    CSR_matrix<double> m(5, 5, 5);
    std::vector<size_t> cols = { 0, 2, 4, 1, 3 };
    std::vector<size_t> rows = { 0, 2, 2, 4, 5 };
    std::vector<double> vals = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    m.set_from_vectors(cols, rows, vals);
    EXPECT_EQ(true, m == m);
}

TEST(TestMatrix, can2)
{
    CSR_matrix<double> m(5, 5, 5), m2(5, 5, 5);
    std::vector<size_t> cols = { 0, 2, 4, 1, 3 };
    std::vector<size_t> rows = { 0, 2, 2, 4, 5 };
    std::vector<double> vals = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    m.set_from_vectors(cols, rows, vals);

    vals[0] = 1.0 + DBL_EPSILON * 10;
    m2.set_from_vectors(cols, rows, vals);

    EXPECT_EQ(false, m == m2);
}

TEST(TestMatrix, eq_different_dimensions)
{
    auto a = make_matrix<double>(2, 2, { 0 }, { 0,1,1 }, { 1.0 });
    auto b = make_matrix<double>(3, 2, { 0 }, { 0,1,1,1 }, { 1.0 });
    EXPECT_FALSE(a == b);

    auto c = make_matrix<double>(2, 3, { 0 }, { 0,1,1 }, { 1.0 });
    EXPECT_FALSE(a == c);
}

TEST(TestMatrix, eq_different_nz)
{
    auto a = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0,2.0 });
    auto b = make_matrix<double>(2, 2, { 0 }, { 0,1,1 }, { 1.0 });
    EXPECT_FALSE(a == b);
}

TEST(TestMatrix, eq_different_structure)
{
    auto a = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0,2.0 });
    auto b = make_matrix<double>(2, 2, { 0,0 }, { 0,1,2 }, { 1.0,2.0 });
    EXPECT_FALSE(a == b);
}

TEST(TestMatrix, eq_float_uses_tolerance)
{
    auto a = make_matrix<float>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0f, 2.0f });
    std::vector<float> v = { 1.0f + std::numeric_limits<float>::epsilon(),
                            2.0f };
    auto b = make_matrix<float>(2, 2, { 0,1 }, { 0,1,2 }, v);
    EXPECT_TRUE(a == b);
}

TEST(TestMatrix, eq_double_uses_tolerance)
{
    auto a = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0, 2.0 });
    std::vector<double> v = { 1.0 + std::numeric_limits<double>::epsilon() * 0.5,
                             2.0 };
    auto b = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, v);
    EXPECT_TRUE(a == b);
}

TEST(TestMatrix, eq_double_beyond_tolerance)
{
    auto a = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0, 2.0 });
    std::vector<double> v = { 1.0 + 1e-6, 2.0 };
    auto b = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, v);
    EXPECT_FALSE(a == b);
}

TEST(TestMatrix, eq_empty_matrices)
{
    CSR_matrix<double> a(3, 3, 0);
    CSR_matrix<double> b(3, 3, 0);
    EXPECT_TRUE(a == b);
}

TEST(TestMatrix, eq_int_exact)
{
    auto a = make_matrix<int>(2, 2, { 0,1 }, { 0,1,2 }, { 1, 2 });
    auto b = make_matrix<int>(2, 2, { 0,1 }, { 0,1,2 }, { 1, 2 });
    auto c = make_matrix<int>(2, 2, { 0,1 }, { 0,1,2 }, { 1, 3 });
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST(TestMatrix, sum_then_transpose_then_multiply)
{
    auto a = make_matrix<double>(2, 2, { 0,1,0,1 }, { 0,2,4 }, { 1.0,2.0,3.0,4.0 });
    auto b = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0,1.0 });

    auto c = (a + b).transpose();
    std::vector<double> x = { 1.0, 1.0 };
    auto y = c * x;
    EXPECT_DOUBLE_EQ(y[0], 5.0); 
    EXPECT_DOUBLE_EQ(y[1], 7.0); 
}

TEST(TestMatrix, transpose_preserves_multiplication_dot)
{
    auto A = make_matrix<double>(3, 2,
        { 0,1,0,1,0 }, { 0,2,3,4,5 },
        { 1.0,2.0,3.0,4.0,5.0 });
    std::vector<double> x = { 1.0, 2.0 };
    std::vector<double> y = { 1.0, 1.0, 1.0 };

    auto Ax = A * x;
    auto ATy = A.transpose() * y;

    double lhs = 0, rhs = 0;

    for (size_t i = 0; i < Ax.size(); ++i) lhs += Ax[i] * y[i];
    for (size_t i = 0; i < ATy.size(); ++i) rhs += ATy[i] * x[i];
    EXPECT_NEAR(lhs, rhs, 1e-12);
}

TEST(TestMatrix, copy_assign_then_modify_independent)
{
    auto a = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 1.0,2.0 });
    auto b = make_matrix<double>(2, 2, { 0,1 }, { 0,1,2 }, { 3.0,4.0 });
    b = a;
    b.Val[0] = 100.0;
    EXPECT_DOUBLE_EQ(a.Val[0], 1.0);
    EXPECT_DOUBLE_EQ(b.Val[0], 100.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
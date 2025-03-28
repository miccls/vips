#include <gtest/gtest.h>

#include <format>
#include <ranges>
#include <vector>

#include "../src/lp_model.h"

TEST(ExampleTests, DemonstrateGTest) { EXPECT_EQ(true, true); }

// Test case for add function
TEST(TestLpSolver, TestAddVariable) {
    lp::LpSolver lp_solver;

    // Add two variables
    lp_solver.addVariable();
    lp_solver.addVariable();
    const auto vars = lp_solver.getVars();

    // Check size is right
    EXPECT_EQ(vars.size(), 2);
    // Check id's are right
    for (int id = 0; id < vars.size(); ++id) {
        EXPECT_EQ(id, vars.at(id)->getId());
    }
}

TEST(TestLpSolver, TestVariableToRow) {
    // Assume problem has 3 variables
    const size_t num_variables = 3;
    // Construct two different variables, one on standard form and one general.
    const std::pair<double, double> bounds_standard = {0.0, 10.0};
    const std::pair<double, double> bounds_general = {
        -10, std::numeric_limits<double>::max()};
    const lp::LpVariable var_standard{
        .id = 0, .val = 0, .bounds = bounds_standard};
    const lp::LpVariable var_general{
        .id = 1, .val = 0, .bounds = bounds_general};

    const auto row_standard = var_standard.generateRows(num_variables);
    Eigen::MatrixXd expected_standard_lhs(2, num_variables);
    Eigen::MatrixXd expected_standard_rhs(2, 1);
    expected_standard_lhs << 1, 0, 0, -1, 0, 0;

    expected_standard_rhs << bounds_standard.first, -bounds_standard.second;
    EXPECT_EQ(row_standard.lhs, expected_standard_lhs);
    EXPECT_EQ(row_standard.rhs, expected_standard_rhs);

    const auto row_general = var_general.generateRows(num_variables);
    Eigen::MatrixXd expected_general_lhs(1, num_variables);
    Eigen::MatrixXd expected_general_rhs(1, 1);
    expected_general_lhs << 0, 1, 0;
    expected_general_rhs << bounds_general.first;

    EXPECT_EQ(row_general.lhs, expected_general_lhs);
    EXPECT_EQ(row_general.rhs, expected_general_rhs);
}

TEST(TestLpSolver, TestConstraintWithBounds) {
    // See if it works to set bounds
    lp::LpConstraint lp_constraint_only_lb;
    lp::LpConstraint lp_constraint_only_ub;
    lp::LpConstraint lp_constraint_both;

    const double lb = -1;
    const double ub = 1;

    // lb
    lp_constraint_only_lb.setLb(lb);
    EXPECT_DOUBLE_EQ(lp_constraint_only_lb.bounds.first, lb);
    EXPECT_DOUBLE_EQ(lp_constraint_only_lb.bounds.second,
                     std::numeric_limits<double>::max());

    // ub
    lp_constraint_only_ub.setUb(ub);
    EXPECT_DOUBLE_EQ(lp_constraint_only_ub.bounds.first,
                     std::numeric_limits<double>::lowest());
    EXPECT_DOUBLE_EQ(lp_constraint_only_ub.bounds.second, ub);

    // Both
    lp_constraint_both.setBounds(lb, ub);
    EXPECT_DOUBLE_EQ(lp_constraint_both.bounds.first, lb);
    EXPECT_DOUBLE_EQ(lp_constraint_both.bounds.second, ub);

    // Check throw when bounds are invalid
    ASSERT_THROW(lp_constraint_both.setBounds(ub, lb), std::invalid_argument);
}

TEST(TestLpSolver, TestConstraintRow) {
    lp::LpSolver lp_solver;
    const auto var1 = lp_solver.addVariable();
    const auto var2 = lp_solver.addVariable();

    const double coeff1 = 1.0;
    const double coeff2 = 2.0;

    const double lb = -1.0;
    const double ub = 5.0;

    // Add some vars
    lp::LpConstraint lp_constraint;
    lp_constraint.addVariable(coeff1, var1);
    lp_constraint.addVariable(coeff2, var2);

    // Add some bounds
    lp_constraint.setBounds(lb, ub);

    // 2 is # of vars
    const auto constraint_rows = lp_constraint.generateRows(2);

    Eigen::MatrixXd expected_lhs(2, 2);
    Eigen::MatrixXd expected_rhs(2, 1);
    expected_lhs << coeff1, coeff2, -coeff1, -coeff2;  // | 1,  2 |
                                                       // |-1, -2 |
    expected_rhs << lb, -ub;                           // | -1 |
                                                       // | -5 |
}

TEST(TestLpSolver, TestAddConstraint) {
    lp::LpSolver lp_solver;
    EXPECT_EQ(lp_solver.numConstraintRows(), 0);
    const auto constraint = lp_solver.addConstraint();
    EXPECT_EQ(lp_solver.numConstraintRows(), 0);
    constraint->setUb(10);
    EXPECT_EQ(lp_solver.numConstraintRows(), 1);
    constraint->setLb(0);
    EXPECT_EQ(lp_solver.numConstraintRows(), 2);
}

TEST(TestLpSolver, TestObjective) {
    lp::LpSolver lp_solver;
    auto var1 = lp_solver.addVariable();
    auto var2 = lp_solver.addVariable();

    var1->val = 2;
    var2->val = 4;

    lp_solver.addToObjective(*var1, 1.0);
    lp_solver.addToObjective(*var2, 10.0);

    EXPECT_EQ(lp_solver.evaluateObjective(), 42);
}

TEST(TestLpSolver, TestToStandardForm) {
    /**
     * Convert the problem
     *
     * maximize 1.0 * x_1 + 2.0 * x_2 - 3.0 * x_3
     *      subject to
     *               x_1 +            10.0 * x_3 <= 50
     *                     8.0 * x_2 - 3.0 * x_3 >= 10
     *         2.0 * x_1 - 6.0 * x_2             >= 3
     *               x_1                         >= -10
     *              -x_1                         >= -10
     *                           x_2             >= -1
     *                          -x_2             >= -1
     *                                       x_3 >= -0.1
     *                                      -x_3 >= -0.1
     * to standard frorm
     *
     * The correct answer is obtained by replacing each variable
     * by a difference of non-negative variables: x_i = x_ip - x_im
     *
     * We obtain:
     *
     * maximize 1.0 * x_1p - 1.0 * x_1m + 2.0 * x_2p - 2.0 * x_2m - 3.0 * x_3p
     * + 3.0 * x_3m
     * subject to
     * x_1p - x1_m + 10.0 * x_3p - 10.0 * x_3m <= 50
     * 8.0 * x_2p - 8.0 * x_2m - 3.0 * x_3p + 3.0 * x_3m >= 10
     * 2.0 * x_1p - 2.0 * x_1m - 6.0 * x_2p + 6.0 * x_2m >= 3
     * -x_1m >= -10
     * -x_1p >= -10
     * -x_2m >= -1
     * -x_2p >= -1
     * -x_3m >= -0.1
     * -x_3p >= -0.1
     *  x_1m >= 0
     *  x_1p >= 0
     *  x_2m >= 0
     *  x_2p >= 0
     *  x_3m >= 0
     *  x_3p <= 0
     *
     *  In matrix representation, we have the following:
     *  General:
     *
     *       1   0   10
     *       0   8  -3
     *       2  -6   0
     *      -1   0   0
     *  A = -1   0   0
     *       0  -1   0
     *       0  -1   0
     *       0   0  -1
     *       0   0  -1
     *
     *  b^T = 50 10 3 -10 10 -1 1 -0.1 0.1
     *
     *  c^T = 1  2 -3
     *
     *  Standard:
     *
     *       1  -1   0   0  10 -10
     *       0   0   8  -8  -3   3
     *       2  -2  -6   6   0   0
     *      -1   0   0   0   0   0
     *       0  -1   0   0   0   0
     *      -1   0   0   0   0   0
     *       0   0   0  -1   0   0
     *       0   0  -1   0   0   0
     *  A =  0   0   0   0   0  -1
     *       0   0   0   0  -1   0
     *       1   0   0   0   0   0
     *       0   1   0   0   0   0
     *       0   0   1   0   0   0
     *       0   0   0   1   0   0
     *       0   0   0   0   1   0
     *       0   0   0   0   0   1
     *
     *  b^T = 50 10 3 -10 -10 -1 -1 -0.1 -0.1 0 0 0 0 0 0
     *
     *  c^T = 1 -1 2 -2 3 -3
     *
     * It may be the case that the columns are not in the prescribed order due
     * to the way it will be implemented.
     *
     */

    lp::LpSolver lp_solver;
    // The objective
    std::vector<double> c = {1, 2, -3};
    // Variable bounds
    std::vector<std::pair<double, double>> bounds = {
        {-10, 10}, {-1, 1}, {-0.1, 0.1}};
    // Constraints
    std::vector<std::vector<double>> constraint_rows = {
        {1, 0, 10}, {0, 8, -3}, {2, -6, 0}};
    // Constraint bounds
    std::vector<std::pair<double, double>> constraint_bounds = {
        {std::numeric_limits<double>::lowest(), 50},
        {10, std::numeric_limits<double>::max()},
        {3, std::numeric_limits<double>::max()}};

    // Add the variables
    for (int i = 0; i < 3; ++i) {
        auto var = lp_solver.addVariable();
        var->setBounds(bounds.at(i).first, bounds.at(i).second);
        lp_solver.addToObjective(*var, c.at(i));
    }
    // Add the constraints
    for (int i = 0; i < 3; ++i) {
        auto constraint = lp_solver.addConstraint();
        constraint->setBounds(constraint_bounds.at(i).first,
                              constraint_bounds.at(i).second);
        for (int var_id = 0; var_id < 3; ++var_id) {
            constraint->addVariable(constraint_rows.at(i).at(var_id),
                                    lp_solver.getVars().at(var_id));
        }
    }

    // 6 constraints from the variable bounds
    // 3 constraints from the bounds on the constraints themselves
    EXPECT_EQ(lp_solver.numConstraintRows(), 9);

    const auto mc = lp_solver.getMatrices();

    // Now, lets check the matrices are as expected.
    std::cout << "A = " << *mc.A << std::endl;
    std::cout << "b = " << *mc.b << std::endl;
}